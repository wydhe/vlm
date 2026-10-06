"""Qwen3-VL 接入 nano-vllm：文本部分用本项目的分页 KV Qwen3 解码器，视觉部分复用 HF 的视觉编码器。

和纯文本 Qwen3 的区别，以及在引擎里的对应做法：

- 视觉编码器（ViT + patch merger + DeepStack merger）：直接复用 transformers 的实现，
  每个请求加入时编码一次（ModelRunner.encode_images）。
- 图像占位 token：它们的输入 embedding 换成视觉特征。
- DeepStack：ViT 第 5、11、17 层的特征，分别加到语言模型第 0、1、2 层输出的图像 token 位置上。
- 交错 M-RoPE：位置是 (t, h, w) 三元组；频率槽 1::3 / 2::3（前 3×section 个）取 h / w 的角度。
  文本 token 的 t = h = w，这时旋转结果和普通 RoPE 完全一样，所以 decode 步继续用一维位置
  （pos + mrope_delta），原有的 CUDA Graph 不用改。
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from nanovllm.layers.embed_head import ParallelLMHead
from nanovllm.layers.rotary_embedding import RotaryEmbedding, apply_rotary_emb
from nanovllm.models.qwen3 import Qwen3Model
from nanovllm.utils.context import get_context


class MRotaryEmbedding(nn.Module):
    """在普通 RoPE 的 cos/sin 缓存之上实现交错 M-RoPE。

    positions 是 [N]（纯文本，和普通 RoPE 相同）或 [3, N]（t, h, w 三个分量）。
    """

    def __init__(self, base: RotaryEmbedding, mrope_section: list[int]):
        super().__init__()
        self.base = base
        half = base.head_size // 2
        src = torch.zeros(half, dtype=torch.long)          # 每个频率槽取哪个分量：0 = t，1 = h，2 = w
        for dim in (1, 2):
            src[dim:mrope_section[dim] * 3:3] = dim
        self.register_buffer("src", src, persistent=False)

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        if positions.dim() == 1:
            return self.base(positions, query, key)
        cache = self.base.cos_sin_cache                   # [max_pos, 1, 2 * half]，前一半是 cos，后一半是 sin
        half = cache.size(-1) // 2
        per_axis = cache[positions].squeeze(2)              # [3, N, 2 * half]
        idx = self.src.to(positions.device).view(1, 1, -1).expand(1, per_axis.size(1), -1)
        cos = per_axis[..., :half].gather(0, idx)[0].unsqueeze(1)
        sin = per_axis[..., half:].gather(0, idx)[0].unsqueeze(1)
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)


def mrope_positions(input_ids: list[int], grid_thw: list[tuple[int, int, int]], merge: int,
                    image_token_id: int) -> tuple[torch.Tensor, int]:
    """计算含图片 prompt 的 (t, h, w) 位置（移植自 HF 的 get_rope_index，只处理图片）。

    Returns:
        positions：[3, L] 的位置（int64，CPU）；
        delta：prompt 之后第 i 个 token 的位置是 i + delta（三个分量相同）。
    """
    parts: list[torch.Tensor] = []
    st, nxt = 0, 0
    for t, h, w in grid_thw:
        ed = input_ids.index(image_token_id, st)
        if ed > st:
            parts.append(torch.arange(ed - st).view(1, -1).expand(3, -1) + nxt)
            nxt += ed - st
        gh, gw = h // merge, w // merge
        ti = torch.arange(t).view(-1, 1).expand(-1, gh * gw).flatten()
        hi = torch.arange(gh).view(1, -1, 1).expand(t, -1, gw).flatten()
        wi = torch.arange(gw).view(1, 1, -1).expand(t, gh, -1).flatten()
        parts.append(torch.stack([ti, hi, wi]) + nxt)
        nxt = int(parts[-1].max()) + 1
        st = ed + t * gh * gw
    if st < len(input_ids):
        parts.append(torch.arange(len(input_ids) - st).view(1, -1).expand(3, -1) + nxt)
    pos = torch.cat(parts, dim=1)
    return pos, int(pos.max()) + 1 - len(input_ids)


class Qwen3VLForCausalLM(nn.Module):
    """nano-vllm 上的 Qwen3-VL（稠密版）。图像输入通过每一步的 context 传进来。"""

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    # 权重文件里的前缀 → 本模块里的前缀（在 packed_modules_mapping 之前应用）
    hf_name_mapping = (("model.language_model.", "model."), ("model.visual.", "visual."))

    def __init__(self, config) -> None:
        super().__init__()
        text = config.text_config
        self.image_token_id = config.image_token_id
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.model = Qwen3Model(text)
        self.lm_head = ParallelLMHead(text.vocab_size, text.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        section = text.rope_scaling["mrope_section"]
        for layer in self.model.layers:
            attn = layer.self_attn
            attn.rotary_emb = MRotaryEmbedding(attn.rotary_emb, section)
        # flash_attention_2：多张图拼成一批时按 cu_seqlens 分段算 attention；
        # sdpa 会构造覆盖整批的块对角掩码，计算量随总 patch 数平方增长，批量编码反而更慢
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config,
                                                      attn_implementation="flash_attention_2")

    @torch.inference_mode()
    def encode_images(self, pixel_values: torch.Tensor,
                      grid_thw: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """跑视觉编码器：返回合并后的图像特征，以及各层的 DeepStack 特征。"""
        dtype = self.model.embed_tokens.weight.dtype
        embeds, deepstack = self.visual(pixel_values.to("cuda", dtype), grid_thw=grid_thw.to("cuda"))
        return embeds, deepstack

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        hidden = self.model.embed_tokens(input_ids)
        mask = None
        if ctx.mm_embeds is not None:
            mask = input_ids == self.image_token_id
            hidden[mask] = ctx.mm_embeds.to(hidden.dtype)
        residual = None
        for i, layer in enumerate(self.model.layers):
            hidden, residual = layer(positions, hidden, residual)
            if mask is not None and i < len(ctx.mm_deepstack):
                # 真正的隐状态 = hidden + residual，所以加到哪一项上都一样
                hidden[mask] = hidden[mask] + ctx.mm_deepstack[i].to(hidden.dtype)
        hidden, _ = self.model.norm(hidden, residual)
        return hidden

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
