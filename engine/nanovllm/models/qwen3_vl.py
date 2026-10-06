"""Qwen3-VL for nano-vllm: our paged-KV Qwen3 text decoder + the HF vision tower.

Differences from text-only Qwen3, and how they map onto the engine:

- Vision tower (ViT + patch merger + DeepStack mergers): reused from transformers
  and run once per request when it is added (``ModelRunner.encode_images``).
- Image placeholder tokens: their input embeddings are replaced by vision features.
- DeepStack: features from ViT layers (5, 11, 17) are added to the hidden states of
  LLM layers 0..2 at image-token positions.
- Interleaved M-RoPE: positions are (t, h, w) triples; frequency slots 1::3 / 2::3
  (within ``3 * section``) take the h / w angle. For text tokens t == h == w, so the
  rotation equals plain RoPE at that position: decode steps keep 1-D positions
  (``pos + mrope_delta``) and the existing CUDA graphs.
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
    """Interleaved multimodal RoPE over a plain RoPE cos/sin cache.

    ``positions`` is ``[N]`` (text-only: identical to RoPE) or ``[3, N]`` (t, h, w).
    """

    def __init__(self, base: RotaryEmbedding, mrope_section: list[int]):
        super().__init__()
        self.base = base
        half = base.head_size // 2
        src = torch.zeros(half, dtype=torch.long)          # 0 = t, 1 = h, 2 = w per frequency slot
        for dim in (1, 2):
            src[dim:mrope_section[dim] * 3:3] = dim
        self.register_buffer("src", src, persistent=False)

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        if positions.dim() == 1:
            return self.base(positions, query, key)
        cache = self.base.cos_sin_cache                   # [max_pos, 1, 2 * half] = cat(cos, sin)
        half = cache.size(-1) // 2
        per_axis = cache[positions].squeeze(2)              # [3, N, 2 * half]
        idx = self.src.to(positions.device).view(1, 1, -1).expand(1, per_axis.size(1), -1)
        cos = per_axis[..., :half].gather(0, idx)[0].unsqueeze(1)
        sin = per_axis[..., half:].gather(0, idx)[0].unsqueeze(1)
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)


def mrope_positions(input_ids: list[int], grid_thw: list[tuple[int, int, int]], merge: int,
                    image_token_id: int) -> tuple[torch.Tensor, int]:
    """(t, h, w) position ids for a prompt with images (port of ``get_rope_index``, images only).

    Returns:
        positions ``[3, L]`` (int64, CPU) and ``delta`` such that a later token at
        sequence index ``i`` has position ``i + delta`` on all three axes.
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
    """Qwen3-VL (dense) on nano-vllm. Multimodal inputs arrive through the step context."""

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    # checkpoint prefix -> module prefix (applied before packed-module mapping)
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
        """Run the vision tower: merged image embeddings + per-layer DeepStack features."""
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
                # true hidden = hidden + residual, so adding to either term is equivalent
                hidden[mask] = hidden[mask] + ctx.mm_deepstack[i].to(hidden.dtype)
        hidden, _ = self.model.norm(hidden, residual)
        return hidden

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
