"""混合步 / prefill 步的分段 CUDA Graph（vLLM V1 的做法）。

整张 CUDA Graph 盖不住 varlen 步：attention 用到的 cu_seqlens 和 block table 每一步形状都不同。
但 attention 之间的计算都是逐 token 的，只和（补齐后的）token 数有关。所以在每次 attention 处把模型切开：

    第 0 段       ：embedding → 第 0 层 attention 之前（输出 q, k, v, residual）
    第 0 次 attention：eager 执行（写 KV + 分页 KV 上的 flash_attn_varlen）
    第 i 段       ：第 i-1 层 attention 之后 → 第 i 层 attention 之前
    ...
    第 L 段       ：第 L-1 层 attention 之后 → 最终的 norm（输出 hidden）

每个 token 数档位录制这 L+1 段图。一步有 T 个真实 token 时，补齐到下一个档位；补齐行的
slot_mapping = -1，不会写 KV cache，attention 和 LM head 也只处理真实行。
所有档位、所有分段共用一组按最大档位分配的静态缓冲区（用 [:T] 视图），临时张量都在同一个
graph 内存池里，显存不会随档位数和层数成倍增长。
"""

import gc

import torch
from torch import nn

from nanovllm.utils.context import get_context


class PiecewiseCudaGraphRunner:
    """为 Qwen3 结构的解码器录制、回放"以 attention 为界切开"的 CUDA Graph。"""

    def __init__(self, model: nn.Module, sizes: list[int], pool=None):
        """
        Args:
            model: Qwen3ForCausalLM（需要 model.model 下的 embed_tokens、layers、norm）。
            sizes: 要录制的 token 数档位。
            pool: 可选，和其他 CUDA Graph 共用的显存池。
        """
        inner = model.model
        self.embed = inner.embed_tokens
        self.layers = inner.layers
        self.norm = inner.norm
        self.sizes = sorted(sizes)
        self.max_tokens = self.sizes[-1]
        attn0 = self.layers[0].self_attn
        hidden = self.norm.weight.shape[0]
        nh, nkv, hd = attn0.num_heads, attn0.num_kv_heads, attn0.head_dim
        n = self.max_tokens
        dt = self.norm.weight.dtype

        def buf(*shape: int) -> torch.Tensor:
            return torch.zeros(*shape, dtype=dt, device="cuda")

        self.input_ids = torch.zeros(n, dtype=torch.int64, device="cuda")
        self.positions = torch.zeros(n, dtype=torch.int64, device="cuda")
        self.hidden, self.residual = buf(n, hidden), buf(n, hidden)
        self.q, self.o = buf(n, nh, hd), buf(n, nh, hd)
        self.k, self.v = buf(n, nkv, hd), buf(n, nkv, hd)
        self.graphs: dict[int, list[torch.cuda.CUDAGraph]] = {}
        self.pool = pool

    # ------------------------------------------------------------------ 各段的计算

    def _pre_attn(self, i: int, hidden: torch.Tensor, residual: torch.Tensor | None, t: int) -> None:
        layer = self.layers[i]
        attn = layer.self_attn
        if residual is None:
            h, new_residual = layer.input_layernorm(hidden), hidden
        else:
            h, new_residual = layer.input_layernorm(hidden, residual)
        qkv = attn.qkv_proj(h)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q = q.view(-1, attn.num_heads, attn.head_dim)
        k = k.view(-1, attn.num_kv_heads, attn.head_dim)
        v = v.view(-1, attn.num_kv_heads, attn.head_dim)
        if not attn.qkv_bias:
            q = attn.q_norm(q)
            k = attn.k_norm(k)
        q, k = attn.rotary_emb(self.positions[:t], q, k)
        self.q[:t].copy_(q)
        self.k[:t].copy_(k)
        self.v[:t].copy_(v)
        self.residual[:t].copy_(new_residual)

    def _post_attn(self, i: int, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.layers[i]
        out = layer.self_attn.o_proj(self.o[:t].flatten(1, -1))
        h, residual = layer.post_attention_layernorm(out, self.residual[:t])
        return layer.mlp(h), residual

    def _piece(self, p: int, t: int) -> None:
        """对 t 个（补齐后的）token 执行第 p 段，结果写进静态缓冲区。"""
        if p == 0:
            self._pre_attn(0, self.embed(self.input_ids[:t]), None, t)
        elif p < len(self.layers):
            hidden, residual = self._post_attn(p - 1, t)
            self._pre_attn(p, hidden, residual, t)
        else:
            hidden, residual = self._post_attn(p - 1, t)
            hidden, _ = self.norm(hidden, residual)
            self.hidden[:t].copy_(hidden)

    def _attention(self, i: int, n: int) -> None:
        """对 n 个真实 token 做 eager attention（写 KV + varlen flash-attn）。"""
        o = self.layers[i].self_attn.attn(self.q[:n], self.k[:n], self.v[:n])
        self.o[:n].copy_(o)

    # ------------------------------------------------------------------ 录制与回放

    @torch.inference_mode()
    def capture(self) -> None:
        """录制所有档位。从最大的档位开始，这样共用的显存池只需要按最大需求分配一次。

        直接用 capture_begin / capture_end：torch.cuda.graph 每次进入都会执行
        gc.collect() 和 empty_cache()，要录几百张图、进程里 Python 对象又很多时，启动会非常慢。
        """
        num_pieces = len(self.layers) + 1
        stream = torch.cuda.Stream()
        gc.collect()
        torch.cuda.empty_cache()
        for t in reversed(self.sizes):
            for p in range(num_pieces):          # 预热：让 torch.compile 在录制之前完成编译
                self._piece(p, t)
            torch.cuda.synchronize()
            graphs = []
            with torch.cuda.stream(stream):
                for p in range(num_pieces):
                    g = torch.cuda.CUDAGraph()
                    g.capture_begin(pool=self.pool)
                    self._piece(p, t)
                    g.capture_end()
                    if self.pool is None:
                        self.pool = g.pool()
                    graphs.append(g)
            self.graphs[t] = graphs
            torch.cuda.synchronize()

    def can_run(self, num_tokens: int) -> bool:
        """这一步的 token 数有没有对应的已录制档位。"""
        return num_tokens <= self.max_tokens

    @torch.inference_mode()
    def run(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """前向计算一个 varlen 步（attention 用到的 context 需要事先设置好）。

        Returns:
            真实 token 的最终隐状态，形状 [n, hidden]。
        """
        n = input_ids.size(0)
        t = next(s for s in self.sizes if s >= n)
        self.input_ids[:n].copy_(input_ids)
        self.positions[:n].copy_(positions)
        if t > n:                                # 补齐行：填 token 0、位置 0，不影响结果
            self.input_ids[n:t].zero_()
            self.positions[n:t].zero_()
        assert get_context().slot_mapping.numel() == n
        graphs = self.graphs[t]
        for i in range(len(self.layers)):
            graphs[i].replay()
            self._attention(i, n)
        graphs[-1].replay()
        return self.hidden[:n]
