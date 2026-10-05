"""Piecewise CUDA graphs for mixed / prefill steps (the vLLM V1 approach).

A full CUDA graph cannot cover a varlen step: attention's ``cu_seqlens`` and
block tables change shape every step. Everything *between* attentions is
token-wise, so it only depends on the (padded) token count. We therefore cut
the model at every attention call::

    piece 0      : embed            -> layer0 pre-attn   (q, k, v, residual)
    attention 0  : eager (store KV + flash_attn_varlen over paged KV)
    piece i      : layer i-1 post-attn -> layer i pre-attn
    ...
    piece L      : layer L-1 post-attn -> final norm      (hidden)

and capture the L+1 pieces once per token-count bucket. A step with T real
tokens is padded to the next bucket; padded rows have ``slot_mapping = -1`` so
they never touch the KV cache, and only the real rows reach attention and the
LM head. All pieces of all buckets share one set of max-size static buffers
(views ``[:T]``), so temporaries live in one graph memory pool.
"""

import torch
from torch import nn

from nanovllm.utils.context import get_context


class PiecewiseCudaGraphRunner:
    """Capture and replay attention-split CUDA graphs for a Qwen3-style decoder."""

    def __init__(self, model: nn.Module, sizes: list[int], pool=None):
        """
        Args:
            model: ``Qwen3ForCausalLM`` (needs ``model.model.{embed_tokens, layers, norm}``).
            sizes: Token-count buckets to capture.
            pool: Optional CUDA graph memory pool to share with other graphs.
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

    # ------------------------------------------------------------------ pieces

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
        """Run graph piece ``p`` for ``t`` (padded) tokens; outputs go to static buffers."""
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
        """Eager attention for the ``n`` real tokens (KV write + varlen flash-attn)."""
        o = self.layers[i].self_attn.attn(self.q[:n], self.k[:n], self.v[:n])
        self.o[:n].copy_(o)

    # ------------------------------------------------------------------ capture / run

    @torch.inference_mode()
    def capture(self) -> None:
        """Capture every bucket, largest first so the shared pool is sized once."""
        num_pieces = len(self.layers) + 1
        for t in reversed(self.sizes):
            for p in range(num_pieces):          # warmup: trigger torch.compile outside capture
                self._piece(p, t)
            graphs = []
            for p in range(num_pieces):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, self.pool):
                    self._piece(p, t)
                if self.pool is None:
                    self.pool = g.pool()
                graphs.append(g)
            self.graphs[t] = graphs
            torch.cuda.synchronize()

    def can_run(self, num_tokens: int) -> bool:
        """Whether a step of ``num_tokens`` fits a captured bucket."""
        return num_tokens <= self.max_tokens

    @torch.inference_mode()
    def run(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Forward one varlen step; the attention context must already be set.

        Returns:
            Final hidden states for the real tokens, shape ``[n, hidden]``.
        """
        n = input_ids.size(0)
        t = next(s for s in self.sizes if s >= n)
        self.input_ids[:n].copy_(input_ids)
        self.positions[:n].copy_(positions)
        if t > n:                                # padded rows: harmless token 0 at position 0
            self.input_ids[n:t].zero_()
            self.positions[n:t].zero_()
        assert get_context().slot_mapping.numel() == n
        graphs = self.graphs[t]
        for i in range(len(self.layers)):
            graphs[i].replay()
            self._attention(i, n)
        graphs[-1].replay()
        return self.hidden[:n]
