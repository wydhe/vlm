from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # multimodal (Qwen3-VL): vision features for the image tokens of this step, in token order
    mm_embeds: torch.Tensor | None = None
    mm_deepstack: list[torch.Tensor] | None = None
    # 混合步：前 num_decode_tokens 行是 decode（每条序列 1 个 token），用 decode kernel 单独计算；
    # 此时 context_lens 是这些 decode 序列的长度，cu_seqlens / max_seqlen 只描述后面的 prefill 部分
    num_decode_tokens: int = 0

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, mm_embeds=None, mm_deepstack=None, num_decode_tokens=0):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, mm_embeds, mm_deepstack, num_decode_tokens)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
