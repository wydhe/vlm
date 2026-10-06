import os
from dataclasses import dataclass

import torch
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    scheduling_policy: str = "mixed"    # "mixed"（每步混合 prefill 分块和 decode）| "prefill_first"（上游的方式）
    piecewise_cudagraph: bool = True    # 混合步 / prefill 步使用以 attention 为界切分的 CUDA Graph
    piecewise_max_tokens: int = 2048    # 录制的最大 token 档位，更大的步走 eager
    mm_config: AutoConfig | None = None  # 完整的多模态配置（Qwen3-VL）；hf_config 是其中的文本部分
    dist_port: int = 2333               # 多卡张量并行时进程间通信的端口；单卡时自动选空闲端口

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.scheduling_policy in ("mixed", "prefill_first")
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.hf_config.model_type == "qwen3_vl":
            assert self.tensor_parallel_size == 1, "Qwen3-VL is single-GPU only"
            self.mm_config = self.hf_config
            self.hf_config = self.hf_config.text_config
            if isinstance(self.hf_config.dtype, str):
                self.hf_config.dtype = getattr(torch, self.hf_config.dtype)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
