import pickle
import socket
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.piecewise_graph import PiecewiseCudaGraphRunner
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_vl import Qwen3VLForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


def find_free_port() -> int:
    """向系统要一个当前空闲的 TCP 端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.piecewise = None  # 由 capture_piecewise_cudagraph() 设置；warmup 在它之前运行

        # 单卡时自动选一个空闲端口，这样一台机器上可以同时跑多个引擎（例如演示界面的两个模型）；
        # 多卡张量并行时各进程必须用同一个端口，取 config.dist_port
        port = config.dist_port if self.world_size > 1 else find_free_port()
        dist.init_process_group("nccl", f"tcp://localhost:{port}", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        if config.mm_config is not None:
            self.model = Qwen3VLForCausalLM(config.mm_config)
        else:
            self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
            if config.piecewise_cudagraph:
                self.capture_piecewise_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
            self.piecewise = None
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        mrope = []                          # 每条序列的 [3, n] M-RoPE 位置（只有 Qwen3-VL 用）
        mm_rows = []                        # (seq.mm, 起始行, 结束行)：这一步要用到的视觉特征行
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            # 混合批里的 decode 序列相当于长度为 1 的分块；张量并行的其他进程只收到 last_token
            input_ids.extend(seq[start:end] if seq.is_prefill else [seq.last_token])
            positions.extend(range(start, end))
            if self.config.mm_config is not None:
                mrope.append(self._mrope_chunk(seq, start, end))
                if seq.mm is not None:
                    p = seq.mm.img_prefix
                    lo, hi = p[min(start, len(p) - 1)], p[min(end, len(p) - 1)]
                    if hi > lo:
                        mm_rows.append((seq.mm, lo, hi))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)

        # 混合步：调度器把 decode 序列（每条 1 个 token）排在最前面，后面是 prefill 分块。
        # attention 分两段算：只要批里有 prefill，varlen kernel 就按 prefill 的查询长度给每条 decode
        # 分配计算块，GQA 打包优化也会失效，实测 bs=224 时 attention 慢 4 倍。
        num_decode = 0
        while num_decode < len(seqs) and not seqs[num_decode].is_prefill:
            num_decode += 1
        decode_lens = None
        if 0 < num_decode < len(seqs):
            decode_lens = [cu_seqlens_k[i + 1] - cu_seqlens_k[i] for i in range(num_decode)]
            prefill_q = [cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(num_decode, len(seqs))]
            prefill_k = [cu_seqlens_k[i + 1] - cu_seqlens_k[i] for i in range(num_decode, len(seqs))]
            cu_seqlens_q = [c - cu_seqlens_q[num_decode] for c in cu_seqlens_q[num_decode:]]
            cu_seqlens_k = [c - cu_seqlens_k[num_decode] for c in cu_seqlens_k[num_decode:]]
            max_seqlen_q, max_seqlen_k = max(prefill_q), max(prefill_k)
            decode_lens = torch.tensor(decode_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        else:
            num_decode = 0

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if mrope:
            pos3 = torch.cat(mrope, dim=1)
            # 纯文本分块的 t == h == w，用一维位置即可（可以走 CUDA Graph）
            positions = pos3[0] if torch.equal(pos3[0], pos3[1]) and torch.equal(pos3[0], pos3[2]) else pos3
            positions = positions.pin_memory().cuda(non_blocking=True)
        else:
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        mm_embeds = mm_deepstack = None
        if mm_rows:
            mm_embeds = torch.cat([mm.embeds[lo:hi] for mm, lo, hi in mm_rows]).cuda(non_blocking=True)
            mm_deepstack = [torch.cat([mm.deepstack[k][lo:hi] for mm, lo, hi in mm_rows]).cuda(non_blocking=True)
                            for k in range(len(mm_rows[0][0].deepstack))]
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, decode_lens,
                    block_tables, mm_embeds, mm_deepstack, num_decode_tokens=num_decode)
        return input_ids, positions

    @staticmethod
    def _mrope_chunk(seq: Sequence, start: int, end: int) -> torch.Tensor:
        """[3, end - start] M-RoPE positions; tokens past the prompt sit at ``i + delta``."""
        # 显式放在 CPU：初始化期间默认设备是 CUDA，而这些张量之后要锁页再拷到 GPU
        if seq.mm is None:
            return torch.arange(start, end, device="cpu").view(1, -1).expand(3, -1)
        n_prompt = seq.mm.positions.size(1)
        parts = []
        if start < n_prompt:
            parts.append(seq.mm.positions[:, start:min(end, n_prompt)])
        if end > n_prompt:
            parts.append(torch.arange(max(start, n_prompt), end, device="cpu").view(1, -1).expand(3, -1)
                         + seq.mm.delta)
        return torch.cat(parts, dim=1)

    def encode_images(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        """跑视觉编码器（Qwen3-VL），返回 (embeds, deepstack)。"""
        return self.model.encode_images(pixel_values, grid_thw)

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1 + (seq.mm.delta if seq.mm is not None else 0))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def capture_piecewise_cudagraph(self):
        """为 varlen 步（混合步 / prefill 步）录制以 attention 为界切分的 CUDA Graph。"""
        cap = min(self.config.piecewise_max_tokens, self.config.max_num_batched_tokens)
        # 档位细一些，补齐浪费才小：512 以内每 32 一档，之后每 128 一档（浪费不超过约 12%）
        candidates = list(range(32, 513, 32)) + list(range(640, 2049, 128))
        sizes = sorted({s for s in candidates if s < cap} | {cap})
        self.piecewise = PiecewiseCudaGraphRunner(self.model, sizes, pool=self.graph_pool)
        self.piecewise.capture()

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if (is_prefill and self.piecewise is not None and self.piecewise.can_run(input_ids.size(0))
                and positions.dim() == 1 and get_context().mm_embeds is None):
            return self.model.compute_logits(self.piecewise.run(input_ids, positions))
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
