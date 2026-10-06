from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.policy = getattr(config, "scheduling_policy", "mixed")
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """挑出下一步要跑的序列。返回 (seqs, is_prefill)；is_prefill 为 True 表示这一步走 varlen 路径。"""
        if self.policy == "mixed":
            return self._schedule_mixed()
        return self._schedule_prefill_first()

    def _schedule_mixed(self) -> tuple[list[Sequence], bool]:
        """在一个 token 预算内混合调度 prefill 和 decode（vLLM V1 / Sarathi-Serve 的做法）。

        1. 先给每条正在运行的序列安排 1 个 decode token，新来的 prompt 不会打断正在生成的请求。
        2. 剩下的预算用来 prefill 等待中的序列，最后一条放不下就分块。
        decode 可以看成"查询长度为 1 的分块"，和 prefill 一起放进同一次 varlen 前向。
        """
        scheduled_seqs = []
        budget = self.max_num_batched_tokens

        # 1. decode: running sequences (preempting from the tail when out of blocks)
        num_running = len(self.running)
        for _ in range(num_running):
            if not self.running or len(scheduled_seqs) >= self.max_num_seqs or budget == 0:
                break
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
                budget -= 1
        num_decode = len(scheduled_seqs)
        # 这一步排上的 decode 序列按原顺序放回队首，排在这一步没轮到的序列前面
        self.running.extendleft(reversed(scheduled_seqs))

        # 2. prefill: waiting sequences (including a partially prefilled head) with what is left
        while self.waiting and budget > 0 and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(seq.num_tokens - seq.num_cached_tokens, budget)
            budget -= seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        assert scheduled_seqs or not self.running, "running sequences but nothing scheduled"
        return scheduled_seqs, len(scheduled_seqs) > num_decode

    def _schedule_prefill_first(self) -> tuple[list[Sequence], bool]:
        """上游 nano-vllm 的调度方式：一步要么全是 prefill，要么全是 decode。"""
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def abort(self, seq: Sequence):
        """取消一条还没结束的序列（例如客户端断开），并释放它的 KV 块。

        只能在两步之间调用，不能在一个批正在计算时调用。
        """
        if seq.is_finished:
            return
        if seq.status == SequenceStatus.WAITING:
            # 分块 prefill 做到一半的序列状态仍是 WAITING，但已经占了 KV 块
            self.waiting.remove(seq)
        else:
            self.running.remove(seq)
        if seq.block_table:
            self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.FINISHED
        seq.mm = None

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                seq.mm = None           # 释放视觉特征
