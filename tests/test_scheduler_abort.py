"""Scheduler.abort frees KV blocks in every sequence state (CPU only, no model)."""

from types import SimpleNamespace

import pytest
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

BLOCK = 256
NUM_BLOCKS = 8


def make_scheduler(max_batched_tokens: int = 1024) -> Scheduler:
    Sequence.block_size = BLOCK
    cfg = SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=max_batched_tokens, eos=-1,
                          kvcache_block_size=BLOCK, num_kvcache_blocks=NUM_BLOCKS)
    return Scheduler(cfg)


def make_seq(n_tokens: int) -> Sequence:
    return Sequence(list(range(1, n_tokens + 1)), SamplingParams(max_tokens=16))


def free_blocks(s: Scheduler) -> int:
    return len(s.block_manager.free_block_ids)


def test_abort_waiting_before_scheduling():
    s = make_scheduler()
    seq = make_seq(300)
    s.add(seq)
    s.abort(seq)
    assert not s.waiting and not s.running
    assert seq.status == SequenceStatus.FINISHED
    assert free_blocks(s) == NUM_BLOCKS
    assert s.is_finished()


def test_abort_running_frees_blocks():
    s = make_scheduler()
    seq = make_seq(300)
    s.add(seq)
    seqs, is_prefill = s.schedule()
    assert is_prefill and seqs == [seq]
    s.postprocess(seqs, [7], is_prefill)
    assert seq.status == SequenceStatus.RUNNING
    assert free_blocks(s) == NUM_BLOCKS - 2
    s.abort(seq)
    assert not s.running
    assert free_blocks(s) == NUM_BLOCKS
    assert all(b.ref_count == 0 for b in s.block_manager.blocks)


def test_abort_partially_chunk_prefilled():
    s = make_scheduler(max_batched_tokens=BLOCK)
    seq = make_seq(600)                      # needs 3 blocks, prefilled in 3 chunks
    s.add(seq)
    seqs, is_prefill = s.schedule()
    s.postprocess(seqs, [0], is_prefill)
    assert seq.status == SequenceStatus.WAITING and seq.block_table
    assert seq.num_cached_tokens == BLOCK
    s.abort(seq)
    assert not s.waiting
    assert free_blocks(s) == NUM_BLOCKS


def test_abort_is_noop_for_finished():
    s = make_scheduler()
    seq = make_seq(10)
    s.add(seq)
    s.abort(seq)
    s.abort(seq)                             # second abort must not raise
    assert free_blocks(s) == NUM_BLOCKS


def test_other_sequences_unaffected():
    s = make_scheduler()
    a, b = make_seq(300), make_seq(300)
    s.add(a)
    s.add(b)
    seqs, is_prefill = s.schedule()
    s.postprocess(seqs, [1, 2], is_prefill)
    s.abort(a)
    assert list(s.running) == [b]
    assert free_blocks(s) == NUM_BLOCKS - 2
    seqs, is_prefill = s.schedule()
    assert not is_prefill and seqs == [b]


@pytest.mark.parametrize("n", [1, 255, 256, 257])
def test_abort_at_block_boundaries(n: int):
    s = make_scheduler()
    seq = make_seq(n)
    s.add(seq)
    seqs, is_prefill = s.schedule()
    s.postprocess(seqs, [3], is_prefill)
    s.abort(seq)
    assert free_blocks(s) == NUM_BLOCKS
