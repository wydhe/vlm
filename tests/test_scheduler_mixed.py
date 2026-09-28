"""Mixed (stall-free) scheduling invariants (CPU only, no model)."""

from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

BLOCK = 256


def make_scheduler(budget: int, num_blocks: int = 64, max_num_seqs: int = 16,
                   policy: str = "mixed") -> Scheduler:
    Sequence.block_size = BLOCK
    cfg = SimpleNamespace(max_num_seqs=max_num_seqs, max_num_batched_tokens=budget, eos=-1,
                          kvcache_block_size=BLOCK, num_kvcache_blocks=num_blocks,
                          scheduling_policy=policy)
    return Scheduler(cfg)


def make_seq(n_tokens: int, max_tokens: int = 64) -> Sequence:
    return Sequence(list(range(1, n_tokens + 1)), SamplingParams(max_tokens=max_tokens))


def run_step(s: Scheduler) -> tuple[list[Sequence], bool]:
    """schedule + postprocess with a dummy sampled token."""
    seqs, is_prefill = s.schedule()
    s.postprocess(seqs, [42] * len(seqs), is_prefill)
    return seqs, is_prefill


def start_running(s: Scheduler, n: int) -> list[Sequence]:
    seqs = [make_seq(100) for _ in range(n)]
    for q in seqs:
        s.add(q)
    for _ in range(4):                                  # prompts may be chunked if budget < total
        run_step(s)
        if all(q.status == SequenceStatus.RUNNING for q in seqs):
            break
    assert all(q.status == SequenceStatus.RUNNING for q in seqs)
    return seqs


def test_decodes_not_stalled_by_arriving_prompt():
    s = make_scheduler(budget=512)
    running = start_running(s, 4)
    long_prompt = make_seq(2000)
    s.add(long_prompt)
    seqs, is_prefill = s.schedule()
    assert is_prefill                                   # varlen path
    assert seqs[:4] == running                          # every decode kept its slot
    assert all(q.num_scheduled_tokens == 1 for q in running)
    assert long_prompt.num_scheduled_tokens == 512 - 4  # prefill gets the leftover budget


def test_prefill_first_stalls_decodes():
    s = make_scheduler(budget=512, policy="prefill_first")
    running = start_running(s, 4)
    s.add(make_seq(2000))
    seqs, is_prefill = s.schedule()
    assert is_prefill and not set(seqs) & set(running)  # the behaviour we are fixing


def test_chunked_prompt_resumes_across_steps():
    s = make_scheduler(budget=300)
    start_running(s, 2)
    prompt = make_seq(1000)
    s.add(prompt)
    progress = []
    for _ in range(10):
        run_step(s)
        progress.append(prompt.num_cached_tokens)
        if prompt.status == SequenceStatus.RUNNING:
            break
    assert progress[:3] == [298, 596, 894]              # 300 - 2 decodes per step
    assert prompt.status == SequenceStatus.RUNNING
    assert prompt.num_completion_tokens == 1            # token sampled once the prompt completed


def test_scheduled_tokens_within_budget():
    s = make_scheduler(budget=256)
    start_running(s, 3)
    for n in (700, 50, 400):
        s.add(make_seq(n))
    for _ in range(8):
        seqs, is_prefill = s.schedule()
        assert sum(q.num_scheduled_tokens for q in seqs) <= 256
        s.postprocess(seqs, [1] * len(seqs), is_prefill)


def test_pure_decode_step_uses_graph_path():
    s = make_scheduler(budget=512)
    start_running(s, 3)
    _, is_prefill = s.schedule()
    assert not is_prefill                               # decode-only batch -> CUDA graph path


def test_blocks_balanced_after_all_finish():
    s = make_scheduler(budget=256, num_blocks=16)
    for n in (300, 700, 20, 256):
        s.add(make_seq(n, max_tokens=5))
    for _ in range(200):
        if s.is_finished():
            break
        run_step(s)
    assert s.is_finished()
    assert len(s.block_manager.free_block_ids) == 16
