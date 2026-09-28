"""Per-step latency: decode-only (CUDA graph) steps vs mixed steps with a prefill chunk.

Holds ``bs`` sequences in steady decode, then times single engine steps that add
one fresh prompt of a given length (random tokens, so no prefix-cache hits).
Separates the cost of "a mixed step runs eager" from the prefill compute itself.

Usage:
    uv run python -m bench.step_cost --model weights/Qwen3-8B
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import time

import torch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


def timed_step(engine: LLMEngine) -> float:
    """Run one engine step and return its wall time in ms (GPU-synchronized)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    engine.step_updates()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000


def reset(engine: LLMEngine) -> None:
    """Drop all sequences and return every KV block to the free pool."""
    sched = engine.scheduler
    for seq in list(sched.running) + list(sched.waiting):
        sched.abort(seq)


def main() -> None:
    """Measure median step latency for several decode batch sizes and prefill lengths."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=os.environ.get("MODEL_DIR"))
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 128, 224])
    p.add_argument("--prefill-lens", type=int, nargs="+", default=[1, 32, 256, 1024])
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--repeats", type=int, default=7)
    args = p.parse_args()
    if not args.model:
        p.error("pass --model or set MODEL_DIR")

    rng = random.Random(0)
    engine = LLMEngine(args.model, max_num_batched_tokens=args.max_num_batched_tokens,
                       max_num_seqs=256, max_model_len=4096)
    long_run = SamplingParams(temperature=0, max_tokens=3000, ignore_eos=True)
    one_token = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)

    def rand_prompt(n: int) -> list[int]:
        return [rng.randint(1000, 20000) for _ in range(n)]

    print(f"{'decode bs':>9} | {'decode-only':>11} | " +
          " | ".join(f"{'+prefill ' + str(n):>13}" for n in args.prefill_lens))
    for bs in args.batch_sizes:
        reset(engine)
        for _ in range(bs):
            engine.add_request(rand_prompt(300), long_run)
        while engine.scheduler.waiting:
            timed_step(engine)
        for _ in range(5):
            timed_step(engine)
        decode_ms = statistics.median(timed_step(engine) for _ in range(2 * args.repeats))
        mixed_ms = []
        for n in args.prefill_lens:
            samples = []
            for _ in range(args.repeats):
                engine.add_request(rand_prompt(n), one_token)
                samples.append(timed_step(engine))
            mixed_ms.append(statistics.median(samples))
        print(f"{bs:>9} | {decode_ms:>9.1f}ms | " +
              " | ".join(f"{m:>7.1f}ms {m / decode_ms:>4.1f}x" for m in mixed_ms))


if __name__ == "__main__":
    main()
