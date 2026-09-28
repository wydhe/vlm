"""Offline throughput benchmark for the nano-vllm engine.

Reproduces the workload of upstream ``engine/bench.py`` (seeded random prompts,
input/output lengths uniform in [100, 1024]) so numbers are comparable with the
upstream README, but takes the model path from CLI/env instead of a hardcoded
path and writes a JSON record with the run environment.

Usage:
    MODEL_DIR=weights/Qwen3-0.6B uv run python -m bench.offline_throughput
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from random import randint, seed

import torch
from nanovllm import LLM, SamplingParams


@dataclass
class BenchResult:
    """One benchmark run: workload, environment and measured throughput."""

    model: str
    num_seqs: int
    max_input_len: int
    max_output_len: int
    enforce_eager: bool
    total_output_tokens: int
    elapsed_s: float
    throughput_tok_s: float
    gpu: str
    torch_version: str
    git_commit: str


def git_commit() -> str:
    """Return the short HEAD commit of the repository, or ``"unknown"``."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(model: str, num_seqs: int, max_input_len: int, max_output_len: int,
        enforce_eager: bool) -> BenchResult:
    """Run one offline generation pass and measure output-token throughput.

    Args:
        model: Local directory of a Hugging Face checkpoint supported by nano-vllm.
        num_seqs: Number of requests submitted at once.
        max_input_len: Upper bound of the random prompt length (lower bound 100).
        max_output_len: Upper bound of the forced output length (lower bound 100).
        enforce_eager: Disable CUDA graphs when True.

    Returns:
        The measured ``BenchResult``.
    """
    seed(0)
    llm = LLM(model, enforce_eager=enforce_eager, max_model_len=4096)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len))
        for _ in range(num_seqs)
    ]

    llm.generate(["Benchmark: "], SamplingParams())  # warmup
    start = time.perf_counter()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - start

    total = sum(sp.max_tokens for sp in sampling_params)
    return BenchResult(
        model=os.path.basename(os.path.normpath(model)),
        num_seqs=num_seqs,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        enforce_eager=enforce_eager,
        total_output_tokens=total,
        elapsed_s=round(elapsed, 3),
        throughput_tok_s=round(total / elapsed, 2),
        gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        git_commit=git_commit(),
    )


def main() -> None:
    """Parse CLI arguments, run the benchmark and save the JSON record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.environ.get("MODEL_DIR"),
                        help="checkpoint dir (default: $MODEL_DIR)")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--out-dir", default="results/bench")
    args = parser.parse_args()
    if not args.model:
        parser.error("pass --model or set MODEL_DIR")

    result = run(args.model, args.num_seqs, args.max_input_len, args.max_output_len,
                 args.enforce_eager)
    print(json.dumps(asdict(result), indent=2))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"offline_{result.model}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_file.write_text(json.dumps(asdict(result), indent=2))
    print(f"saved -> {out_file}")


if __name__ == "__main__":
    main()
