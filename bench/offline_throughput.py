"""nano-vllm 引擎的离线吞吐测试。

负载和上游 engine/bench.py 相同（固定种子的随机 prompt，输入和输出长度都在 [100, 1024] 均匀分布），
可以和上游 README 的数字直接对比；模型路径改成从命令行或环境变量读取，并把结果和运行环境存成 JSON。

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
    """一次测试：负载、运行环境和测得的吞吐。"""

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
    """返回仓库当前提交的短哈希，取不到时返回 "unknown"。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(model: str, num_seqs: int, max_input_len: int, max_output_len: int,
        enforce_eager: bool) -> BenchResult:
    """跑一次离线生成，测量输出 token 的吞吐。

    Args:
        model: 本地模型目录（nano-vllm 支持的 Hugging Face 格式）。
        num_seqs: 一次性提交的请求数。
        max_input_len: 随机 prompt 长度的上限（下限 100）。
        max_output_len: 强制输出长度的上限（下限 100）。
        enforce_eager: 为 True 时关闭 CUDA Graph。

    Returns:
        测量结果 BenchResult。
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
    """解析命令行参数，跑测试，保存 JSON 结果。"""
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
