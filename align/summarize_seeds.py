"""汇总多个随机种子的 GRPO 训练结果：训练前 / 最好检查点 / 最终，各指标的均值 ± 标准差。

"最好检查点"按评测集上的平均奖励挑选（step 0 除外），和 align/grpo.py 保存 best/ 的规则一致。

用法：
    uv run python -m align.summarize_seeds --runs results/grpo/seed0 results/grpo/seed1 results/grpo/seed2
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

METRICS = ["answer_acc", "citation_acc", "over_refusal", "refusal_recall", "format_rate", "mean_reward"]


def read_evals(run_dir: str) -> list[dict]:
    with open(Path(run_dir) / "eval.jsonl") as f:
        return [json.loads(line) for line in f]


def pick(evals: list[dict]) -> dict[str, dict]:
    """取训练前、最好（按平均奖励）、最终三个时刻的评测结果。"""
    trained = [e for e in evals if e["step"] > 0]
    return {
        "start": evals[0],
        "best": max(trained, key=lambda e: e["mean_reward"]),
        "final": trained[-1],
    }


def mean_std(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def main() -> None:
    p = argparse.ArgumentParser(description="汇总多个种子的 GRPO 结果")
    p.add_argument("--runs", nargs="+", required=True)
    args = p.parse_args()

    picked = [pick(read_evals(run)) for run in args.runs]
    best_steps = [str(x["best"]["step"]) for x in picked]
    print(f"{len(args.runs)} 个种子，最好检查点所在的 step：{', '.join(best_steps)}\n")
    print("| 指标 | 训练前 | 最好检查点 | 最终（step 150） |")
    print("|---|---|---|---|")
    for metric in METRICS:
        cells = [mean_std([x[moment][metric] for x in picked]) for moment in ("start", "best", "final")]
        print(f"| {metric} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
