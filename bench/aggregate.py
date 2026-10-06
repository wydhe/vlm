"""汇总重复压测：把同一个配置的多次运行算成"均值 ± 标准差"。

bench/serving.py 每次运行保存一个 results/serving/<label>_<时间>.json。
标签约定为 <配置名>-rr<速率>-rep<次序>，这里按去掉 "-rep<次序>" 之后的部分分组。

用法：
    uv run python -m bench.aggregate --prefix rep-
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

# (显示名, 取值路径)
METRICS = [
    ("输出 tok/s", ("output_throughput",)),
    ("TTFT P99", ("ttft_ms", "p99")),
    ("TPOT 均值", ("tpot_ms", "mean")),
    ("TPOT P99", ("tpot_ms", "p99")),
    ("ITL P90", ("itl_ms", "p90")),
    ("ITL P99", ("itl_ms", "p99")),
]


def read_metric(summary: dict, path: tuple[str, ...]) -> float:
    value = summary
    for key in path:
        value = value[key]
    return value


def mean_std(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.mean(values):.1f} ± {statistics.stdev(values):.1f}"


def main() -> None:
    p = argparse.ArgumentParser(description="汇总重复压测结果")
    p.add_argument("--dir", default="results/serving")
    p.add_argument("--prefix", default="", help="只统计标签以此开头的运行")
    args = p.parse_args()

    groups: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(Path(args.dir).glob(f"{args.prefix}*.json")):
        record = json.loads(path.read_text())
        group = re.sub(r"-rep\d+$", "", record["label"])
        groups[group].append(record["summary"])

    header = "| 配置 | 次数 | " + " | ".join(name for name, _ in METRICS) + " |"
    print(header)
    print("|" + "---|" * (len(METRICS) + 2))
    for group, summaries in sorted(groups.items()):
        cells = [mean_std([read_metric(s, path) for s in summaries]) for _, path in METRICS]
        print(f"| {group} | {len(summaries)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
