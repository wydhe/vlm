"""在线服务压测：按一定的请求到达过程发请求，统计 TTFT / TPOT / ITL / E2E 延迟。

适用于任何支持 token id 形式 prompt、ignore_eos 和 stream_options.include_usage 的
OpenAI 兼容 /v1/completions 接口——本项目的服务和官方 vLLM 都支持，所以可以用完全相同的
请求对比两者。prompt 直接发 token id，排除两边 tokenizer 不同带来的差异。

Usage:
    uv run python -m bench.serving --tokenizer weights/Qwen3-8B --dataset sharegpt \
        --num-prompts 500 --request-rate 8 --label nano-8b-rr8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

SHAREGPT_FILE = "data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"


@dataclass
class Sample:
    """一条压测请求。"""

    prompt_ids: list[int]
    output_len: int


@dataclass
class RequestResult:
    """客户端测到的一条请求的数据。"""

    success: bool = False
    prompt_len: int = 0
    output_len: int = 0          # 服务端报告的生成 token 数
    ttft: float = 0.0            # 秒，发出请求 → 收到第一个分片
    e2e: float = 0.0             # 秒，发出请求 → 收到最后一个分片
    itl: list[float] = field(default_factory=list)  # 秒，相邻分片之间的间隔
    error: str = ""


# ----------------------------------------------------------------------------- datasets

def sample_sharegpt(path: str, n: int, tokenizer: PreTrainedTokenizerBase, rng: random.Random,
                    max_prompt_len: int = 1024, max_total_len: int = 2048) -> list[Sample]:
    """取对话中用户的第一句作为 prompt，模型第一次回复的长度作为输出长度。

    过滤规则和 vLLM 的 benchmark_serving 默认值一致，方便和别人的结果比较。
    """
    with open(path) as f:
        convs = [d["conversations"] for d in json.load(f) if len(d.get("conversations", [])) >= 2]
    rng.shuffle(convs)
    samples: list[Sample] = []
    for conv in convs:
        if len(samples) == n:
            break
        prompt_ids = tokenizer.encode(conv[0]["value"])
        output_len = len(tokenizer.encode(conv[1]["value"]))
        if len(prompt_ids) < 4 or output_len < 4:
            continue
        if len(prompt_ids) > max_prompt_len or len(prompt_ids) + output_len > max_total_len:
            continue
        samples.append(Sample(prompt_ids, output_len))
    if len(samples) < n:
        raise ValueError(f"only {len(samples)} usable ShareGPT samples, asked for {n}")
    return samples


def sample_random(n: int, input_len: int, output_len: int, rng: random.Random,
                  range_ratio: float = 0.0) -> list[Sample]:
    """随机 token id；长度在 [len × (1 - ratio), len] 之间均匀分布。"""
    def draw(length: int) -> int:
        return rng.randint(max(1, int(length * (1 - range_ratio))), length)
    return [Sample([rng.randint(1000, 20000) for _ in range(draw(input_len))], draw(output_len))
            for _ in range(n)]


# ----------------------------------------------------------------------------- client

async def send(session: aiohttp.ClientSession, url: str, model: str, sample: Sample,
               temperature: float) -> RequestResult:
    """发一个流式请求，记录每个分片到达的时间。"""
    payload = {"model": model, "prompt": sample.prompt_ids, "max_tokens": sample.output_len,
               "temperature": temperature, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}}
    res = RequestResult(prompt_len=len(sample.prompt_ids))
    start = last = time.perf_counter()
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                res.error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return res
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    res.error = str(chunk["error"])
                    return res
                now = time.perf_counter()
                if chunk.get("choices"):
                    if res.ttft == 0.0:
                        res.ttft = now - start
                    else:
                        res.itl.append(now - last)
                    last = now
                if chunk.get("usage"):
                    res.output_len = chunk["usage"]["completion_tokens"]
        res.e2e = last - start
        res.success = res.output_len > 0
        if not res.success:
            res.error = "no usage in stream"
    except aiohttp.ClientError as e:
        res.error = f"{type(e).__name__}: {e}"
    return res


async def run_load(base_url: str, model: str, samples: list[Sample], request_rate: float,
                   max_concurrency: int | None, temperature: float, seed: int
                   ) -> tuple[list[RequestResult], float]:
    """按泊松过程的间隔发请求（速率为 inf 时一次性全部发出）。"""
    url = base_url.rstrip("/") + "/v1/completions"
    rng = random.Random(seed)
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    timeout = aiohttp.ClientTimeout(total=6 * 3600)
    async with aiohttp.ClientSession(timeout=timeout,
                                     connector=aiohttp.TCPConnector(limit=0)) as session:
        # 预热：先并发发一批请求，让一次性的 JIT 编译（例如 torch.compile 遇到新的批大小时重新编译）
        # 发生在正式计时之前，不要落进 P99
        await asyncio.gather(*(send(session, url, model, Sample(s.prompt_ids[:64], 16), temperature)
                               for s in samples[:16]))

        async def guarded(s: Sample) -> RequestResult:
            if sem is None:
                return await send(session, url, model, s, temperature)
            async with sem:
                return await send(session, url, model, s, temperature)

        tasks = []
        start = time.perf_counter()
        for s in samples:
            tasks.append(asyncio.create_task(guarded(s)))
            if request_rate != float("inf"):
                await asyncio.sleep(rng.expovariate(request_rate))
        results = await asyncio.gather(*tasks)
        return results, time.perf_counter() - start


# ----------------------------------------------------------------------------- metrics

def _stats(values: list[float], scale: float = 1000.0) -> dict[str, float]:
    if not values:
        return {k: float("nan") for k in ("mean", "median", "p90", "p99")}
    a = np.asarray(values) * scale
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99))}


def summarize(results: list[RequestResult], duration: float) -> dict:
    """汇总所有请求的结果，延迟单位是毫秒。"""
    ok = [r for r in results if r.success]
    out_tokens = sum(r.output_len for r in ok)
    in_tokens = sum(r.prompt_len for r in ok)
    tpot = [(r.e2e - r.ttft) / (r.output_len - 1) for r in ok if r.output_len > 1]
    return {
        "completed": len(ok),
        "failed": len(results) - len(ok),
        "errors": sorted({r.error for r in results if r.error})[:5],
        "duration_s": duration,
        "request_throughput": len(ok) / duration,
        "output_throughput": out_tokens / duration,
        "total_token_throughput": (in_tokens + out_tokens) / duration,
        "total_input_tokens": in_tokens,
        "total_output_tokens": out_tokens,
        "ttft_ms": _stats([r.ttft for r in ok]),
        "tpot_ms": _stats(tpot),
        "itl_ms": _stats([x for r in ok for x in r.itl]),
        "e2e_ms": _stats([r.e2e for r in ok]),
    }


def print_summary(s: dict) -> None:
    """打印一张简洁的结果表。"""
    print(f"completed {s['completed']}  failed {s['failed']}  duration {s['duration_s']:.1f}s")
    print(f"req/s {s['request_throughput']:.2f}   out tok/s {s['output_throughput']:.1f}   "
          f"total tok/s {s['total_token_throughput']:.1f}")
    print(f"{'ms':>6} {'mean':>9} {'median':>9} {'p90':>9} {'p99':>9}")
    for key in ("ttft_ms", "tpot_ms", "itl_ms", "e2e_ms"):
        v = s[key]
        print(f"{key[:-3]:>6} {v['mean']:9.1f} {v['median']:9.1f} {v['p90']:9.1f} {v['p99']:9.1f}")
    if s["errors"]:
        print("errors:", *s["errors"], sep="\n  ")


def main() -> None:
    """解析命令行参数，跑压测，保存 JSON 结果。"""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default=None, help="model field sent to the server (default: tokenizer dir name)")
    p.add_argument("--tokenizer", default=os.environ.get("MODEL_DIR"), help="default: $MODEL_DIR")
    p.add_argument("--dataset", choices=["sharegpt", "random"], default="sharegpt")
    p.add_argument("--dataset-path", default=SHAREGPT_FILE)
    p.add_argument("--random-input-len", type=int, default=1024)
    p.add_argument("--random-output-len", type=int, default=128)
    p.add_argument("--random-range-ratio", type=float, default=0.0)
    p.add_argument("--num-prompts", type=int, default=500)
    p.add_argument("--request-rate", type=float, default=float("inf"), help="req/s, Poisson; inf = burst")
    p.add_argument("--max-concurrency", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="run")
    p.add_argument("--out-dir", default="results/serving")
    args = p.parse_args()
    if not args.tokenizer:
        p.error("pass --tokenizer or set MODEL_DIR")

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.dataset == "sharegpt":
        samples = sample_sharegpt(args.dataset_path, args.num_prompts, tokenizer, rng)
    else:
        samples = sample_random(args.num_prompts, args.random_input_len, args.random_output_len,
                                rng, args.random_range_ratio)
    model = args.model or os.path.basename(os.path.normpath(args.tokenizer))

    results, duration = asyncio.run(run_load(args.base_url, model, samples, args.request_rate,
                                             args.max_concurrency, args.temperature, args.seed))
    summary = summarize(results, duration)
    print_summary(summary)

    record = {"label": args.label, "config": vars(args) | {"request_rate": str(args.request_rate)},
              "summary": summary,
              "requests": [asdict(r) | {"itl": None} for r in results]}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.label}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(record, indent=1))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
