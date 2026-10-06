"""拆解一步的耗时：CPU 各阶段花了多少，GPU 真正在算多少。

一步引擎 step 依次做：
    schedule      调度器挑选这一步要跑的序列（CPU）
    prepare       拼输入、拷到 GPU（CPU）
    gpu           前向 + 采样（GPU，用 CUDA event 计时）
    sync          把采样结果拷回 CPU 时的等待（.tolist()）
    postprocess   追加 token、判断结束、释放 KV 块（CPU）

如果 "墙钟时间 - GPU 时间" 占比很大，说明 GPU 在等 CPU，overlap 调度（CPU 准备第 N+1 步时
GPU 同时算第 N 步）才有意义；占比很小，就不值得为它增加代码复杂度。

用法：
    uv run python -m bench.step_breakdown --model weights/Qwen3-8B
"""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict

import torch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


class StepTimer:
    """包住引擎内部的几个函数，记录每次调用的耗时。"""

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.times: dict[str, float] = defaultdict(float)
        runner = engine.model_runner
        self._wrap(engine.scheduler, "schedule", "schedule")
        self._wrap(engine.scheduler, "postprocess", "postprocess")
        self._wrap(runner, "prepare_prefill", "prepare")
        self._wrap(runner, "prepare_decode", "prepare")
        self._wrap_gpu(runner)

    def _wrap(self, obj, method: str, label: str) -> None:
        original = getattr(obj, method)

        def timed(*args, **kwargs):
            start = time.perf_counter()
            result = original(*args, **kwargs)
            self.times[label] += time.perf_counter() - start
            return result

        setattr(obj, method, timed)

    def _wrap_gpu(self, runner) -> None:
        """run_model + sampler 用 CUDA event 计 GPU 时间；采样结果回传单独计 sync。"""
        original_run_model = runner.run_model
        original_sampler = runner.sampler
        timer = self

        class TimedTokens:
            """sampler 返回的张量：调用 .tolist() 时才真正等待 GPU。"""

            def __init__(self, tokens):
                self.tokens = tokens

            def tolist(self):
                gpu_end = torch.cuda.Event(enable_timing=True)
                gpu_end.record()
                start = time.perf_counter()
                values = self.tokens.tolist()
                timer.times["sync"] += time.perf_counter() - start
                timer.times["gpu"] += timer.gpu_start.elapsed_time(gpu_end) / 1000
                return values

        def run_model(*args, **kwargs):
            timer.gpu_start = torch.cuda.Event(enable_timing=True)
            timer.gpu_start.record()
            return original_run_model(*args, **kwargs)

        def sampler(*args, **kwargs):
            return TimedTokens(original_sampler(*args, **kwargs))

        runner.run_model = run_model
        runner.sampler = sampler

    def reset(self) -> None:
        self.times.clear()


def measure(engine: LLMEngine, timer: StepTimer, steps: int, add_prefill: int) -> dict[str, float]:
    """跑 steps 步，每步可选地加入一个 add_prefill 长度的新请求，返回每步平均耗时（ms）。"""
    rng = random.Random(0)
    one_token = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    timer.reset()
    wall = 0.0
    for _ in range(steps):
        if add_prefill:
            engine.add_request([rng.randint(1000, 20000) for _ in range(add_prefill)], one_token)
        torch.cuda.synchronize()
        start = time.perf_counter()
        engine.step_updates()
        wall += time.perf_counter() - start
    result = {name: 1000 * t / steps for name, t in timer.times.items()}
    result["wall"] = 1000 * wall / steps
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="拆解每一步的 CPU / GPU 耗时")
    p.add_argument("--model", default="weights/Qwen3-8B")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 128, 224])
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args()

    engine = LLMEngine(args.model, max_num_seqs=256, max_model_len=4096, max_num_batched_tokens=1024)
    timer = StepTimer(engine)
    rng = random.Random(1)
    long_run = SamplingParams(temperature=0, max_tokens=3000, ignore_eos=True)

    print("| 场景 | 墙钟 | GPU | schedule | prepare | sync 等待 | postprocess | CPU 占比 |")
    print("|---|---|---|---|---|---|---|---|")
    for bs in args.batch_sizes:
        for seq in list(engine.scheduler.running) + list(engine.scheduler.waiting):
            engine.scheduler.abort(seq)
        for _ in range(bs):
            engine.add_request([rng.randint(1000, 20000) for _ in range(300)], long_run)
        while engine.scheduler.waiting:
            engine.step_updates()
        for _ in range(5):
            engine.step_updates()
        for label, add_prefill in ((f"decode bs={bs}", 0), (f"decode bs={bs} + prefill 64", 64)):
            r = measure(engine, timer, args.steps, add_prefill)
            cpu_share = 1 - r.get("gpu", 0) / r["wall"]
            print(f"| {label} | {r['wall']:.1f} | {r.get('gpu', 0):.1f} | {r.get('schedule', 0):.2f} | "
                  f"{r.get('prepare', 0):.2f} | {r.get('sync', 0):.2f} | {r.get('postprocess', 0):.2f} | "
                  f"{cpu_share:.0%} |")


if __name__ == "__main__":
    main()
