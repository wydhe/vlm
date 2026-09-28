# P1：在线服务与压测（nano-vllm vs vLLM）

## 本阶段实现

| 模块 | 文件 | 说明 |
|---|---|---|
| 调度器取消 | `engine/nanovllm/engine/scheduler.py` `Scheduler.abort` | 等待中 / 运行中 / 分块 prefill 到一半的序列均可取消并释放 KV 块 |
| 逐步输出 | `engine/nanovllm/engine/llm_engine.py` `step_updates` | 返回本步采样出新 token 的序列（排除未完成的分块 prefill）；`step()` 行为不变 |
| 异步引擎 | `engine/nanovllm/engine/async_engine.py` | 独立引擎线程 + asyncio 队列；每步一次 `call_soon_threadsafe` 批量投递；准入校验防止永远无法调度的请求阻塞队头 |
| 增量反解码 | `engine/nanovllm/utils/detokenizer.py` | 滑动窗口，O(1)/token；多字节字符未完整前暂存 |
| HTTP 服务 | `engine/nanovllm/entrypoints/openai_server.py` | OpenAI 兼容 `/v1/completions`、`/v1/chat/completions`，SSE 流式、`include_usage`、客户端断开即 abort |
| 压测客户端 | `bench/serving.py` | 泊松到达、token-id 提示词、TTFT/TPOT/ITL/E2E 分位数，可直接压官方 vLLM |

验证：`tests/` 16 项单测通过；64 个流式请求在 1 s 后同时断开，GPU 利用率 1 s 内回到 0%，后续请求正常（72 ms / 16 token）。

## 压测设置

- 1 × A800-80GB；Qwen3-8B bf16；两端均 `max_num_seqs=256, max_num_batched_tokens=16384, max_model_len=4096, gpu_memory_utilization=0.9`
- 官方 vLLM 0.11.0（V1 引擎，默认开启 chunked prefill 与 prefix caching）；nano-vllm 为本分支
- 负载：ShareGPT V3，500 条，`seed=0`，按 vLLM `benchmark_serving` 规则过滤（prompt ≤1024，prompt+output ≤2048）；
  输出长度取真实回复长度并 `ignore_eos`；两端收到**完全相同**的 token-id 请求；正式计时前并发 16 条预热
- 命令：`uv run python -m bench.serving --tokenizer weights/Qwen3-8B --request-rate <rr> [--base-url ...]`

## 结果（延迟单位 ms）

| 请求速率 | 引擎 | 输出 tok/s | TTFT 中位 | TTFT P99 | TPOT 均值 | TPOT P99 | ITL P99 |
|---|---|---|---|---|---|---|---|
| 4 req/s | nano | 797 | 52 | 366 | 16.1 | 26.4 | 66.6 |
| | vLLM | 798 | 30 | 86 | 13.3 | 16.4 | 36.3 |
| 8 req/s | nano* | 1,485 | 54 | 210 / 464 | 21.2 | 35 | 137 |
| | vLLM | 1,495 | 26 | 35 | 13.8 | 14.8 | 16.9 |
| 16 req/s | nano | 2,461 | 62 | 305 | **43.1** | 87.1 | **233.7** |
| | vLLM | 2,622 | 35 | 148 | 17.2 | 22.7 | 24.2 |
| 突发 (inf) | nano | **4,795** | 2,249 | 2,511 | 102.1 | **1,201** | 51.7 |
| | vLLM | 4,403 | 710 | 10,176 | 44.8 | 85.9 | 68.5 |

\* rr=8 首次测量 TTFT P99 为 3,121 ms，两次重测为 210 / 464 ms，判定为首轮一次性 JIT 开销；表中 TPOT/ITL 取重测值（两次几乎一致）。

## 分析

1. **在线负载下 decode 被 prefill 打断是主要差距。** 上游调度器只要等待队列非空就执行纯 prefill 步，
   本步所有正在 decode 的请求全部停顿。泊松到达下几乎每个新请求都会打断一次：
   16 req/s 时 nano 的 TPOT 均值是 vLLM 的 2.5×，ITL P99 是 9.7×（234 vs 24 ms）。
   ITL P90 ≈ 54 ms（rr=8）正好对应一个 prefill 步的耗时，而中位数（14.5 ms）与 vLLM 接近——
   说明**纯 decode 步本身不慢，慢在被插队**。
2. **突发负载下 nano 吞吐反而更高（+9%）、平均 TTFT 更低**：prefill 优先会尽快把所有请求放进 batch；
   代价是 decode 饥饿，TPOT P99 达 1.2 s（vLLM 86 ms）。这是典型的吞吐 / 尾延迟权衡。
3. 低负载（4 req/s）下 TPOT 中位数仍高约 2.5 ms（15.7 vs 13.2）：除插队外还有每步固定开销
   （Python 调度、引擎线程与 HTTP 事件循环争用 GIL、采样），需在 P2 用 profiler 拆分。

## 下一步（P2）

- 调度器改为 **prefill 分块与 decode 混合批**（token 预算内优先放 decode，剩余预算给 prefill 分块，
  允许多条序列同时分块）。目标：16 req/s 下 ITL P99 从 234 ms 显著下降，同时保住突发吞吐。
- Profiling：拆分每步 CPU 开销；评估把引擎循环从线程改为独立进程（避免 GIL 争用）。

## 局限

- 除 rr=8 外每个配置只跑 1 次；P99 在 500 条样本上只对应 5 个请求，波动较大（见 rr=8 的 210 / 464）。
  后续关键对比改为 ≥3 次重复并报告均值 ± 标准差。
- 客户端按 SSE 块计时；nano 每 token 一块，vLLM 在高负载下也基本每 token 一块，ITL 口径一致。
