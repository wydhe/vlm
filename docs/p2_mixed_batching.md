# P2：Prefill / Decode 混合批调度

参考：vLLM V1 统一 token 预算调度器、Sarathi-Serve（stall-free batching）。
mini-SGLang（MIT）调度主循环仍是 prefill 优先，其 overlap 调度留待后续移植。

## 实现

| 改动 | 位置 |
|---|---|
| `scheduling_policy="mixed"`（默认）：先给所有运行中序列各 1 个 decode token，剩余预算分块 prefill 等待序列；上游策略保留为 `prefill_first` 用于对照 | `engine/nanovllm/engine/scheduler.py` |
| 混合步复用 varlen 路径：decode 视为 q_len=1 的分块，经 `block_table` 读分页 KV；纯 decode 步仍走 CUDA Graph | `scheduler.py` / `model_runner.py` |
| TP 安全：混合批中 decode 序列用 `last_token` 作为输入（TP worker 只收到它） | `model_runner.prepare_prefill` |
| 真贪心：`temperature=0` 走 argmax。上游用 1e-5 近似贪心，bf16 并列最大值被随机打破，**同配置重跑只有 30/64 条输出一致** | `layers/sampler.py`、`sampling_params.py` |
| 单测：decode 不被新 prompt 打断、分块跨步续算、预算约束、块回收 | `tests/test_scheduler_mixed.py` |

## 正确性

Qwen3-0.6B，64 条 ShareGPT 提示词（≤1500 token），贪心生成 64 token；以 HF transformers 逐条 bf16 贪心为参考：

| 对比 | 完全一致 | 前缀一致 token 比例 |
|---|---|---|
| prefill_first vs prefill_first（重跑） | 64/64 | 100% |
| HF vs prefill_first（上游） | 31/64 | 69.0% |
| HF vs prefill_first, budget 512（仅分块） | 27/64 | 64.8% |
| HF vs mixed, budget 16384 | 31/64 | 68.6% |
| HF vs mixed, budget 512 | 33/64 | 68.4% |

混合批与参考的一致率与上游持平；残余差异来自 bf16 下 batch 组成不同导致的数值差（上游自身分块也同样存在），不是逻辑错误。

## 性能（Qwen3-8B，ShareGPT 500 条，设置同 P1，延迟 ms）

| 配置 | rr | 输出 tok/s | TTFT P99 | TPOT 均值 | TPOT P99 | ITL P90 | ITL P99 |
|---|---|---|---|---|---|---|---|
| prefill_first（上游） | 8 | 1,485 | 161 | 22.1 | 53.2 | 54.1 | 140.9 |
| mixed, 16384 | 8 | 1,486 | 128 | 19.0 | 30.0 | 41.9 | 59.1 |
| mixed, 1024 | 8 | 1,478 | 140 | 18.6 | 32.5 | 41.4 | 59.5 |
| vLLM 0.11.0（P1） | 8 | 1,495 | 35 | 13.8 | 14.8 | 14.6 | 16.9 |
| prefill_first（上游） | 16 | 2,418 | 682 | 45.7 | 115.0 | 91.3 | 234.8 |
| mixed, 16384 | 16 | 2,550 | 282 | 27.5 | 40.6 | 49.2 | 56.8 |
| mixed, 2048 | 16 | 2,554 | 331 | 28.5 | 47.8 | 50.8 | 55.9 |
| **mixed, 1024** | 16 | **2,574** | **124** | **26.9** | **40.1** | 48.5 | **52.9** |
| mixed, 512 | 16 | 2,567 | 321 | 27.6 | 46.3 | 49.4 | 54.5 |
| vLLM 0.11.0（P1） | 16 | 2,622 | 148 | 17.2 | 22.7 | 21.8 | 24.2 |
| prefill_first（上游） | inf | 4,670 | 2,860 | 106.1 | 1,200 | 30.1 | 34.0 |
| mixed, 16384 | inf | 4,149 | 12,445 | 48.7 | 78.5 | 64.6 | 80.0 |
| mixed, 1024 | inf | 3,992 | 13,463 | 53.9 | 103.5 | 66.7 | 90.4 |
| vLLM 0.11.0（P1） | inf | 4,403 | 10,176 | 44.8 | 85.9 | 56.6 | 68.5 |

### 结论

1. **16 req/s 下（mixed, 1024 vs 上游）**：ITL P99 234.8 → 52.9 ms（**−77%**），TPOT 均值 45.7 → 26.9 ms（−41%），
   TPOT P99 115 → 40 ms（−65%），TTFT P99 682 → 124 ms，输出吞吐 +6%。
2. **突发负载**：行为与 vLLM 一致——TPOT P99 从 1.2 s 降到 79 ms，代价是吞吐 −11%、TTFT 变差
   （decode 优先占用预算，新请求排队更久）。这是 stall-free 调度的固有权衡。
3. **token 预算在 512~16384 之间对 ITL 影响很小**（ITL P99 53~59 ms）：ShareGPT 提示词多数很短，
   大 prefill 块本就少见。ITL P90 仍约 48 ms、是 vLLM 的 2 倍多。
   原因见下方"瓶颈定位"：混合步的 eager 执行有约 29 ms 的固定 CPU 开销。

## 瓶颈定位：混合步 eager 执行的 kernel launch 开销

`uv run python -m bench.step_cost --model weights/Qwen3-8B`（稳态 decode 中每步加入一条随机 token 新请求，单步中位数）：

| decode bs | 纯 decode（CUDA Graph） | +1 token prefill | +32 | +256 | +1024 |
|---|---|---|---|---|---|
| 32 | 12.9 ms | 42.1 ms | 40.2 ms | 40.6 ms | 78.3 ms |
| 128 | 18.0 ms | 43.1 ms | 41.2 ms | 42.4 ms | 92.3 ms |
| 224 | 27.4 ms | 44.0 ms | 48.0 ms | 64.3 ms | 113.9 ms |

- 只混入 **1 个** prefill token，单步就从 12.9 ms 变成 42 ms；并且这个约 40 ms 的"地板"几乎不随 bs、prefill 长度（≤256）变化 → 固定开销，而非计算量。
- torch.profiler（bs=32）：纯 decode 步 GPU kernel 总时长 **12.6 ms**，混合（+1）步 **12.9 ms**——GPU 工作量几乎相同，
  多出的约 29 ms 全部是 host 侧：eager 模式下 Python 逐层发起约 600 个 kernel，GPU 在等 CPU。
- 这解释了 ITL P90 ≈ 48 ms 且对 token 预算不敏感：在线负载下大部分步都会混入新请求的 prefill。

**下一步（P2.5）**：移植 vLLM V1 的 piecewise CUDA Graph——按 token 数分桶捕获"除 attention 以外"的计算图，
attention（varlen + 分页 KV）在图间 eager 执行。预期混合步回落到接近 GPU 实际耗时，届时再做 overlap 调度隐藏剩余 CPU 开销。

## 局限

- 每个配置 1 次运行；P99 基于 500 条样本，波动较大（参见 P1 中 rr=8 的重复实验）。关键结论需补 3 次重复。
- vLLM 数据来自 P1 同负载运行（当时 nano 尚未修复贪心；vLLM 本身一直为真贪心）。
