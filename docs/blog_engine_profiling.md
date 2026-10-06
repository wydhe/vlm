# 给 nano-vllm 做在线服务优化：一次"先测量再动手"的性能分析

> 单卡 A800，Qwen3-8B，ShareGPT 500 条请求，16 req/s 泊松到达。
> 起点：ITL P99 248 ms；终点：35 ms。中间猜错过两次，每次都是测量把我拉回来的。

nano-vllm 是一个约 1200 行的 vLLM 精简实现，离线吞吐和 vLLM 相当，代码非常适合阅读。但它只有离线 `generate` 接口。
我给它加了一个异步引擎线程和 OpenAI 兼容的流式服务之后，第一次在线压测就看到了问题。

## 1. 第一个现象：ITL 的中位数正常，P99 却很高

8 req/s 下：

| 指标 | nano-vllm | vLLM 0.11.0 |
|---|---|---|
| ITL 中位数 | 14.5 ms | 13.7 ms |
| ITL P90 | 54 ms | 14.6 ms |
| ITL P99 | 137 ms | 16.9 ms |

中位数几乎一样，说明**纯 decode 本身不慢**。P90 恰好约等于一个 prefill 步的耗时——
看调度器的代码就明白了：只要等待队列里有请求，这一步就只做 prefill，所有正在生成的请求都停下来等。
泊松到达下几乎每个新请求都会打断一次。

**改法**：参考 vLLM V1 / Sarathi-Serve，每一步先给所有正在运行的请求各安排 1 个 decode token，剩下的 token 预算再去
prefill 新请求，长 prompt 分块。decode 可以看成"查询长度为 1 的 prefill 分块"，复用现有的 varlen attention 路径即可。

结果：16 req/s 下 ITL P99 从 235 ms 降到 53 ms。但 P90 仍然卡在 48 ms 左右，**而且把 token 预算从 16384 一路调到 512 都没用**。

## 2. 第一次猜错："prefill 块太大"

预算对 ITL 没影响，说明问题不在 prefill 的计算量。我写了一个单步计时工具：稳态 decode 中，每一步混入一条不同长度的新请求。

| 混入的 prefill token 数 | 0 | 1 | 32 | 256 |
|---|---|---|---|---|
| 单步耗时（bs=32） | 12.9 ms | **42.1 ms** | 40.2 ms | 40.6 ms |

只混入 **1 个** token，单步就变成 3 倍多，而且和 prefill 长度几乎无关。这是固定开销。torch.profiler 给出了答案：
两种步骤的 GPU kernel 总时间几乎相同（12.6 vs 12.9 ms）——**GPU 在等 CPU**。纯 decode 步走 CUDA Graph；
混合步因为每一步的序列长度都在变，只能 eager 执行，Python 逐层发起约 600 个 kernel。

**改法**：分段 CUDA Graph（vLLM V1 的思路）。以每层 attention 为界把模型切成 37 段，attention 之外的计算只和 token 数有关，
按 token 数分档录图；attention 留在图外 eager 执行。混入 1 个 token 的步从 42 ms 降到 15 ms。

一个小坑：最初只录了 11 个稀疏档位，1056 个 token 的步会被补齐到 1536，反而比 eager 还慢。档位改成 512 以内步长 32、
之后步长 128 才解决。另一个坑：`torch.cuda.graph` 每进入一次都会 `gc.collect()`，录 800 多张图时在大进程里非常慢，
改用 `capture_begin/capture_end` 只 GC 一次。

## 3. 第二次猜错："那剩下的差距是 CPU 开销"

这时和 vLLM 在 16 req/s 下的 TPOT 差约 10%。我的计划是移植 mini-SGLang 的 overlap 调度：CPU 准备下一步时 GPU 算这一步。
它要把采样结果的回传推迟一步，代码复杂度不低。动手之前先测：

| 场景 | 墙钟 | GPU | CPU 占比 |
|---|---|---|---|
| decode bs=224 | 28.3 ms | 27.4 ms | 3% |
| decode bs=224 + prefill 64 | 61.5 ms | 60.3 ms | 2% |

**CPU 只占 2~4%，overlap 调度最多省这么多，不做。**

但这张表里有更奇怪的东西：只多 64 个 prefill token，GPU 时间从 27 ms 翻到 60 ms，而且 batch 越大涨得越多。
profiler 拆开 kernel：attention 从 6.7 ms 变成 27.3 ms，**4 倍**。

我先写了个微基准对比 decode kernel 和 varlen kernel 处理纯 decode 查询——几乎一样快。又猜错了。
区别在于真实的混合步里**还有一条 prefill**：flash-attn 2 的 varlen kernel 只在所有查询长度都是 1 时，
才会把同一 KV 头对应的多个查询头打包成一个计算块；批里出现一条 64 token 的 prefill，这个优化就失效，
224 条 decode 每条都占一个 64 行的计算块，绝大部分算力花在空行上。

**改法**：调度器本来就把 decode 排在前面，混合步里 attention 拆成两段：前面的 decode 行用 `flash_attn_with_kvcache`，
后面的 prefill 行用 `flash_attn_varlen_func`。约 40 行改动。bs=224 的混合步从 61.5 ms 降到 38.0 ms；
突发负载下吞吐从落后 vLLM 5% 变成持平。

## 4. 结果（每组 3 次重复，均值 ± 标准差）

| 16 req/s | 上游 nano-vllm | 本项目 | vLLM 0.11.0 |
|---|---|---|---|
| ITL P99 | 247.8 ± 47.3 ms | **35.0 ± 2.8 ms** | 23.2 ± 0.1 ms |
| TPOT 均值 | 43.6 ± 1.0 ms | **18.1 ± 0.3 ms** | 16.8 ± 0.2 ms |
| 输出吞吐 | 2,430 tok/s | **2,611 tok/s** | 2,623 tok/s |

还没追平 vLLM 的地方：中低负载下的 ITL P99（35 vs 23 ms）。

## 5. 几点体会

1. **中位数和尾部要分开看。** "中位数正常、P90 等于某个操作的耗时"本身就是很强的线索。
2. **调参没用时，先怀疑自己的假设。** 预算对 ITL 无效，说明瓶颈根本不在预算控制的那部分。
3. **微基准要复现真实条件。** 我的第二个微基准漏掉了"批里有一条 prefill"，结论完全相反。
4. **测量也能帮你决定"不做什么"。** overlap 调度是业界成熟方案，但在这里只值 2~4%，不值得它带来的复杂度。

代码与完整数据：仓库中 `engine/`、`bench/step_cost.py`、`bench/step_breakdown.py` 以及 `docs/` 下的各阶段报告。
混合批与分段 CUDA Graph 的设计参考了 vLLM V1 与 Sarathi-Serve；nano-vllm 原作者为 GeeeekExplorer（MIT）。
