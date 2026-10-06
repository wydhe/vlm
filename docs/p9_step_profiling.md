# 单步耗时拆解：不做 overlap 调度，改为拆分混合步的 attention

原计划是移植 mini-SGLang 的 overlap 调度（CPU 准备第 N+1 步的同时 GPU 算第 N 步），用来缩小和 vLLM 的差距。
它需要把采样结果的回传推迟一步，会明显增加代码复杂度。所以先测量，再决定做不做。

## 1. 测量：CPU 开销只占 2~4%

`bench/step_breakdown.py`：给一步里的各阶段计时，并用 CUDA event 测 GPU 真正在算的时间（Qwen3-8B，单位 ms）。

| 场景 | 墙钟 | GPU | schedule | prepare | postprocess | CPU 占比 |
|---|---|---|---|---|---|---|
| decode bs=32 | 13.0 | 12.6 | 0.03 | 0.20 | 0.02 | 3% |
| decode bs=32 + prefill 64 | 17.2 | 16.7 | 0.03 | 0.28 | 0.03 | 3% |
| decode bs=128 | 18.3 | 17.7 | 0.09 | 0.34 | 0.08 | 4% |
| decode bs=128 + prefill 64 | 34.2 | 33.4 | 0.10 | 0.48 | 0.10 | 2% |
| decode bs=224 | 28.3 | 27.4 | 0.15 | 0.48 | 0.14 | 3% |
| decode bs=224 + prefill 64 | 61.5 | 60.3 | 0.16 | 0.69 | 0.18 | 2% |

**结论：overlap 调度最多能藏掉 2~4% 的时间，不值得为它增加代码复杂度，不做。**

## 2. 新发现：混合步的 GPU 时间异常

同一张表里，只多 64 个 prefill token，bs=224 时 GPU 时间从 27.4 ms 涨到 60.3 ms，而且 batch 越大涨得越多——
这不是 prefill 本身的计算量。torch.profiler 拆开 GPU kernel（bs=224）：

| kernel | 纯 decode | 混合步（+64） |
|---|---|---|
| flash-attn（splitkv） | 6.7 ms | **27.3 ms（4×）** |
| GEMM | 15.8 ms | 21.0 ms（token 从 224 变成 288，正常增长） |

**原因**：flash-attn 2 的 varlen kernel 只在"批里所有查询都只有 1 个 token"时，才会把同一组 KV 头对应的多个查询头
打包成一个计算块（GQA 打包）。混合步里有一条 64 token 的 prefill，最大查询长度变成 64，这个优化失效，
224 条 decode 都按 64 行的计算块计算，大部分算力浪费在空行上。

验证过程中有一次误判：单独用微基准对比"全是 decode 时"的两种 kernel，几乎没有差别（1.1×），因为那时打包优化正常生效；
加上一条 prefill 后用 profiler 看真实的混合步，才定位到问题。

## 3. 修复：混合步的 attention 分两段算

混合调度器本来就把 decode 序列排在批的最前面。所以在混合步里：

- 前 n 行（decode）用 decode 专用的 `flash_attn_with_kvcache`；
- 后面的 prefill 行用 `flash_attn_varlen_func`；
- 两段结果拼起来。LM head 取"每条序列最后一个 token"时，decode 行直接取前 n 行，prefill 部分加上偏移 n。

改动：`utils/context.py`（一个字段）、`engine/model_runner.py`（拆分元数据）、`layers/attention.py`、`layers/embed_head.py`，共约 40 行。

**正确性**：Qwen3-0.6B，64 条提示词贪心 64 token，与 HF 参考的一致率 70.2%（拆分前 68.4%，同一水平）；
与拆分前 60/64 条完全一致，其余是换 kernel 后 bf16 数值差异导致的个别 token 不同。

## 4. 效果

单步（`bench/step_breakdown.py`，ms）：

| 场景 | 拆分前 | 拆分后 |
|---|---|---|
| decode bs=224 + prefill 64 | 61.5 | **38.0（−38%）** |
| decode bs=128 + prefill 64 | 34.2 | **25.9（−24%）** |
| decode bs=32 + prefill 64 | 17.2 | 16.4 |

在线压测（Qwen3-8B，ShareGPT 500 条，每组 3 次）：

| 负载 | 指标 | 拆分前 | **拆分后** | vLLM 0.11.0 |
|---|---|---|---|---|
| 突发 | 输出 tok/s | 4,187 ± 50 | **4,451 ± 64** | 4,409 ± 68 |
| | TPOT 均值 | 50.2 ± 2.1 | **43.7 ± 1.8** | 44.3 ± 4.9 |
| | ITL P90 | 59.9 ± 0.5 | **46.9 ± 0.2** | 56.2 ± 0.3 |
| | ITL P99 | 96.7 ± 3.9 | 81.6 ± 0.8 | 67.4 ± 13.4 |
| 16 req/s | TPOT 均值 | 18.5 ± 0.3 | 18.1 ± 0.3 | 16.8 ± 0.2 |
| | ITL P99 | 37.2 ± 1.3 | 35.0 ± 2.8 | 23.2 ± 0.1 |
| 8 req/s | TPOT 均值 | 15.1 ± 0.8 | 15.3 ± 0.9 | 14.4 ± 0.8 |

- **高负载（大 batch）下收益明显**：突发负载的吞吐从落后 vLLM 5% 变为持平，TPOT 均值持平，ITL P90 低于 vLLM。
- 16 req/s 及以下变化在误差范围内：此时同时在跑的请求只有几十条，这项浪费本来就小。
- 与 vLLM 剩下的主要差距是中低负载下的 ITL P99（35 vs 23 ms）。
