# P2.5：混合步的 Piecewise CUDA Graph

参考：vLLM V1 piecewise CUDA graph（以 attention 为界切分计算图）。

## 动机

P2 定位到：混合步（decode + 任意 prefill）只能 eager 执行，GPU kernel 时间与纯 decode 步几乎相同（12.9 vs 12.6 ms），
但单步墙钟时间 42 ms vs 12.9 ms——约 29 ms 是 Python 逐层发起 ~600 个 kernel 的 host 开销。

## 实现（`engine/nanovllm/engine/piecewise_graph.py`）

```
piece 0     : embed → layer0 [norm, qkv, q/k-norm, rope]          → 静态缓冲 q,k,v,residual
attention 0 : eager（写分页 KV + flash_attn_varlen，按真实 token 数）  → 静态缓冲 o
piece i     : layer i-1 [o_proj, norm, MLP] → layer i [norm, qkv, rope]
...
piece L     : layer L-1 [o_proj, norm, MLP] → final norm           → 静态缓冲 hidden
```

- 每个 token 数档位捕获 L+1 = 37 张图（Qwen3-8B），档位：≤512 步长 32，512~2048 步长 128（补齐浪费 ≤ ~12%）。
  初版只有 11 个稀疏档位，1056 token 的步被补到 1536，反而比 eager 慢（101 vs 78 ms），细化后消除。
- 补齐行 `slot_mapping=-1` 不写 KV；attention 与 LM head 只处理真实行。
- 所有档位、所有分段共用一组最大尺寸静态缓冲（`[:T]` 视图），临时张量共用一个 graph memory pool。
- 纯 decode 步仍走原有整图；超过最大档位的步回退 eager。开关：`Config.piecewise_cudagraph`、服务端 `--no-piecewise-cudagraph`。

## 单步耗时（`bench/step_cost.py`，Qwen3-8B，中位数，ms）

| decode bs | 纯 decode | +1 token：eager → 分段图 | +32 | +256 | +1024 |
|---|---|---|---|---|---|
| 32 | 12.9 | 42.1 → **14.7** | 40.2 → **15.5** | 40.6 → **30.4** | 78.3 → 80.1 |
| 128 | 18.1 | 43.1 → **24.4** | 41.2 → **31.9** | 42.4 → 42.4 | 92.3 → 93.2 |
| 224 | 27.5 | 44.0 → **30.0** | 48.0 → 48.0 | 64.3 → 65.1 | 113.9 → 115.4 |

大 prefill 步本身计算密集（compute-bound），launch 开销占比小，分段图收益趋零但不再变慢。

## 正确性

Qwen3-0.6B，64 条 ShareGPT 提示词贪心 64 token：分段图 vs eager 混合批 **64/64 完全一致**；与 HF 参考一致率 68.4%（同 eager）。

## 在线压测（Qwen3-8B，ShareGPT 500 条，mixed，budget 1024，延迟 ms）

| rr | 配置 | 输出 tok/s | TTFT P99 | TPOT 均值 | ITL P90 | ITL P99 |
|---|---|---|---|---|---|---|
| 8 | 上游 prefill_first | 1,485 | 161 | 22.1 | 54.1 | 140.9 |
| 8 | mixed eager | 1,478 | 140 | 18.6 | 41.4 | 59.5 |
| 8 | **mixed + 分段图** | 1,490 | 225 | **16.3** | **17.1** | **57.9** |
| 8 | vLLM 0.11.0 | 1,495 | 35 | 13.8 | 14.6 | 16.9 |
| 16 | 上游 prefill_first | 2,418 | 682 | 45.7 | 91.3 | 234.8 |
| 16 | mixed eager | 2,574 | 124 | 26.9 | 48.5 | 52.9 |
| 16 | **mixed + 分段图（3 次均值）** | **2,610** | 282 ± 196 | **19.0 ± 0.7** | **24.7** | **39.4 ± 4.2** |
| 16 | vLLM 0.11.0 | 2,622 | 148 | 17.2 | 21.8 | 24.2 |
| inf | mixed + 分段图 | 4,139 | 12,158 | 50.7 | 60.0 | 93.0 |
| inf | vLLM 0.11.0 | 4,403 | 10,176 | 44.8 | 56.6 | 68.5 |

rr=16 三次重复：ITL P99 37.2 / 36.8 / 44.2，TPOT 均值 18.8 / 18.5 / 19.8，吞吐 2,616 / 2,611 / 2,605；TTFT P99 83~474 波动大（500 样本的 P99 只对应 5 个请求）。

**相对上游（rr=16）**：ITL P99 −83%，TPOT 均值 −58%，吞吐 +8%；与 vLLM 吞吐持平（−0.5%），TPOT 均值差距约 10%。

## 剩余差距与后续

- 启动时捕获 28 档 × 37 张图，启动时间约增加 1.5 分钟。
- ITL P99 仍高于 vLLM（39 vs 24 ms）：每步仍有 36 次 eager attention + Python 调度/准备开销；
  可继续做 mini-SGLang 式 overlap 调度（CPU 准备第 N+1 步与 GPU 执行第 N 步重叠）。
