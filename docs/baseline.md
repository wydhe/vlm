# Baseline：原版 nano-vllm 离线吞吐

- 引擎：nano-vllm 上游 `bb823b3`（仓库提交 `44a43d7`，未做任何修改）
- 硬件：1 × A800-SXM4-80GB，驱动 535.129；torch 2.8.0+cu126，flash-attn 2.8.3，transformers 4.57.6
- 负载：与上游 `engine/bench.py` 相同 —— 256 条请求一次性提交，输入长度 U[100, 1024]（随机 token），
  输出长度 U[100, 1024]（`ignore_eos`），`seed(0)`，`max_model_len=4096`，总输出 133,966 tokens
- 命令：`MODEL_DIR=weights/<model> uv run python -m bench.offline_throughput [--enforce-eager]`

| 模型 | CUDA Graph | 耗时 (s) | 输出吞吐 (tok/s) | Graph 加速比 |
|---|---|---|---|---|
| Qwen3-0.6B | on  | 17.39 | **7,703** | 1.77× |
| Qwen3-0.6B | off | 30.80 | 4,349 | — |
| Qwen3-8B   | on  | 36.85 | **3,635** | 1.28× |
| Qwen3-8B   | off | 47.05 | 2,847 | — |

参考：上游 README 在 RTX 4070 Laptop (8GB) 上 Qwen3-0.6B 为 1,434 tok/s。

## 观察

- CUDA Graph 对小模型收益更大（1.77× vs 1.28×）：0.6B 的 decode 单步计算量小，kernel launch 开销
  占比高（launch-bound）；8B 单步计算和权重读取更重，launch 开销被摊薄。
- 该负载是**离线**、所有请求同时到达，只能反映峰值吞吐，**测不出 TTFT / ITL**
  （上游无在线服务与流式接口），这是 P1 要补的。

## 局限

- 每个配置只跑了 1 次，未报告方差；P1 起压测脚本改为多次重复取均值 ± 标准差。
- 随机 token 输入不触发前缀缓存命中，前缀缓存收益需用 RAG / 多轮负载单独测。
