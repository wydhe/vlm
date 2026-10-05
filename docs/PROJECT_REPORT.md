# 项目总报告：面向医疗问答的轻量推理引擎 × Agentic RAG × GRPO 对齐 × 多模态

> 单卡 A800-80GB 上完成的端到端系统：在开源推理引擎 nano-vllm 之上做在线服务、调度与 CUDA Graph 改造并接入 Qwen3-VL，
> 用它承载医疗 RAG / Agent 负载，并作为 GRPO 强化学习的 rollout 引擎。所有数字均来自本仓库可复现的实验，
> 局限与负面结果一并列出。各阶段细节见 `docs/` 下的分报告。

## 1. 系统总览

```
                         ┌────────────────────────────────────────────────────┐
  用户 / 评测脚本 ──────► │ OpenAI 兼容服务（SSE 流式、tools、/metrics）        │  engine/nanovllm/entrypoints
                         ├────────────────────────────────────────────────────┤
                         │ AsyncLLMEngine：独立引擎线程 + asyncio 流            │  engine/nanovllm/engine/async_engine.py
                         ├────────────────────────────────────────────────────┤
                         │ Scheduler：mixed（stall-free）/ prefill_first         │  scheduler.py
                         │ BlockManager：分页 KV + 前缀缓存（含图片内容哈希）    │  block_manager.py
                         │ ModelRunner：decode 整图 + 混合步分段 CUDA Graph      │  model_runner.py / piecewise_graph.py
                         │ 模型：Qwen3（文本）/ Qwen3-VL（M-RoPE + DeepStack）  │  models/
                         └──────────────▲───────────────────────▲─────────────┘
                                        │                       │ 原地权重同步 + 前缀缓存清空
   rag/  BM25 + MedCPT → RRF → 重排 ────┤                       │
   agent/ 工具调用循环（search_medical_kb）                      │
                                                    align/  GRPO 训练器（同卡同进程）
```

代码规模：在上游 nano-vllm（约 1,200 行）上修改 / 新增引擎代码约 1,090 行（17 个文件）；
RAG / Agent / 对齐 / 压测 / 测试模块约 2,860 行；29 项 CPU 单元测试。

## 2. 完成情况

| 阶段 | 内容 | 状态 | 分报告 |
|---|---|---|---|
| P0 | 环境（uv）、subtree 引入、原版离线基线 | ✅ | `baseline.md` |
| P1 | 在线流式服务、压测客户端、与 vLLM 对比 | ✅ | `p1_online_serving.md` |
| P2 | Prefill / Decode 混合批调度、真贪心采样 | ✅ | `p2_mixed_batching.md` |
| P2.5 | 混合步分段 CUDA Graph | ✅ | `p25_piecewise_cudagraph.md` |
| P3 | Qwen3-VL 支持、多模态前缀缓存 | ✅ | `p3_qwen3_vl.md` |
| P4 | Radix Tree 前缀缓存 | ⏸ 未做（理由见 §6） | — |
| P5 | 医疗混合检索 RAG、Agentic RAG | ✅ | `p5_rag_agent.md` |
| P6 | GRPO 对齐（引用 + 拒答） | ✅ | `p6_grpo.md` |
| P7 | Qwen3.5 混合注意力（Gated DeltaNet） | ❌ 未做 | — |

## 3. 核心结果

### 3.1 推理引擎（Qwen3-8B，ShareGPT 500 条，16 req/s 泊松到达，单卡 A800）

| 配置 | 输出 tok/s | TPOT 均值 | ITL P90 | ITL P99 |
|---|---|---|---|---|
| 上游 nano-vllm（+ 本项目在线服务层） | 2,418 | 45.7 ms | 91.3 ms | 234.8 ms |
| + 混合批调度（eager） | 2,574 | 26.9 ms | 48.5 ms | 52.9 ms |
| **+ 分段 CUDA Graph（3 次重复）** | **2,610** | **19.0 ± 0.7 ms** | **24.7 ms** | **39.4 ± 4.2 ms** |
| 官方 vLLM 0.11.0（同负载同配置） | 2,622 | 17.2 ms | 21.8 ms | 24.2 ms |

- 相对上游：**ITL P99 −83%，TPOT 均值 −58%，吞吐 +8%**；吞吐与 vLLM 持平（−0.5%），TPOT 均值仍差约 10%，ITL P99 仍差约 15 ms。
- 单步诊断：bs=32 时混入 1 个 prefill token 的步从 **42.1 ms 降到 14.7 ms**（纯 decode 12.9 ms）。
- 突发负载（500 条同时到达）：混合批使 TPOT P99 从 1.2 s 降到约 90 ms，代价是吞吐 −11%（与 vLLM 行为一致的固有权衡）。

### 3.2 多模态（Qwen3-VL-4B，SLAKE 英文测试集 1,061 题）

- 与 HF transformers 贪心输出 **61 / 64 逐 token 一致**（3 处为首 token 大小写 / 空格变体）；纯文本路径回归 64 / 64 一致。
- 准确率 0.597（封闭题 0.810 / 开放题 0.459）；批处理 9.33 req/s vs HF 逐条 6.57 req/s（瓶颈在串行视觉编码）。
- **多模态前缀缓存**：图片内容哈希混入块哈希。去掉它时，8 对同 token 不同图片中 6 对复用了错误的 KV、4 对答案被带偏；本实现 0 / 8。

### 3.3 RAG 与 Agent（Qwen3-8B 由本引擎服务，MIRAGE，MedRAG 教材语料 12.6 万片段）

| 方法（同一批 1,000 题） | 平均准确率 |
|---|---|
| 不检索 | **0.729** |
| 单次 RAG（BM25 + MedCPT → RRF → 重排） | 0.627 |
| Agentic RAG（模型自主决定检索，≤3 次） | 0.683 |

- 检索器之间：混合 + 重排最好；但**教材语料下强制检索整体降低准确率**（4,118 题：0.669 → 0.610），PubMedQA / BioASQ 尤甚（需要 PubMed 语料）。
- Agent 比单次 RAG 高 5.6 个点、在 MedQA / MMLU 上超过不检索，但仍检索过多（90% 的题）；前缀缓存命中率 21.1%。

### 3.4 GRPO 对齐（Qwen3-1.7B，150 步，单卡 36 分钟，评测 500 条留出集）

| 指标 | 训练前 | 训练后 |
|---|---|---|
| 答案准确率 | 0.439 | **0.633** |
| 引用准确率 | 0.711 | **0.897** |
| 过度拒答（↓） | 0.134 | **0.085** |
| 拒答召回 | 0.982 | 0.982 |
| 格式正确率 | 0.920 | 0.994 |

rollout 由本项目引擎在训练进程内完成（每步约 4.4 s），每步后原地同步权重。

## 4. 关键工程问题与定位过程（面试素材）

| # | 现象 | 定位 | 解决 |
|---|---|---|---|
| 1 | 在线负载下 ITL P99 达 235 ms，但 ITL 中位数与 vLLM 接近 | ITL P90 ≈ 一个 prefill 步耗时 → decode 被 prefill 插队 | 统一 token 预算的混合批（vLLM V1 / Sarathi-Serve 思路） |
| 2 | 混合批后调小 token 预算，ITL 几乎不变 | `bench/step_cost.py`：只混入 1 个 token 单步也从 13 → 42 ms；profiler 显示 GPU kernel 时间几乎相同（12.6 vs 12.9 ms） | 瓶颈是 eager 模式逐层发起 ~600 个 kernel 的 host 开销 → 分段 CUDA Graph |
| 3 | 分段图在 1056 token 的步上反而比 eager 慢 | 稀疏档位把 1056 补到 1536（浪费 45%） | 档位细化：≤512 步长 32、其后步长 128 |
| 4 | 训练进程里引擎启动"卡住" | py-spy：每张图进入 `torch.cuda.graph` 都执行 `gc.collect()`，812 张图 × 大 Python 堆 | 改用 `capture_begin/capture_end`，只 GC 一次 |
| 5 | 上游同配置重跑贪心输出只有 30/64 一致 | 采样器用 T=1e-5 近似贪心，bf16 并列 logit 由随机噪声打破 | `temperature=0` 走 argmax，重跑 64/64 |
| 6 | RL 中 rollout 可能静默 off-policy | 所有样本共享 system prompt，前缀缓存会复用旧权重算出的 KV | 每次权重同步后 `reset_prefix_cache()` |
| 7 | 不同图片可能共享 KV | 图片占位 token 完全相同，块哈希只看 token | 混入图片内容哈希；对照实验证明 6/8 对会出错 |
| 8 | 1,061 张图评测 OOM | 排队请求的视觉特征全部驻留 GPU | 特征放锁页内存，按分块拷贝，结束即释放 |
| 9 | GRPO 用 lr 1e-6 训练 | 纯 bf16 下多数更新会被舍入 | fp32 主权重 + bf16 autocast |
| 10 | 训练数据 yes:no ≈ 15:1 | 策略可靠"永远答 yes"拿奖励 | 训练集按标签平衡 |
| 11 | 运行中根分区写满 | 30 GB overlay 被 uv / pip 缓存占满 | 清理缓存；大文件全部放 EFS |

## 5. 负面结果与局限（如实记录）

- 引擎在线延迟仍落后 vLLM：ITL P99 39 vs 24 ms、TPOT 19.0 vs 17.2 ms；突发负载吞吐 4,139 vs 4,403 tok/s。
- 以教材为语料时 RAG 降低整体准确率；Agent 也未超过不检索的平均水平。没有下载 PubMed 语料（约 70 GB）。
- GRPO 只有单个随机种子，step 125 出现过回落；训练集没有 maybe 标签。
- Qwen3-VL：视觉编码串行、含图像的 prefill 走 eager、服务端未接入图像输入、不支持视频与多卡。
- 统计功效：多数在线压测为单次运行，P99 在 500 样本上只对应 5 个请求；关键对比（rr=16 分段图）做了 3 次重复。

## 6. 与开源项目的关系（简历与面试中需如实说明）

| 来自上游 nano-vllm（GeeeekExplorer，MIT） | 本项目实现 / 移植 |
|---|---|
| 分页 KV、按块哈希的前缀缓存、CUDA Graph（decode）、TP、torch.compile、仅首条序列可分块的 chunked prefill、离线 `generate` | 在线流式服务与 abort、混合批调度、真贪心、分段 CUDA Graph、Qwen3-VL 支持、多模态前缀缓存、权重原地同步与缓存失效、`/metrics`、压测与诊断工具 |

- 混合批的设计参考 **vLLM V1 调度器 / Sarathi-Serve**；分段 CUDA Graph 参考 **vLLM V1 piecewise CUDA graph**；
  mini-SGLang（MIT）的调度器经阅读确认仍是 prefill 优先，其 overlap 调度尚未移植。
- Radix Tree（P4）未做：上游已有按块哈希的前缀缓存（vLLM V1 同样采用哈希方案），且 flash-attn 分页 KV 要求块大小为 256 的倍数，
  Radix 树无法带来更细粒度；改为在 Agent 负载上实测现有前缀缓存（命中率 21%）。
- 视觉编码器复用 transformers 的 `Qwen3VLVisionModel`；检索模型使用 NCBI MedCPT；数据集均为公开数据。

## 7. 复现

```bash
uv sync                                                    # .venv：torch 2.8 cu126、flash-attn 2.8.3 等
bash scripts/download_assets.sh                            # 模型与数据 → weights/ data/
# 引擎
MODEL_DIR=weights/Qwen3-8B uv run python -m bench.offline_throughput
uv run python -m nanovllm.entrypoints.openai_server --model weights/Qwen3-8B --max-num-batched-tokens 1024
uv run python -m bench.serving --tokenizer weights/Qwen3-8B --request-rate 16
uv run python -m bench.step_cost --model weights/Qwen3-8B
# 多模态
uv run python -m bench.vqa_eval --model weights/Qwen3-VL-4B-Instruct
uv run python -m bench.mm_cache_ablation
# RAG / Agent（需先启动上面的服务）
uv run python -m rag.build_index --corpus data/medrag/textbooks/chunk --out data/indexes/textbooks
uv run python -m rag.eval_mirage --max-per-dataset 1000
uv run python -m agent.medical_agent --per-dataset 200 --compare results/rag/<run>.json
# GRPO
uv run python -m align.task && uv run python -m align.grpo --model weights/Qwen3-1.7B --steps 150
# 单元测试
uv run pytest tests
```

官方 vLLM 对照环境：`.venv-vllm`（vllm 0.11.0，需把 transformers 固定为 4.57.6）。
根分区只有 30 GB，大文件（权重、数据、结果、编译缓存）放仓库下被忽略的目录（位于 EFS）。

## 8. 简历写法建议（数字均可在本仓库复现）

**基于 nano-vllm 的医疗问答推理与对齐系统**（个人项目，单卡 A800）
- 基于开源 nano-vllm 深度改造推理引擎：实现异步流式 OpenAI 兼容服务；参考 vLLM V1 / Sarathi-Serve 实现 prefill/decode
  混合批调度；通过单步计时 + profiler 定位混合步 eager 执行的 kernel launch 开销（GPU 时间相同、墙钟 3.3×），移植分段 CUDA Graph。
  Qwen3-8B 在 16 req/s 在线负载下 **ITL P99 降低 83%（235→39 ms）、TPOT 降低 58%**，吞吐与 vLLM 持平。
- 为引擎接入 Qwen3-VL（交错 M-RoPE、DeepStack、视觉特征分块注入），与 HF 输出 61/64 逐 token 一致；
  发现并修复多模态前缀缓存的跨图片 KV 误复用（对照实验中 6/8 对出错）。
- 搭建医疗混合检索（BM25 + MedCPT + RRF + 交叉编码器重排）与工具调用 Agent，在 MIRAGE 上系统评估，Agent 比单次 RAG 高 5.6 个点。
- 手写 GRPO 训练器，以自研引擎做同卡 rollout 并原地同步权重（处理前缀缓存失效导致的隐性 off-policy），
  训练 Qwen3-1.7B "有据引用 / 无据拒答"：答案准确率 0.44→0.63，引用准确率 0.71→0.90。

## 9. 面试可能追问

- prefill 计算受限、decode 访存受限，混合批为什么能提升？token 预算如何在 TTFT / ITL / 吞吐间权衡？
- 为什么分段 CUDA Graph 以 attention 为界？补齐行如何保证不写 KV？档位粒度如何选？
- 前缀缓存为什么按块哈希、为什么链式哈希？多模态下为什么必须混入图片哈希？
- GRPO 与 PPO / DPO 的区别；组内全同奖励为什么没有梯度；μ=1 时 clip 为什么不生效；k3 KL 估计的性质。
- RL 中 rollout 引擎与训练权重如何保持一致？前缀缓存为什么会让 rollout 变成 off-policy？
- 为什么教材语料下 RAG 反而降低准确率？如何让 Agent 学会"不检索"？

## 10. 后续工作

1. 视觉编码批处理；服务端接入图像输入（OpenAI `image_url`）。
2. overlap 调度（参考 mini-SGLang），隐藏剩余的 CPU 开销，追平 vLLM 的 ITL P99。
3. 下载 PubMed 语料，重做 PubMedQA / BioASQ 的检索实验；用 GRPO 训练 Agent 的"何时检索"。
4. GRPO 多种子、保存最优检查点，评估 maybe 类数据。
5. P7：Qwen3.5 混合注意力（分页 KV 与 DeltaNet 递归状态共存）。
