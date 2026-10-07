# 项目总报告：面向医疗问答的轻量推理引擎 × 检索增强 × GRPO 对齐 × 多模态

> 单卡 A800-80GB 上完成的端到端系统：在开源推理引擎 nano-vllm 之上做在线服务、调度、CUDA Graph 与 attention 改造并接入
> Qwen3-VL；用它承载医疗 RAG / Agent 负载，作为 GRPO 强化学习的 rollout 引擎，并把对齐后的模型部署回检索流程。
> 所有数字均来自本仓库可复现的实验，关键结论做了重复实验；负面结果一并列出。各阶段细节见 `docs/` 下的分报告。

## 1. 系统总览

```
                         ┌────────────────────────────────────────────────────────┐
  评测脚本 / 演示界面 ──► │ OpenAI 兼容服务：SSE 流式、tools、图片输入、/metrics      │ engine/nanovllm/entrypoints
                         ├────────────────────────────────────────────────────────┤
                         │ AsyncLLMEngine：独立引擎线程 + asyncio 流                │ engine/nanovllm/engine/async_engine.py
                         ├────────────────────────────────────────────────────────┤
                         │ Scheduler：混合批（decode 优先 + prefill 分块）           │ scheduler.py
                         │ BlockManager：分页 KV + 前缀缓存（含图片内容哈希）        │ block_manager.py
                         │ ModelRunner：decode 整图 + 混合步分段 CUDA Graph          │ model_runner.py / piecewise_graph.py
                         │ Attention：混合步拆成 decode kernel + varlen kernel        │ layers/attention.py
                         │ 模型：Qwen3（文本）/ Qwen3-VL（M-RoPE、DeepStack）       │ models/
                         └──────────────▲──────────────────────────▲──────────────┘
                                        │                          │ 原地权重同步 + 前缀缓存清空
   rag/   教材 / PubMed（2390 万条）/ 中文检索 ─┤                          │
   agent/ 工具调用 Agent、有据作答 + 闭卷兜底、演示界面             align/ GRPO 训练器（同卡同进程）
```

代码规模：在上游 nano-vllm（约 1,200 行）上修改 / 新增引擎代码约 1,250 行（19 个文件）；检索、Agent、对齐、压测、测试和脚本约 3,600 行；
34 项 CPU 单元测试。项目自己写的代码，注释和文档字符串均为中文；上游代码保持原样。

## 2. 完成情况

| 内容 | 状态 | 分报告 |
|---|---|---|
| 环境（uv）、subtree 引入、上游离线基线 | ✅ | `baseline.md` |
| 在线流式服务、压测客户端、与 vLLM 对比 | ✅ | `p1_online_serving.md` |
| 混合批调度、真贪心采样 | ✅ | `p2_mixed_batching.md` |
| 混合步分段 CUDA Graph | ✅ | `p25_piecewise_cudagraph.md` |
| 单步耗时拆解 → 不做 overlap 调度，改为拆分混合步 attention | ✅ | `p9_step_profiling.md` |
| Qwen3-VL、多模态前缀缓存 | ✅ | `p3_qwen3_vl.md` |
| 批量视觉编码、服务端图片输入、演示界面 | ✅ | §3.2 |
| 医疗混合检索 RAG、Agentic RAG | ✅ | `p5_rag_agent.md` |
| PubMed 全量语料（2390 万条）两阶段检索 | ✅ | `p10_pubmed_rag.md` |
| 中文医学检索与考试题评测 | ✅（负面结果） | `p11_chinese_rag.md` |
| PubMed 语料下重做闭环评测与 Agent | ✅ | `p12_pubmed_followups.md` |
| GRPO 对齐（引用 + 拒答），3 个随机种子 | ✅ | `p6_grpo.md`、`p8_robustness.md` |
| 闭环：GRPO 模型部署回检索流程 | ✅ | `p7_closed_loop.md`、`p8_robustness.md` |
| 压测重复实验（上游 / 本项目 / vLLM × 3 档负载 × 3 次） | ✅ | `p8_robustness.md` |
| 技术博客草稿 | ✅ | `blog_engine_profiling.md` |
| overlap 调度 | ❌ 测量后决定不做（CPU 只占单步 2~4%） | `p9_step_profiling.md` |
| Radix Tree 前缀缓存 | ❌ 未做（理由见 §6） | — |
| 用 GRPO 训练 Agent 的"何时检索" | ❌ 未做 | — |
| Qwen3.5 混合注意力（Gated DeltaNet） | ❌ 未做 | — |

## 3. 核心结果

### 3.1 推理引擎（Qwen3-8B，ShareGPT 500 条，单卡 A800，每组 3 次重复，均值 ± 标准差）

| 16 req/s 泊松到达 | 上游 nano-vllm* | **本项目** | vLLM 0.11.0 |
|---|---|---|---|
| 输出吞吐 | 2,430 ± 12 tok/s | **2,611 ± 13 tok/s** | 2,623 ± 3 tok/s |
| TPOT 均值 | 43.6 ± 1.0 ms | **18.1 ± 0.3 ms** | 16.8 ± 0.2 ms |
| ITL P99 | 247.8 ± 47.3 ms | **35.0 ± 2.8 ms** | 23.2 ± 0.1 ms |

| 突发负载（500 条同时到达） | 上游 nano-vllm* | **本项目** | vLLM 0.11.0 |
|---|---|---|---|
| 输出吞吐 | 4,740 ± 34 tok/s | 4,451 ± 64 tok/s | 4,409 ± 68 tok/s |
| TPOT 均值 | 103.2 ± 4.3 ms | **43.7 ± 1.8 ms** | 44.3 ± 4.9 ms |
| ITL P90 | 30.4 ± 0.2 ms | 46.9 ± 0.2 ms | 56.2 ± 0.3 ms |

\* 上游的调度方式 + 本项目的在线服务层（上游本身没有在线服务）。

- 相对上游（16 req/s）：**ITL P99 −86%，TPOT 均值 −58%，吞吐 +7%**；与 vLLM 吞吐持平，TPOT 均值差约 8%，ITL P99 差约 12 ms。
- 突发负载下吞吐、TPOT 都与 vLLM 持平。上游吞吐最高，但 TPOT P99 达 1.2 秒（prefill 优先的固有权衡）。
- 三步优化各自的作用：混合批让 16 req/s 的 ITL P99 从 235 降到 53 ms；分段 CUDA Graph 降到 39 ms；
  拆分 attention 主要改善高负载（混合步 bs=224 + 64 token 从 61.5 降到 38.0 ms，突发吞吐 +6%）。

### 3.2 多模态（Qwen3-VL-4B，SLAKE 英文测试集 1,061 题）

- 与 HF transformers 贪心输出 **60 / 64 逐 token 一致**（分歧都是同一答案的大小写 / 空格变体；视觉编码器从 sdpa 换成
  flash-attention 之前是 61/64）；纯文本路径每次改动前后输出 64/64 一致。
- 准确率 0.598（封闭题 0.810 / 开放题 0.462）。吞吐 10.15 req/s；HF 逐条 6.41 req/s。
- **批量视觉编码**：1,061 张图从 73.8 秒降到 34.4 秒（2.1×），但端到端只从 9.33 提升到 10.15 req/s——含图片的 prefill 步需要三维位置，
  只能 eager 执行，成为新的瓶颈。
- **多模态前缀缓存**：图片内容哈希混入块哈希。去掉它时，8 对"token 相同、图片不同"的样本里 6 对复用了错误的 KV、4 对答案被带偏；本实现 0/8。
- 服务端支持 OpenAI 格式的 `image_url`（只接受 base64 和本地文件，拒绝远程地址，避免 SSRF）。
- 演示界面（`agent/demo.py`，Gradio）：医学影像问答 + 有据问答（引用原文；资料不足时明确提示"未经资料核实"，再闭卷作答）。

### 3.3 检索增强（Qwen3-8B 由本项目引擎服务）

| 数据 | 语料 | 不检索 | 检索（混合 + 重排） | 说明 |
|---|---|---|---|---|
| MIRAGE 5 个数据集（4,118 题） | MedRAG 教材 | **0.669** | 0.610 | 语料覆盖不到 PubMedQA / BioASQ |
| PubMedQA（500）/ BioASQ（618） | 教材 | 0.558 / 0.794 | 0.420 / 0.660 | 下降 |
| PubMedQA / BioASQ | **PubMed（2390 万条）** | 0.558 / 0.794 | 0.734† / **0.871** | 上升；Recall@5 0.97 / 0.72 |
| MCMLE（3,426）/ CMB（280） | 华佗百科（中文） | **0.851 / 0.689** | 0.830 / 0.632 | 下降：百科问答覆盖不到考试知识 |

† PubMedQA 不给摘要的结论段，而 PubMed 语料收的是完整摘要：96% 的检索结果包含原文结论，等于泄漏了答案，这个数字不能当作检索收益。

- **核心结论：RAG 有没有用取决于语料能否覆盖问题**。干净的证据是 BioASQ：教材检索 −13.4 个点，PubMed 检索 +7.7 个点。
- Agentic RAG（Qwen3-8B，同一批 1,000 题）：教材语料 0.683，低于不检索（0.729）；**PubMed 语料 0.738，略高于不检索**（+0.9 个点，不显著），
  BioASQ +5.5 个点。工具描述换成"检索科研文献"后，模型在 MMLU 上的检索次数减少了 40%。
  注意：这 1,000 题是每个数据集的前 200 题，PubMedQA 的前 200 题全部是 yes，这一列有偏。

### 3.4 GRPO 对齐与闭环（Qwen3-1.7B，150 步，单卡约 36 分钟 / 种子）

留出集 500 条（3 个随机种子；"最好检查点"按留出集平均奖励挑选）：

| 指标 | 训练前 | 最好检查点 | 最终（step 150） |
|---|---|---|---|
| 答案准确率 | 0.439 | **0.652 ± 0.015** | 0.626 ± 0.037 |
| 引用准确率 | 0.711 | **0.922 ± 0.019** | 0.900 ± 0.043 |
| 过度拒答（↓） | 0.134 | **0.055 ± 0.003** | 0.075 ± 0.024 |
| 拒答召回 | 0.982 | 0.965 ± 0.009 | 0.971 ± 0.005 |

闭环：把最好检查点部署到推理服务，在 MIRAGE PubMedQA 上评测（资料里含原始摘要）：

| 指标 | 原始模型 | GRPO（3 个种子） |
|---|---|---|
| 只用资料作答 | 0.428 | **0.633 ± 0.008** |
| 资料不足时闭卷兜底 | 0.504 | **0.653 ± 0.005** |
| 引用准确率 | 0.650 | **0.913 ± 0.025** |

换成 PubMed 检索（`p12_pubmed_followups.md`）：

| 条件 | 原始模型 | GRPO（3 个种子） |
|---|---|---|
| BioASQ，只用资料 | 0.542 | **0.818 ± 0.016** |
| BioASQ，资料不足时闭卷兜底 | 0.706 | **0.846 ± 0.011**（Qwen3-8B 检索后为 0.871） |
| PubMedQA，原始摘要 + 3 篇相似 PubMed 摘要，引用准确率 | 0.314 | **0.604 ± 0.026** |

- 有据作答的提升在三个种子上非常稳定。闭卷准确率也上升了（0.298 → 0.393 ± 0.043），但主要来自"少答 maybe"这一副作用。
- 1.7B 的 GRPO 模型配合 PubMed 检索，在 BioASQ 上接近 8B 模型检索后的水平，并且超过 8B 闭卷（0.794）。
- 新偏差：训练集按 yes/no 1:1 平衡，模型回答 "no" 的比例高于真实分布。

## 4. 关键工程问题与定位过程（面试素材）

| # | 现象 | 定位 | 解决 |
|---|---|---|---|
| 1 | ITL P99 达 235 ms，中位数却和 vLLM 接近 | P90 ≈ 一个 prefill 步的耗时 → decode 被插队 | 混合批（decode 优先 + prefill 分块） |
| 2 | 调小 token 预算，ITL 不变 | 单步计时：只混入 1 个 token 就从 13 → 42 ms；profiler：GPU 时间几乎相同 | eager 的 kernel 发起开销 → 分段 CUDA Graph |
| 3 | 分段图在 1056 token 的步上比 eager 还慢 | 档位太稀，1056 被补到 1536 | 档位细化 |
| 4 | 训练进程里引擎启动"卡住" | py-spy：每录一张图都执行 `gc.collect()` | `capture_begin/capture_end`，只 GC 一次 |
| 5 | 计划做 overlap 调度 | 拆解单步耗时：CPU 只占 2~4% | **不做**，避免无谓的复杂度 |
| 6 | 混合步的 GPU 时间随 batch 暴涨 | profiler：attention 涨到 4×；微基准一度误判（没测"批里有 prefill"的情况） | varlen kernel 的 GQA 打包失效 → 拆成 decode + varlen 两段 |
| 7 | 上游同配置重跑，贪心输出只有 30/64 一致 | T=1e-5 近似贪心，bf16 并列由随机噪声打破 | `temperature=0` 走 argmax |
| 8 | RL 中 rollout 可能静默 off-policy | 共享 system prompt，前缀缓存复用旧权重的 KV | 每次同步权重后清空前缀缓存 |
| 9 | 不同图片可能共享 KV | 图片占位 token 完全相同 | 图片内容哈希混入块哈希；对照实验 6/8 出错 |
| 10 | 1,061 张图评测 OOM | 排队请求的视觉特征全在 GPU 上 | 特征放锁页内存，按分块拷贝 |
| 11 | 批量视觉编码"反而更慢" | 测量错误：预热不足，且用前 100 张外推 | 公平测量：批量快 2.1× |
| 12 | GRPO 保存的检查点无法部署 | `save_pretrained` 按参数当前类型把 config 写成 fp32 | 保存后显式写入 bf16 |
| 13 | 一台机器上无法同时跑两个引擎 | 上游写死了分布式端口 2333 | 单卡时自动选空闲端口 |
| 14 | GRPO lr 1e-6、数据 yes:no≈15:1 | 纯 bf16 更新被舍入；策略靠"永远答 yes"拿奖励 | fp32 主权重；按标签平衡 |
| 15 | 运行中根分区写满 | 30 GB 根分区被 uv / pip 缓存占满 | 清理缓存，大文件放 EFS |

## 5. 负面结果与局限

- 中低负载下 ITL P99 仍落后 vLLM（35 vs 23 ms）。
- 教材语料和中文百科语料下，检索都降低了准确率；Agent 只在 PubMed 语料下略微超过不检索，且不显著。
- PubMedQA 的 PubMed 检索存在答案泄漏（结论段）；Agent 评测按"前 200 题"抽样，PubMedQA 子集全部是 yes。两者都应改进后重测。
- GRPO：留出集同时用于挑选最好检查点；训练集没有 maybe 类；引入了偏向 "no" 的新偏差。
- Qwen3-VL：含图片的 prefill 走 eager，是当前端到端瓶颈；不支持视频与多卡。
- P99 类指标在 500 条样本上只对应 5 个请求，8 req/s 时波动很大（标准差 17~22 ms），不适合写成精确数字。

## 6. 与开源项目的关系（简历与面试中需如实说明）

| 来自上游 nano-vllm（GeeeekExplorer，MIT） | 本项目实现 / 移植 |
|---|---|
| 分页 KV、按块哈希的前缀缓存、decode CUDA Graph、TP、torch.compile、仅首条序列可分块的 chunked prefill、离线 `generate` | 在线流式服务与取消、混合批调度、真贪心、分段 CUDA Graph、混合步 attention 拆分、Qwen3-VL、多模态前缀缓存、批量视觉编码、图片输入、权重原地同步与缓存失效、`/metrics`、多引擎共存、压测与诊断工具 |

- 混合批参考 **vLLM V1 / Sarathi-Serve**；分段 CUDA Graph 参考 **vLLM V1**；mini-SGLang（MIT）的 overlap 调度经测量后决定不移植。
- Radix Tree 未做：上游已有按块哈希的前缀缓存（vLLM V1 同样用哈希方案），而 flash-attn 分页 KV 要求块大小是 256 的倍数，
  Radix 树带不来更细的粒度。
- 视觉编码器复用 transformers 的 `Qwen3VLVisionModel`；检索模型使用 NCBI MedCPT、BAAI bge-m3 / bge-reranker；数据集均为公开数据。

## 7. 复现

```bash
uv sync                                                    # .venv：torch 2.8 cu126、flash-attn 2.8.3 等
bash scripts/download_assets.sh                            # 模型与数据 → weights/ data/
# 推理引擎
uv run python -m nanovllm.entrypoints.openai_server --model weights/Qwen3-8B --max-num-batched-tokens 1024
uv run python -m bench.serving --tokenizer weights/Qwen3-8B --request-rate 16
bash scripts/run_serving_repeats.sh                        # 上游 / 本项目 / vLLM 各 3 次；用 bench.aggregate 汇总
uv run python -m bench.step_breakdown --model weights/Qwen3-8B
# 多模态
uv run python -m bench.vqa_eval --model weights/Qwen3-VL-4B-Instruct
uv run python -m bench.mm_cache_ablation
# 检索
uv run python -m rag.build_index --corpus data/medrag/textbooks/chunk --out data/indexes/textbooks
uv run python -m rag.eval_mirage --max-per-dataset 1000    # 需先启动上面的服务
uv run python -m rag.pubmed build --shards 24 --workers 12
uv run python -m rag.chinese
bash scripts/run_rag_evals.sh                              # PubMed + 中文评测
bash scripts/run_pubmed_followups.sh                       # PubMed 语料下的闭环评测与 Agent
# GRPO 与闭环
uv run python -m align.task && uv run python -m align.grpo --seed 0 --label seed0
uv run python -m align.summarize_seeds --runs results/grpo/seed0 results/grpo/seed1 results/grpo/seed2
bash scripts/run_grounded_seeds.sh
# 演示（需先启动 VL 与文本两个服务，见 agent/demo.py 文件头）
uv run --extra demo python -m agent.demo
# 单元测试
uv run pytest tests
```

官方 vLLM 对照环境：`.venv-vllm`（vllm 0.11.0，需把 transformers 固定为 4.57.6）。
根分区只有 30 GB，权重、数据、结果都放仓库下被忽略的目录（位于 EFS）。

## 8. 简历写法建议（数字均可在本仓库复现）

**基于 nano-vllm 的医疗问答推理与对齐系统**（个人项目，单卡 A800）
- 基于开源 nano-vllm 深度改造推理引擎：实现异步流式 OpenAI 兼容服务；参考 vLLM V1 实现 prefill/decode 混合批与分段 CUDA Graph；
  通过单步耗时拆解与 profiler 发现混合批中 flash-attn varlen kernel 的 GQA 打包失效，拆分 attention 后高负载吞吐与 vLLM 持平。
  Qwen3-8B 在 16 req/s 下 **ITL P99 降低 86%（248 → 35 ms）、TPOT 降低 58%**（3 次重复）。
- 为引擎接入 Qwen3-VL（交错 M-RoPE、DeepStack、批量视觉编码、OpenAI 图片输入），发现并修复多模态前缀缓存的跨图片 KV 误复用。
- 搭建医疗检索：MedCPT / BM25 / RRF / 重排；为 2390 万条 PubMed 语料设计分片 BM25 + 两阶段检索（8 分钟建完索引），
  BioASQ 准确率 0.794 → 0.871；系统评估并如实报告了语料不匹配时检索有害、以及 PubMedQA 检索存在答案泄漏的问题。
- 手写 GRPO 训练器，用自研引擎做同卡 rollout 并原地同步权重（处理前缀缓存导致的隐性 off-policy）；
  3 个种子下引用准确率 0.71 → 0.92；部署回 PubMed 检索流程后，1.7B 模型在 BioASQ 上从 0.706 提升到 0.846。

## 9. 面试可能追问

- prefill 计算受限、decode 访存受限，混合批为什么能提升？token 预算如何权衡 TTFT / ITL / 吞吐？
- 为什么以 attention 为界切 CUDA Graph？补齐行如何保证不写 KV？档位粒度怎么选？
- flash-attn varlen kernel 在混合批里为什么变慢？GQA 打包是什么？为什么微基准一开始没测出来？
- 为什么 CPU 只占 2~4% 就不做 overlap 调度？什么场景下它会值得做？
- 前缀缓存为什么按块、为什么用链式哈希？多模态为什么必须混入图片哈希？
- GRPO 与 PPO / DPO 的区别；组内奖励全相同为什么没有梯度；每批只更新一次时 clip 为什么不生效；k3 KL 估计的性质。
- RL 里 rollout 引擎和训练权重如何保持一致？前缀缓存为什么会导致 off-policy？
- 为什么同一套检索在教材 / 百科语料上有害、在 PubMed 上有效？PubMedQA 的检索结果为什么会泄漏答案，你是怎么发现的？

## 10. 后续工作

1. 含图片的 prefill 步支持三维位置的分段 CUDA Graph（当前多模态端到端的瓶颈）。
2. 中低负载下的 ITL P99（与 vLLM 差 12 ms）：分析尾部请求的调度时序。
3. 中文教材 / 临床指南语料；用 GRPO 训练 Agent 的"何时检索"；Agent 评测改为随机抽样，PubMedQA 检索时去掉原文结论段。
4. GRPO：用独立验证集挑选检查点，加入 maybe 类数据，修正 yes/no 先验偏差。
5. Qwen3.5 混合注意力（分页 KV 与 DeltaNet 递归状态共存）。
