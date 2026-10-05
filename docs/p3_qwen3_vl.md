# P3：Qwen3-VL 多模态推理与多模态前缀缓存

模型：Qwen3-VL-4B-Instruct（36 层文本解码器 + 24 层 ViT，DeepStack 取 ViT 第 5/11/17 层）。

## 实现（`engine/nanovllm/models/qwen3_vl.py` 及引擎改动）

| Qwen3-VL 特性 | 在引擎中的实现 |
|---|---|
| 视觉编码器 | 复用 HF `Qwen3VLVisionModel`，每个请求加入时编码一次（`ModelRunner.encode_images`） |
| 图像占位 token | prefill 时按当前分块取出对应的视觉特征行，替换 embedding（`Context.mm_embeds`） |
| DeepStack | 第 0/1/2 层输出后在图像 token 位置加上对应 ViT 层特征；nano-vllm 的 hidden/residual 分离存储，真实隐状态为二者之和，加到任一项等价 |
| 交错 M-RoPE | `MRotaryEmbedding`：频率槽 1::3 / 2::3（前 3×20 个）取 h / w 角度。**文本 token 的 t=h=w 时与普通 RoPE 数学等价**（单测验证），所以 decode 继续用 1-D 位置 `i + delta`，原有 decode CUDA Graph 无需改动；只有含图像的 prefill 步用 3-D 位置、走 eager |
| 位置计算 | `mrope_positions`：`get_rope_index` 的图像部分移植，算出 `[3, L]` 位置与 `delta` |
| 权重加载 | `hf_name_mapping`：`model.language_model.*` → `model.*`，`model.visual.*` → `visual.*` |
| 显存 | 视觉特征放在 CPU 锁页内存，prefill 时只拷贝当前分块的行；请求结束 / 取消时释放（否则 1061 个排队请求的特征把 GPU 撑爆，实测 OOM） |

### 多模态前缀缓存的正确性问题

前缀缓存按 token id 哈希。**所有图片的占位 token 完全相同**，若只按 token 哈希，两张尺寸相同的不同图片会命中同一块 KV。
解决：从第一个图像 token 所在块起，把图片内容哈希（像素 + 网格）混入块哈希（`BlockManager._extra`）。

对照实验（`bench/mm_cache_ablation.py`）：SLAKE 中 token 完全相同（同网格尺寸）、部位不同的 8 对图片，先问 A 再问 B：

| | B 复用了 A 的 KV | B 的答案被带偏 |
|---|---|---|
| 去掉图片哈希 | **6 / 8** | **4 / 8**（如脑部图像被答成 "Thorax"） |
| 本实现 | 0 / 8 | 0 / 8（与冷缓存单独提问一致） |

探针（`bench/vqa_eval.py`，从空缓存开始，284 token 的 prompt）：首次请求命中 0；同一请求再次命中 256（一整块）；**镜像翻转图片（token 完全相同、像素不同）命中 0**。

## 正确性：与 HF transformers 逐 token 对比（SLAKE 前 64 题，贪心 32 token）

- **61 / 64 完全一致**。3 处分歧都在首 token，且都是同一答案的大小写 / 前导空格变体：`' lung'` vs `' Lung'`、`' no'` vs `'No'`、`'Yes'` vs `' Yes'`。
- HF 自身在这 64 题中有 5 题首 token 的 top-2 logit 差 ≤ 0.125（4 题 bf16 下完全并列）；分歧的第 55 题即属完全并列。
  另两题 HF 的差值 > 0.25，归因于两边实现在 bf16 下的数值差异（未逐层验证）。
- 纯文本回归：Qwen3-0.6B 的贪心输出在引入多模态改动前后 **64 / 64 完全一致**。

## SLAKE 英文测试集（1,061 题，贪心）

| 题型 | n | 准确率 |
|---|---|---|
| 封闭题（yes/no 等） | 416 | 0.810 |
| 开放题（精确匹配或答案包含于输出） | 645 | 0.459 |
| 总体 | 1,061 | 0.597 |

吞吐（含视觉编码）：本引擎批处理 1,061 题 **9.33 req/s**；HF `generate` 逐条 6.57 req/s（前 64 题）。
答案只有几个 token，瓶颈在视觉编码——目前在 `add_request` 中**逐张串行**执行，未批处理，这是下一步的直接优化点。

## 局限

- 仅支持图像（不含视频）、单卡；服务端 OpenAI 接口尚未接入图像输入（目前通过离线 `LLM.generate(..., mm_inputs=...)` 使用）。
- 开放题评分采用精确匹配 / 包含匹配，对同义表达（如 "Thorax" vs "Chest"）偏严。
- 含图像的 prefill 步走 eager（3-D 位置与 DeepStack 不在分段 CUDA Graph 内）。
