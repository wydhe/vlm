"""中文医学检索：jieba 分词 BM25 + bge-m3 向量检索 → RRF 融合 → bge-reranker 重排。

结构和英文版 rag/retrievers.py 一一对应，只是换成了中文能用的分词和模型：
    英文 BM25（按空格分词）   →  jieba 分词的 BM25
    MedCPT（英文医学向量）    →  bge-m3（多语言向量）
    MedCPT 交叉编码器         →  bge-reranker-v2-m3

语料：华佗百科问答（huatuo_encyclopedia_qa），每条把"问题 + 答案"拼成一段文字。

建索引：
    uv run python -m rag.chinese --corpus data/huatuo_encyclopedia/train_datasets.jsonl \
        --out data/indexes/huatuo
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import bm25s
import jieba
import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from rag.retrievers import rrf_fuse

jieba.setLogLevel(60)   # 关掉 jieba 加载词典时的日志


def load_huatuo(path: str, max_chars: int = 600) -> list[str]:
    """每行 {"questions": [[问题, ...]], "answers": [答案]} → "问题\n答案"。"""
    passages = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            question = item["questions"][0][0]
            answer = item["answers"][0][:max_chars]
            passages.append(f"{question}\n{answer}")
    return passages


def cut(texts: list[str]) -> list[list[str]]:
    """jieba 分词，去掉空白词。"""
    return [[w for w in jieba.lcut(t) if w.strip()] for t in texts]


class BGEEncoder:
    """bge-m3 向量：CLS 向量 + L2 归一化，相似度用内积。"""

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir, dtype=torch.float16).to(device).eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], max_length: int = 512, batch_size: int = 128) -> torch.Tensor:
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[start:start + batch_size], truncation=True, padding=True,
                                   max_length=max_length, return_tensors="pt").to(self.device)
            cls = self.model(**batch).last_hidden_state[:, 0]
            vectors.append(torch.nn.functional.normalize(cls, dim=-1))
        return torch.cat(vectors)


class BGEReranker:
    """bge-reranker-v2-m3：把（问题, 段落）放在一起打分。"""

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, dtype=torch.float16).to(device).eval()

    @torch.inference_mode()
    def score(self, query: str, passages: list[str]) -> list[float]:
        batch = self.tokenizer([query] * len(passages), passages, truncation=True, padding=True,
                               max_length=512, return_tensors="pt").to(self.device)
        return self.model(**batch).logits[:, 0].float().tolist()


class ChineseRetriever:
    """检索入口。mode: bm25 / dense / hybrid / hybrid_rerank。"""

    def __init__(self, index_dir: str, encoder_dir: str, reranker_dir: str, candidates: int = 50):
        index_dir = Path(index_dir)
        self.passages = json.loads((index_dir / "passages.json").read_text())
        self.bm25 = bm25s.BM25.load(str(index_dir / "bm25"))
        self.encoder = BGEEncoder(encoder_dir)
        self.matrix = torch.from_numpy(np.load(index_dir / "dense_fp16.npy")).to(self.encoder.device)
        self.reranker = BGEReranker(reranker_dir)
        self.candidates = candidates

    def search_bm25(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        rows, scores = self.bm25.retrieve(cut(queries), k=k, show_progress=False, n_threads=16)
        return [list(zip(r.tolist(), s.tolist())) for r, s in zip(rows, scores)]

    def search_dense(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        q = self.encoder.encode(queries, max_length=128)
        scores, rows = torch.topk(q @ self.matrix.T, k=k, dim=-1)
        return [list(zip(r.tolist(), s.float().tolist())) for r, s in zip(rows, scores)]

    def search(self, queries: list[str], k: int = 5, mode: str = "hybrid_rerank") -> list[list[str]]:
        """每个问题返回 k 段文字。"""
        if mode == "bm25":
            rankings = self.search_bm25(queries, k)
        elif mode == "dense":
            rankings = self.search_dense(queries, k)
        else:
            n = self.candidates
            bm25_hits, dense_hits = self.search_bm25(queries, n), self.search_dense(queries, n)
            rankings = [rrf_fuse([b, d], k=n) for b, d in zip(bm25_hits, dense_hits)]
            if mode == "hybrid_rerank":
                rankings = [self.rerank(q, r) for q, r in zip(queries, rankings)]
        return [[self.passages[row] for row, _ in r[:k]] for r in rankings]

    def rerank(self, query: str, ranking: list[tuple[int, float]]) -> list[tuple[int, float]]:
        scores = self.reranker.score(query, [self.passages[row] for row, _ in ranking])
        order = sorted(range(len(ranking)), key=lambda i: -scores[i])
        return [(ranking[i][0], scores[i]) for i in order]


def build_index(corpus: str, out: str, encoder_dir: str) -> None:
    """依次写出 passages.json、bm25/、dense_fp16.npy；已经存在的步骤会跳过。"""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    passages_file = out_dir / "passages.json"
    if passages_file.exists():
        passages = json.loads(passages_file.read_text())
    else:
        passages = load_huatuo(corpus)
        passages_file.write_text(json.dumps(passages, ensure_ascii=False))
    print(f"{len(passages)} 段文字")

    if not (out_dir / "bm25").exists():
        t = time.perf_counter()
        index = bm25s.BM25()
        index.index(cut(passages), show_progress=False)
        index.save(str(out_dir / "bm25"))
        print(f"BM25 索引：{time.perf_counter() - t:.0f}s")

    if not (out_dir / "dense_fp16.npy").exists():
        t = time.perf_counter()
        order = sorted(range(len(passages)), key=lambda i: len(passages[i]))   # 按长度排序，减少 padding
        vectors = BGEEncoder(encoder_dir).encode([passages[i] for i in order]).cpu().numpy()
        dense = np.empty_like(vectors)
        dense[order] = vectors
        np.save(out_dir / "dense_fp16.npy", dense.astype(np.float16))
        print(f"向量索引 {dense.shape}：{time.perf_counter() - t:.0f}s")


def main() -> None:
    p = argparse.ArgumentParser(description="建立中文医学检索索引")
    p.add_argument("--corpus", default="data/huatuo_encyclopedia/train_datasets.jsonl")
    p.add_argument("--out", default="data/indexes/huatuo")
    p.add_argument("--encoder", default="weights/bge-m3")
    args = p.parse_args()
    build_index(args.corpus, args.out, args.encoder)


if __name__ == "__main__":
    main()
