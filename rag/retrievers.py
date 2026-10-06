"""两阶段医学检索：BM25 + MedCPT 向量检索 → RRF 融合 → MedCPT 交叉编码器重排。

索引目录的内容：

    ids.json            片段 id，和向量矩阵的行一一对应
    dense_fp16.npy      [N, 768] 的 MedCPT 文章向量（CLS），fp16
    bm25/               bm25s 索引（英文词干化）

语料来自 MedRAG：每行一个 JSON 对象 {"id", "title", "content"}。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np
import Stemmer
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


@dataclass(slots=True)
class Hit:
    """一条检索结果，score 是产生它的那一阶段给的分数。"""

    id: str
    title: str
    content: str
    score: float


def load_chunks(corpus_dir: str | Path) -> list[dict]:
    """按文件名顺序读取 MedRAG 语料目录下所有 *.jsonl 文件。"""
    chunks: list[dict] = []
    for path in sorted(Path(corpus_dir).glob("*.jsonl")):
        with open(path) as f:
            chunks.extend(json.loads(line) for line in f if line.strip())
    return chunks


# --------------------------------------------------------------------------- 稀疏检索

class BM25Retriever:
    """用 bm25s 做关键词检索（英文 Snowball 词干化，去停用词）。"""

    def __init__(self, retriever: bm25s.BM25):
        self.retriever = retriever
        self.stemmer = Stemmer.Stemmer("english")

    @classmethod
    def build(cls, chunks: list[dict]) -> BM25Retriever:
        """对每个片段的 "标题 + 正文" 建索引。"""
        stemmer = Stemmer.Stemmer("english")
        tokens = bm25s.tokenize([f"{c['title']} {c['content']}" for c in chunks],
                                stopwords="en", stemmer=stemmer, show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(tokens, show_progress=False)
        return cls(retriever)

    def save(self, path: str | Path) -> None:
        self.retriever.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> BM25Retriever:
        return cls(bm25s.BM25.load(str(path)))

    def search(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        """每个问题返回 k 个 (行号, 分数)，按分数从高到低排序。"""
        tokens = bm25s.tokenize(queries, stopwords="en", stemmer=self.stemmer, show_progress=False)
        rows, scores = self.retriever.retrieve(tokens, k=k, show_progress=False, n_threads=8)
        return [list(zip(r.tolist(), s.tolist())) for r, s in zip(rows, scores)]


# --------------------------------------------------------------------------- 稠密检索

class MedCPTEncoder:
    """MedCPT 双塔模型（NCBI）：问题和文章用两个不同的编码器，取 CLS 向量，用内积算相似度。"""

    def __init__(self, query_model: str, article_model: str | None = None, device: str = "cuda"):
        self.device = device
        self.q_tok = AutoTokenizer.from_pretrained(query_model)
        self.q_enc = AutoModel.from_pretrained(query_model, dtype=torch.float16).to(device).eval()
        self.a_tok = self.a_enc = None
        if article_model:
            self.a_tok = AutoTokenizer.from_pretrained(article_model)
            self.a_enc = AutoModel.from_pretrained(article_model, dtype=torch.float16).to(device).eval()

    @torch.inference_mode()
    def encode_queries(self, queries: list[str], batch_size: int = 256) -> torch.Tensor:
        """返回问题向量，形状 [问题数, 768]，fp16，在 device 上。"""
        out = []
        for i in range(0, len(queries), batch_size):
            enc = self.q_tok(queries[i:i + batch_size], truncation=True, padding=True,
                             max_length=64, return_tensors="pt").to(self.device)
            out.append(self.q_enc(**enc).last_hidden_state[:, 0, :])
        return torch.cat(out)

    @torch.inference_mode()
    def encode_articles(self, chunks: list[dict], batch_size: int = 256) -> np.ndarray:
        """返回文章向量，形状 [片段数, 768]，fp16（输入是 标题/正文 对，最多 512 token）。"""
        assert self.a_enc is not None, "article encoder not loaded"
        out = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            enc = self.a_tok([c["title"] for c in batch], [c["content"] for c in batch],
                             truncation=True, padding=True, max_length=512,
                             return_tensors="pt").to(self.device)
            out.append(self.a_enc(**enc).last_hidden_state[:, 0, :].cpu())
        return torch.cat(out).numpy().astype(np.float16)


class DenseRetriever:
    """向量矩阵放在 GPU 上，用内积做精确检索。"""

    def __init__(self, encoder: MedCPTEncoder, embeddings: np.ndarray):
        self.encoder = encoder
        self.matrix = torch.from_numpy(embeddings).to(encoder.device)

    def search(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        q = self.encoder.encode_queries(queries)
        scores, rows = torch.topk(q @ self.matrix.T, k=k, dim=-1)
        return [list(zip(r.tolist(), s.float().tolist())) for r, s in zip(rows, scores)]


# --------------------------------------------------------------------------- 融合与重排

def rrf_fuse(rankings: list[list[tuple[int, float]]], k: int, c: int = 60) -> list[tuple[int, float]]:
    """倒数排名融合（RRF）：score(d) = Σ 1 / (c + d 在各个排名里的名次)。"""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (row, _) in enumerate(ranking, start=1):
            fused[row] = fused.get(row, 0.0) + 1.0 / (c + rank)
    return sorted(fused.items(), key=lambda x: -x[1])[:k]


class CrossEncoderReranker:
    """MedCPT 交叉编码器：把（问题, 文章）放在一起打分。"""

    def __init__(self, model: str, device: str = "cuda"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, dtype=torch.float16).to(device).eval()

    @torch.inference_mode()
    def score(self, query: str, docs: list[dict], batch_size: int = 64) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            enc = self.tok([query] * len(batch), [f"{d['title']} {d['content']}" for d in batch],
                           truncation=True, padding=True, max_length=512,
                           return_tensors="pt").to(self.device)
            scores.extend(self.model(**enc).logits[:, 0].float().tolist())
        return scores


# --------------------------------------------------------------------------- 完整流程

class HybridRetriever:
    """BM25 和向量检索各取候选，RRF 融合，可选交叉编码器重排。"""

    MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")

    def __init__(self, chunks: list[dict], bm25: BM25Retriever, dense: DenseRetriever,
                 reranker: CrossEncoderReranker | None = None, candidates: int = 50):
        self.chunks = chunks
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.candidates = candidates

    @classmethod
    def from_index(cls, corpus_dir: str | Path, index_dir: str | Path, query_model: str,
                   reranker_model: str | None = None, device: str = "cuda") -> HybridRetriever:
        """加载语料，以及 rag.build_index 建好的 BM25 和向量索引。"""
        index_dir = Path(index_dir)
        chunks = load_chunks(corpus_dir)
        ids = json.loads((index_dir / "ids.json").read_text())
        assert ids == [c["id"] for c in chunks], "index does not match corpus; rebuild it"
        dense = DenseRetriever(MedCPTEncoder(query_model, device=device),
                               np.load(index_dir / "dense_fp16.npy"))
        reranker = CrossEncoderReranker(reranker_model, device) if reranker_model else None
        return cls(chunks, BM25Retriever.load(index_dir / "bm25"), dense, reranker)

    def _hits(self, ranking: list[tuple[int, float]]) -> list[Hit]:
        return [Hit(self.chunks[r]["id"], self.chunks[r]["title"], self.chunks[r]["content"], s)
                for r, s in ranking]

    def search(self, queries: list[str], k: int = 5, mode: str = "hybrid_rerank") -> list[list[Hit]]:
        """每个问题检索 k 个片段。

        Args:
            queries: 自然语言问题。
            k: 每个问题返回几个片段。
            mode: HybridRetriever.MODES 之一。
        """
        assert mode in self.MODES, mode
        if mode == "bm25":
            return [self._hits(r) for r in self.bm25.search(queries, k)]
        if mode == "dense":
            return [self._hits(r) for r in self.dense.search(queries, k)]
        n = self.candidates
        fused = [rrf_fuse([b, d], k=n) for b, d in
                 zip(self.bm25.search(queries, n), self.dense.search(queries, n))]
        if mode == "hybrid" or self.reranker is None:
            return [self._hits(f[:k]) for f in fused]
        out = []
        for q, cand in zip(queries, fused):
            docs = [self.chunks[r] for r, _ in cand]
            scores = self.reranker.score(q, docs)
            order = sorted(range(len(cand)), key=lambda i: -scores[i])[:k]
            out.append(self._hits([(cand[i][0], scores[i]) for i in order]))
        return out
