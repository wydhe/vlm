"""Two-stage medical retrieval: BM25 + MedCPT dense -> RRF fusion -> MedCPT cross-encoder rerank.

Index layout under ``<index_dir>``::

    ids.json            chunk ids, row-aligned with the dense matrix
    dense_fp16.npy      [N, 768] MedCPT article embeddings (CLS), fp16
    bm25/               bm25s index (stemmed English)

Chunks come from MedRAG corpora: one JSON object per line ``{"id", "title", "content"}``.
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
    """One retrieved chunk with the score of the stage that produced it."""

    id: str
    title: str
    content: str
    score: float


def load_chunks(corpus_dir: str | Path) -> list[dict]:
    """Load every ``*.jsonl`` chunk file of a MedRAG corpus directory, sorted by file name."""
    chunks: list[dict] = []
    for path in sorted(Path(corpus_dir).glob("*.jsonl")):
        with open(path) as f:
            chunks.extend(json.loads(line) for line in f if line.strip())
    return chunks


# --------------------------------------------------------------------------- sparse

class BM25Retriever:
    """Lexical retrieval with bm25s (Snowball-stemmed English, stopwords removed)."""

    def __init__(self, retriever: bm25s.BM25):
        self.retriever = retriever
        self.stemmer = Stemmer.Stemmer("english")

    @classmethod
    def build(cls, chunks: list[dict]) -> BM25Retriever:
        """Index ``title + content`` of every chunk."""
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
        """Return, per query, ``k`` (row, score) pairs sorted by score."""
        tokens = bm25s.tokenize(queries, stopwords="en", stemmer=self.stemmer, show_progress=False)
        rows, scores = self.retriever.retrieve(tokens, k=k, show_progress=False, n_threads=8)
        return [list(zip(r.tolist(), s.tolist())) for r, s in zip(rows, scores)]


# --------------------------------------------------------------------------- dense

class MedCPTEncoder:
    """MedCPT bi-encoder (NCBI): separate query and article encoders, CLS pooling, dot product."""

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
        """Return ``[len(queries), 768]`` fp16 query embeddings on ``device``."""
        out = []
        for i in range(0, len(queries), batch_size):
            enc = self.q_tok(queries[i:i + batch_size], truncation=True, padding=True,
                             max_length=64, return_tensors="pt").to(self.device)
            out.append(self.q_enc(**enc).last_hidden_state[:, 0, :])
        return torch.cat(out)

    @torch.inference_mode()
    def encode_articles(self, chunks: list[dict], batch_size: int = 256) -> np.ndarray:
        """Return ``[len(chunks), 768]`` fp16 article embeddings (title/content pairs, max 512 tokens)."""
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
    """Exact inner-product search over a GPU-resident embedding matrix."""

    def __init__(self, encoder: MedCPTEncoder, embeddings: np.ndarray):
        self.encoder = encoder
        self.matrix = torch.from_numpy(embeddings).to(encoder.device)

    def search(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        q = self.encoder.encode_queries(queries)
        scores, rows = torch.topk(q @ self.matrix.T, k=k, dim=-1)
        return [list(zip(r.tolist(), s.float().tolist())) for r, s in zip(rows, scores)]


# --------------------------------------------------------------------------- fusion / rerank

def rrf_fuse(rankings: list[list[tuple[int, float]]], k: int, c: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: ``score(d) = sum_r 1 / (c + rank_r(d))`` over input rankings."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (row, _) in enumerate(ranking, start=1):
            fused[row] = fused.get(row, 0.0) + 1.0 / (c + rank)
    return sorted(fused.items(), key=lambda x: -x[1])[:k]


class CrossEncoderReranker:
    """MedCPT cross-encoder: scores (query, article) pairs jointly."""

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


# --------------------------------------------------------------------------- pipeline

class HybridRetriever:
    """BM25 + dense candidates, RRF fused, optionally cross-encoder reranked."""

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
        """Load chunks, BM25 and dense indexes built by ``rag.build_index``."""
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
        """Retrieve ``k`` chunks per query.

        Args:
            queries: Natural-language queries.
            k: Chunks returned per query.
            mode: One of ``HybridRetriever.MODES``.
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
