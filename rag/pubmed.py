"""PubMed 全量语料（2390 万条摘要片段，约 66 GB）的检索。

语料太大，不能像教材那样整个读进内存，所以做了三件事：
    1. 分片建 BM25：1166 个文件分成若干片，多进程并行建索引；查询时用 mmap 加载
    2. 正文不进内存：只记录每条在哪个文件、哪个字节偏移、PMID，要用时再按偏移读出来
    3. 两阶段检索：BM25 每片取 top-N → 合并成全局 top-N → 现场算 MedCPT 向量做 RRF 融合
       → 交叉编码器重排。这样 GPU 只需要处理候选文档，不用给 2390 万条全部算向量。

近似之处：每个分片的 IDF 是各自算的，BM25 分数跨分片并不严格可比。分片按文件划分、
每片约一百万条，影响很小。

建索引（只用 CPU）：
    uv run python -m rag.pubmed build --shards 24 --workers 12
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import bm25s
import numpy as np
import Stemmer
import torch

from rag.retrievers import CrossEncoderReranker, MedCPTEncoder, rrf_fuse

CORPUS_DIR = Path("data/medrag/pubmed/chunk")
INDEX_DIR = Path("data/indexes/pubmed")


# ---------------------------------------------------------------------- 建索引

def build_shard(task: tuple[int, list[str]]) -> str:
    """为一个分片建 BM25 索引，并记录每条文档的 (文件编号, 字节偏移, PMID)。"""
    shard_id, file_names = task
    texts, file_ids, offsets, pmids = [], [], [], []
    for name in file_names:
        file_id = int(name.removeprefix("pubmed23n").removesuffix(".jsonl"))
        with open(CORPUS_DIR / name, "rb") as f:
            offset = 0
            for line in f:
                doc = json.loads(line)
                texts.append(f"{doc['title']} {doc['content']}")
                file_ids.append(file_id)
                offsets.append(offset)
                pmids.append(int(doc["PMID"]))
                offset += len(line)

    stemmer = Stemmer.Stemmer("english")
    tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False)
    del texts
    index = bm25s.BM25()
    index.index(tokens, show_progress=False)

    out = INDEX_DIR / f"shard_{shard_id:02d}"
    index.save(str(out / "bm25"))
    np.savez(out / "docs.npz", file_ids=np.array(file_ids, dtype=np.int32),
             offsets=np.array(offsets, dtype=np.int64), pmids=np.array(pmids, dtype=np.int64))
    return f"shard {shard_id}: {len(file_ids)} docs"


def build(num_shards: int, workers: int) -> None:
    files = sorted(p.name for p in CORPUS_DIR.glob("*.jsonl"))
    tasks = [(i, files[i::num_shards]) for i in range(num_shards)]     # 交错分配，各片大小接近
    start = time.perf_counter()
    with Pool(workers) as pool:
        for message in pool.imap_unordered(build_shard, tasks):
            print(f"{message}  ({time.perf_counter() - start:.0f}s)", flush=True)


# ---------------------------------------------------------------------- 检索

class PubMedRetriever:
    """BM25（分片）→ MedCPT 向量 RRF 融合 → 交叉编码器重排。"""

    def __init__(self, query_model: str, article_model: str, reranker_model: str):
        self.shards = []
        for shard_dir in sorted(INDEX_DIR.glob("shard_*")):
            docs = np.load(shard_dir / "docs.npz")
            bm25 = bm25s.BM25.load(str(shard_dir / "bm25"), mmap=True)
            self.shards.append((bm25, docs["file_ids"], docs["offsets"], docs["pmids"]))
        self.stemmer = Stemmer.Stemmer("english")
        self.encoder = MedCPTEncoder(query_model, article_model)
        self.reranker = CrossEncoderReranker(reranker_model)

    def read_doc(self, file_id: int, offset: int) -> dict:
        """按字节偏移从原始文件里读出一条文档。"""
        with open(CORPUS_DIR / f"pubmed23n{file_id:04d}.jsonl", "rb") as f:
            f.seek(offset)
            return json.loads(f.readline())

    def bm25_candidates(self, queries: list[str], n: int) -> list[list[dict]]:
        """每个分片取 top-n，按 BM25 分数合并成全局 top-n。"""
        tokens = bm25s.tokenize(queries, stopwords="en", stemmer=self.stemmer, show_progress=False)
        merged: list[list[tuple[float, int, int, int]]] = [[] for _ in queries]
        for bm25, file_ids, offsets, pmids in self.shards:
            rows, scores = bm25.retrieve(tokens, k=n, show_progress=False, n_threads=16)
            for q, (row_list, score_list) in enumerate(zip(rows, scores)):
                for row, score in zip(row_list, score_list):
                    merged[q].append((float(score), int(file_ids[row]), int(offsets[row]), int(pmids[row])))
        results = []
        for candidates in merged:
            candidates.sort(key=lambda c: -c[0])
            results.append([{"bm25": s, "file_id": f, "offset": o, "pmid": p} for s, f, o, p in candidates[:n]])
        return results

    @torch.inference_mode()
    def search(self, queries: list[str], k: int = 5, candidates: int = 100, rerank_top: int = 30) -> list[list[dict]]:
        """返回每个问题的 k 篇文档：{pmid, title, content}。"""
        all_candidates = self.bm25_candidates(queries, candidates)
        query_vectors = self.encoder.encode_queries(queries)
        results = []
        for q, (query, cands) in enumerate(zip(queries, all_candidates)):
            docs = [self.read_doc(c["file_id"], c["offset"]) for c in cands]
            # 阶段二：给候选现场算 MedCPT 向量，和 BM25 排名做 RRF 融合
            article_vectors = torch.from_numpy(self.encoder.encode_articles(docs)).to(query_vectors.device)
            dense_scores = (article_vectors @ query_vectors[q]).float().tolist()
            bm25_rank = [(i, c["bm25"]) for i, c in enumerate(cands)]
            dense_rank = sorted(enumerate(dense_scores), key=lambda x: -x[1])
            fused = rrf_fuse([bm25_rank, dense_rank], k=rerank_top)
            # 阶段三：交叉编码器重排
            top_docs = [docs[i] for i, _ in fused]
            scores = self.reranker.score(query, top_docs)
            order = sorted(range(len(top_docs)), key=lambda i: -scores[i])[:k]
            results.append([{"pmid": int(top_docs[i]["PMID"]), "title": top_docs[i]["title"],
                             "content": top_docs[i]["content"]} for i in order])
        return results


def main() -> None:
    p = argparse.ArgumentParser(description="PubMed 全量语料检索")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="建分片 BM25 索引")
    b.add_argument("--shards", type=int, default=24)
    b.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    if args.command == "build":
        build(args.shards, args.workers)


if __name__ == "__main__":
    main()
