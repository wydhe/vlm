"""PubMed 检索评测：MIRAGE 的 PubMedQA 和 BioASQ（这两个数据集考的是科研文献结论）。

P5 用教材做语料时，检索反而降低了这两个数据集的准确率，原因是教材里没有答案。
这里换成 PubMed 全量语料，比较三种情况：
    none        不检索
    textbook    教材语料（混合检索 + 重排）
    pubmed      PubMed 语料（分片 BM25 → MedCPT 融合 → 重排）

除了准确率，还统计 Recall@k：题目标注的原始文献（PMID）是否出现在检索结果里。

用法（先启动推理服务）：
    uv run python -m rag.eval_pubmed --model Qwen3-8B --label qwen3-8b-pubmed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from rag.eval_mirage import ask_all, build_prompt, load_mirage, parse_letter, summarize_accuracy
from rag.pubmed import PubMedRetriever
from rag.retrievers import Hit, HybridRetriever


def recall_at_k(questions: list[dict], pmid_lists: list[list[int]]) -> dict[str, float]:
    """每个数据集里，检索结果包含至少一篇标注文献的题目比例。"""
    hits: dict[str, list[bool]] = {}
    for q, pmids in zip(questions, pmid_lists):
        gold = {int(p) for p in q.get("PMID", [])}
        hits.setdefault(q["dataset"], []).append(bool(gold & set(pmids)))
    return {name: sum(v) / len(v) for name, v in hits.items()}


def retrieve(mode: str, questions: list[dict], args: argparse.Namespace) -> tuple[list, list[list[int]]]:
    """返回 (每题的文档列表, 每题检索到的 PMID 列表)。"""
    queries = [q["question"] for q in questions]
    if mode == "none":
        return [None] * len(questions), [[] for _ in questions]
    if mode == "textbook":
        retriever = HybridRetriever.from_index(args.textbook_corpus, args.textbook_index,
                                               args.query_model, args.reranker_model)
        return retriever.search(queries, k=args.k, mode="hybrid_rerank"), [[] for _ in questions]
    retriever = PubMedRetriever(args.query_model, args.article_model, args.reranker_model)
    results = retriever.search(queries, k=args.k)
    hits = [[Hit(str(d["pmid"]), d["title"], d["content"], 0.0) for d in docs] for docs in results]
    return hits, [[d["pmid"] for d in docs] for docs in results]


def main() -> None:
    p = argparse.ArgumentParser(description="PubMed 检索评测")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen3-8B")
    p.add_argument("--mirage", default="data/mirage/benchmark.json")
    p.add_argument("--modes", nargs="+", default=["none", "textbook", "pubmed"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--textbook-corpus", default="data/medrag/textbooks/chunk")
    p.add_argument("--textbook-index", default="data/indexes/textbooks")
    p.add_argument("--query-model", default="weights/MedCPT-Query-Encoder")
    p.add_argument("--article-model", default="weights/MedCPT-Article-Encoder")
    p.add_argument("--reranker-model", default="weights/MedCPT-Cross-Encoder")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--label", default="pubmed")
    p.add_argument("--out-dir", default="results/rag")
    args = p.parse_args()

    questions = load_mirage(args.mirage, ["pubmedqa", "bioasq"], None)
    print(f"{len(questions)} 道题")
    record = {"config": vars(args), "results": {}, "items": {}}
    for mode in args.modes:
        start = time.perf_counter()
        hits, pmid_lists = retrieve(mode, questions, args)
        retrieval_seconds = time.perf_counter() - start

        prompts = [build_prompt(q, h) for q, h in zip(questions, hits)]
        replies = asyncio.run(ask_all(args.base_url, args.model, prompts, args.concurrency))
        predictions = [parse_letter(r, set(q["options"])) for r, q in zip(replies, questions)]

        result = summarize_accuracy(questions, predictions)
        if mode == "pubmed":
            result["recall_at_k"] = recall_at_k(questions, pmid_lists)
        result["unparsed"] = sum(pred is None for pred in predictions)
        result["retrieval_s"] = round(retrieval_seconds, 1)
        record["results"][mode] = result
        record["items"][mode] = [{"dataset": q["dataset"], "qid": q["qid"], "pred": pred,
                                  "answer": q["answer"], "pmids": pmids}
                                 for q, pred, pmids in zip(questions, predictions, pmid_lists)]
        scores = "  ".join(f"{name} {r['acc']:.3f}" for name, r in result.items()
                           if isinstance(r, dict) and "acc" in r)
        recall = f"  Recall@{args.k} {result['recall_at_k']}" if "recall_at_k" in result else ""
        print(f"{mode:9} {scores}{recall}  | 未解析 {result['unparsed']}  检索 {retrieval_seconds:.0f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.label}.json"
    out_file.write_text(json.dumps(record, indent=1))
    print(f"结果已保存到 {out_file}")


if __name__ == "__main__":
    main()
