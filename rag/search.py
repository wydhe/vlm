"""按语料名加载检索器，统一返回 {id, title, content} 格式的结果，给 Agent 和评测脚本共用。

    textbooks   MedRAG 教材，12.6 万段（rag.retrievers 的混合检索 + 重排）
    pubmed      PubMed 摘要，2390 万条（rag.pubmed 的两阶段检索）；id 是 PMID
"""

from __future__ import annotations

from collections.abc import Callable

SearchFn = Callable[[list[str], int], list[list[dict]]]
CORPORA = ("textbooks", "pubmed")


def load_search(corpus: str, query_model: str = "weights/MedCPT-Query-Encoder",
                article_model: str = "weights/MedCPT-Article-Encoder",
                reranker_model: str = "weights/MedCPT-Cross-Encoder") -> SearchFn:
    """返回 search(queries, k)：每个问题检索 k 条，结果是 {id, title, content} 字典的列表。"""
    if corpus == "textbooks":
        from rag.retrievers import HybridRetriever
        retriever = HybridRetriever.from_index("data/medrag/textbooks/chunk", "data/indexes/textbooks",
                                               query_model, reranker_model)

        def search(queries: list[str], k: int) -> list[list[dict]]:
            hits = retriever.search(queries, k=k, mode="hybrid_rerank")
            return [[{"id": h.id, "title": h.title, "content": h.content} for h in row] for row in hits]
        return search

    if corpus == "pubmed":
        from rag.pubmed import PubMedRetriever
        retriever = PubMedRetriever(query_model, article_model, reranker_model)

        def search(queries: list[str], k: int) -> list[list[dict]]:
            hits = retriever.search(queries, k=k)
            return [[{"id": h["pmid"], "title": h["title"], "content": h["content"]} for h in row] for row in hits]
        return search

    raise ValueError(f"unknown corpus {corpus!r}, expected one of {CORPORA}")
