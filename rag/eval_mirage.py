"""MIRAGE 端到端 RAG 评测（MedQA、MedMCQA、PubMedQA、BioASQ、MMLU-Med）。

对每种检索方式：把检索到的 top-k 片段放进 prompt，让推理服务（OpenAI 兼容接口，
例如本项目的 nano-vllm 服务）只回答选项字母，统计准确率和 95% 置信区间。

用法：
    uv run python -m rag.eval_mirage --modes none bm25 dense hybrid hybrid_rerank \
        --max-per-dataset 500 --label qwen3-8b-textbooks
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import time
from pathlib import Path

import aiohttp

from rag.retrievers import Hit, HybridRetriever

SYSTEM = ("You are a medical expert. Use the provided documents if they are relevant. "
          "Answer the multiple-choice question with the option letter only.")
LETTER = re.compile(r"\b([A-E])\b")


def load_mirage(path: str, datasets: list[str], max_per_dataset: int | None,
                seed: int = 0) -> list[dict]:
    """把 MIRAGE 展开成 {dataset, qid, question, options, answer} 的列表。

    题目超过 max_per_dataset 时按固定种子随机抽样，保证每次抽到同一批题。
    """
    bench = json.loads(Path(path).read_text())
    rng = random.Random(seed)
    questions = []
    for name in datasets:
        items = [{"dataset": name, "qid": qid, **q} for qid, q in bench[name].items()]
        if max_per_dataset and len(items) > max_per_dataset:
            items = rng.sample(items, max_per_dataset)
        questions.extend(items)
    return questions


def build_prompt(q: dict, hits: list[Hit] | None, max_chars: int = 1500) -> str:
    """拼出 prompt：检索到的文档（如果有）+ 题目 + 选项。"""
    parts = []
    if hits:
        docs = "\n".join(f"[{i}] ({h.title}) {h.content[:max_chars]}" for i, h in enumerate(hits, 1))
        parts.append(f"Documents:\n{docs}\n")
    options = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    parts.append(f"Question: {q['question']}\nOptions:\n{options}\n\nAnswer with a single letter.")
    return "\n".join(parts)


def parse_letter(text: str, valid: set[str]) -> str | None:
    """回复里第一个单独出现、且是合法选项的字母。"""
    for match in LETTER.finditer(text.strip()):
        if match.group(1) in valid:
            return match.group(1)
    return None


def accuracy(correct: list[bool]) -> dict[str, float]:
    """准确率，以及 95% 置信区间的半宽（正态近似）。"""
    n = len(correct)
    if n == 0:
        return {"n": 0, "acc": float("nan"), "ci95": float("nan")}
    acc = sum(correct) / n
    return {"n": n, "acc": acc, "ci95": 1.96 * math.sqrt(acc * (1 - acc) / n)}


async def ask(session: aiohttp.ClientSession, url: str, model: str, prompt: str,
              sem: asyncio.Semaphore) -> str:
    """一次贪心对话补全（关闭 Qwen3 的 thinking）。"""
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 8,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with sem, session.post(url, json=payload) as resp:
        body = await resp.json()
    if "choices" not in body:
        return f"<error {body.get('error')}>"
    return body["choices"][0]["message"]["content"]


async def ask_all(base_url: str, model: str, prompts: list[str], concurrency: int) -> list[str]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as session:
        return await asyncio.gather(*(ask(session, url, model, p, sem) for p in prompts))


def summarize_accuracy(questions: list[dict], predictions: list[str | None]) -> dict:
    """按数据集统计准确率，再算各数据集的平均。"""
    by_dataset: dict[str, list[bool]] = {}
    for q, pred in zip(questions, predictions):
        by_dataset.setdefault(q["dataset"], []).append(pred == q["answer"])
    result = {name: accuracy(correct) for name, correct in by_dataset.items()}
    result["avg"] = {"acc": sum(r["acc"] for r in result.values()) / len(result)}
    return result


def evaluate_mode(mode: str, questions: list[dict], retriever: HybridRetriever | None,
                  args: argparse.Namespace) -> tuple[dict, list[dict]]:
    """评测一种检索方式，返回汇总结果和逐题记录。"""
    start = time.perf_counter()
    if mode == "none":
        hits = [None] * len(questions)
    else:
        hits = retriever.search([q["question"] for q in questions], k=args.k, mode=mode)
    retrieval_seconds = time.perf_counter() - start

    prompts = [build_prompt(q, h) for q, h in zip(questions, hits)]
    replies = asyncio.run(ask_all(args.base_url, args.model, prompts, args.concurrency))
    predictions = [parse_letter(r, set(q["options"])) for r, q in zip(replies, questions)]

    result = summarize_accuracy(questions, predictions)
    result["unparsed"] = sum(pred is None for pred in predictions)
    result["retrieval_s"] = round(retrieval_seconds, 1)
    result["total_s"] = round(time.perf_counter() - start, 1)

    items = []
    for q, pred, hit_list in zip(questions, predictions, hits):
        items.append({"dataset": q["dataset"], "qid": q["qid"], "pred": pred, "answer": q["answer"],
                      "docs": [h.id for h in hit_list] if hit_list else []})
    return result, items


def main() -> None:
    p = argparse.ArgumentParser(description="MIRAGE 端到端 RAG 评测")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen3-8B")
    p.add_argument("--mirage", default="data/mirage/benchmark.json")
    p.add_argument("--datasets", nargs="+", default=["medqa", "medmcqa", "pubmedqa", "bioasq", "mmlu"])
    p.add_argument("--max-per-dataset", type=int, default=None)
    p.add_argument("--modes", nargs="+", default=["none", *HybridRetriever.MODES])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--corpus", default="data/medrag/textbooks/chunk")
    p.add_argument("--index", default="data/indexes/textbooks")
    p.add_argument("--query-model", default="weights/MedCPT-Query-Encoder")
    p.add_argument("--reranker-model", default="weights/MedCPT-Cross-Encoder")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--label", default="run")
    p.add_argument("--out-dir", default="results/rag")
    args = p.parse_args()

    questions = load_mirage(args.mirage, args.datasets, args.max_per_dataset)
    print(f"{len(questions)} 道题，来自 {args.datasets}")
    retriever = None
    if any(mode != "none" for mode in args.modes):
        retriever = HybridRetriever.from_index(args.corpus, args.index, args.query_model,
                                               args.reranker_model)

    record = {"config": vars(args), "results": {}, "items": {}}
    for mode in args.modes:
        result, items = evaluate_mode(mode, questions, retriever, args)
        record["results"][mode] = result
        record["items"][mode] = items
        scores = "  ".join(f"{name} {r['acc']:.3f}" for name, r in result.items()
                           if isinstance(r, dict) and "acc" in r)
        print(f"{mode:14} {scores}  | 未解析 {result['unparsed']}  "
              f"检索 {result['retrieval_s']:.0f}s  总计 {result['total_s']:.0f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.label}.json"
    out_file.write_text(json.dumps(record, indent=1))
    print(f"结果已保存到 {out_file}")


if __name__ == "__main__":
    main()
