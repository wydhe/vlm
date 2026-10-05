"""End-to-end RAG evaluation on MIRAGE (MedQA, MedMCQA, PubMedQA, BioASQ, MMLU-Med).

For every question and retrieval mode, the top-k chunks are put in the prompt and the
served model (OpenAI-compatible ``/v1/chat/completions``, e.g. our nano-vllm server)
answers with an option letter. Reports accuracy with a 95% normal-approximation CI.

Usage:
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
    """Flatten MIRAGE into ``{"dataset", "qid", "question", "options", "answer"}`` records."""
    bench = json.loads(Path(path).read_text())
    rng = random.Random(seed)
    out = []
    for name in datasets:
        items = [{"dataset": name, "qid": qid, **q} for qid, q in bench[name].items()]
        if max_per_dataset and len(items) > max_per_dataset:
            items = rng.sample(items, max_per_dataset)
        out.extend(items)
    return out


def build_prompt(q: dict, hits: list[Hit] | None, max_chars: int = 1500) -> str:
    """Render documents (if any), question and lettered options."""
    parts = []
    if hits:
        docs = "\n".join(f"[{i}] ({h.title}) {h.content[:max_chars]}" for i, h in enumerate(hits, 1))
        parts.append(f"Documents:\n{docs}\n")
    opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    parts.append(f"Question: {q['question']}\nOptions:\n{opts}\n\nAnswer with a single letter.")
    return "\n".join(parts)


def parse_letter(text: str, valid: set[str]) -> str | None:
    """First standalone option letter in the reply, if it is a valid option."""
    for m in LETTER.finditer(text.strip()):
        if m.group(1) in valid:
            return m.group(1)
    return None


async def ask(session: aiohttp.ClientSession, url: str, model: str, prompt: str,
              sem: asyncio.Semaphore) -> str:
    """One greedy chat completion with Qwen3 thinking disabled."""
    payload = {"model": model, "temperature": 0, "max_tokens": 8,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt}],
               "chat_template_kwargs": {"enable_thinking": False}}
    async with sem, session.post(url, json=payload) as resp:
        body = await resp.json()
    if "choices" not in body:
        return f"<error {body.get('error')}>"
    return body["choices"][0]["message"]["content"]


async def answer_all(base_url: str, model: str, prompts: list[str], concurrency: int) -> list[str]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as session:
        return await asyncio.gather(*(ask(session, url, model, p, sem) for p in prompts))


def accuracy(correct: list[bool]) -> dict[str, float]:
    """Accuracy with a 95% normal-approximation confidence half-width."""
    n = len(correct)
    acc = sum(correct) / n if n else float("nan")
    return {"n": n, "acc": acc, "ci95": 1.96 * math.sqrt(acc * (1 - acc) / n) if n else float("nan")}


def main() -> None:
    """Retrieve, query the model, score and save per-mode / per-dataset accuracy."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
    print(f"{len(questions)} questions from {args.datasets}")
    retriever = None
    if any(m != "none" for m in args.modes):
        retriever = HybridRetriever.from_index(args.corpus, args.index, args.query_model,
                                               args.reranker_model)
    record = {"config": vars(args), "results": {}, "items": {}}
    for mode in args.modes:
        t = time.perf_counter()
        hits = ([None] * len(questions) if mode == "none" else
                retriever.search([q["question"] for q in questions], k=args.k, mode=mode))
        t_ret = time.perf_counter() - t
        prompts = [build_prompt(q, h) for q, h in zip(questions, hits)]
        replies = asyncio.run(answer_all(args.base_url, args.model, prompts, args.concurrency))
        t_all = time.perf_counter() - t
        preds = [parse_letter(r, set(q["options"])) for r, q in zip(replies, questions)]
        per_ds: dict[str, list[bool]] = {}
        for q, pred in zip(questions, preds):
            per_ds.setdefault(q["dataset"], []).append(pred == q["answer"])
        res = {ds: accuracy(c) for ds, c in per_ds.items()}
        res["avg"] = {"acc": sum(r["acc"] for r in res.values()) / len(res)}
        res["unparsed"] = sum(pr is None for pr in preds)
        res["retrieval_s"], res["total_s"] = round(t_ret, 1), round(t_all, 1)
        record["results"][mode] = res
        record["items"][mode] = [{"dataset": q["dataset"], "qid": q["qid"], "pred": pr,
                                  "answer": q["answer"], "docs": [h.id for h in hs] if hs else []}
                                 for q, pr, hs in zip(questions, preds, hits)]
        print(f"{mode:14} " + "  ".join(f"{ds} {r['acc']:.3f}" for ds, r in res.items()
                                       if isinstance(r, dict) and "acc" in r)
              + f"  | unparsed {res['unparsed']}  retrieval {t_ret:.0f}s  total {t_all:.0f}s")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps(record, indent=1))
    print(f"saved -> {out / f'{args.label}.json'}")


if __name__ == "__main__":
    main()
