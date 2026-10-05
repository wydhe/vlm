"""Agentic RAG: the model decides when and what to search before answering.

Loop (Qwen3 native function calling through our OpenAI-compatible server)::

    system + tools + question ──► model ──► <tool_call>{"name": "search_medical_kb", ...}</tool_call>
                ▲                                  │
                └──── tool message (top-3 chunks) ◄┘    ... until "Answer: X" or the call budget

Every turn re-sends the whole conversation, so turn n shares its prefix with
turn n-1: the engine's prefix cache turns that into skipped prefill, which we
measure through the server's ``/metrics`` counters.

Usage:
    uv run python -m agent.medical_agent --per-dataset 200 --label agent-qwen3-8b \
        --compare results/rag/qwen3-8b-textbooks-k5.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import threading
import time
from pathlib import Path

import aiohttp

from rag.eval_mirage import accuracy, load_mirage, parse_letter
from rag.retrievers import HybridRetriever

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_medical_kb",
        "description": "Search medical textbooks (anatomy, physiology, pathology, pharmacology, "
                       "internal medicine, ...) and return the most relevant passages.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "A focused search query about one fact."}},
            "required": ["query"],
        },
    },
}]
SYSTEM = ("You are a medical expert answering a multiple-choice question. You may call "
          "search_medical_kb (at most {max_calls} times) to look up facts you are unsure about. "
          "When you are confident, reply with 'Answer: <letter>' and nothing else.")
TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
FINAL = re.compile(r"Answer:\s*\(?([A-E])\b")


class LockedRetriever:
    """Serialize GPU retrieval calls coming from many concurrent agent coroutines."""

    def __init__(self, retriever: HybridRetriever, k: int, max_chars: int):
        self.retriever, self.k, self.max_chars = retriever, k, max_chars
        self.lock = threading.Lock()

    def __call__(self, query: str) -> str:
        with self.lock:
            hits = self.retriever.search([query], k=self.k, mode="hybrid_rerank")[0]
        return "\n".join(f"[{i}] ({h.title}) {h.content[:self.max_chars]}"
                         for i, h in enumerate(hits, 1))


async def chat(session: aiohttp.ClientSession, url: str, model: str, messages: list[dict],
               max_tokens: int) -> str:
    payload = {"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": messages,
               "tools": TOOLS, "chat_template_kwargs": {"enable_thinking": False}}
    async with session.post(url, json=payload) as resp:
        body = await resp.json()
    if "choices" not in body:
        raise RuntimeError(str(body.get("error")))
    return body["choices"][0]["message"]["content"]


async def solve(session: aiohttp.ClientSession, url: str, model: str, q: dict,
                search: LockedRetriever, max_calls: int, max_tokens: int) -> dict:
    """Run the tool loop for one question; returns prediction and trace statistics."""
    opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    messages = [{"role": "system", "content": SYSTEM.format(max_calls=max_calls)},
                {"role": "user", "content": f"Question: {q['question']}\nOptions:\n{opts}"}]
    queries: list[str] = []
    start = time.perf_counter()
    reply = ""
    for _ in range(max_calls + 2):
        reply = await chat(session, url, model, messages, max_tokens)
        messages.append({"role": "assistant", "content": reply})
        calls = TOOL_CALL.findall(reply)
        if not calls:
            break
        results = []
        for raw in calls:
            try:
                query = json.loads(raw)["arguments"]["query"]
            except (json.JSONDecodeError, KeyError, TypeError):
                results.append("Invalid tool call; use {\"query\": \"...\"}.")
                continue
            if len(queries) >= max_calls:
                results.append("Search limit reached. Answer now with 'Answer: <letter>'.")
                continue
            queries.append(query)
            results.append(await asyncio.to_thread(search, query))
        messages.append({"role": "tool", "content": "\n\n".join(results)})
    m = FINAL.search(reply)
    valid = set(q["options"])
    pred = m.group(1) if m and m.group(1) in valid else parse_letter(reply, valid)
    return {"dataset": q["dataset"], "qid": q["qid"], "pred": pred, "answer": q["answer"],
            "queries": queries, "turns": sum(msg["role"] == "assistant" for msg in messages),
            "latency_s": time.perf_counter() - start}


async def run_all(base_url: str, model: str, questions: list[dict], search: LockedRetriever,
                  max_calls: int, max_tokens: int, concurrency: int) -> list[dict]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6 * 3600)) as session:
        async def one(q: dict) -> dict:
            async with sem:
                try:
                    return await solve(session, url, model, q, search, max_calls, max_tokens)
                except (RuntimeError, aiohttp.ClientError) as e:
                    return {"dataset": q["dataset"], "qid": q["qid"], "pred": None,
                            "answer": q["answer"], "queries": [], "turns": 0,
                            "latency_s": 0.0, "error": str(e)}
        return await asyncio.gather(*(one(q) for q in questions))


async def get_metrics(base_url: str) -> dict:
    async with aiohttp.ClientSession() as s, s.get(base_url.rstrip("/") + "/metrics") as r:
        return await r.json()


def main() -> None:
    """Run the agent on a MIRAGE subset and compare with single-shot RAG on the same questions."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen3-8B")
    p.add_argument("--mirage", default="data/mirage/benchmark.json")
    p.add_argument("--datasets", nargs="+", default=["medqa", "medmcqa", "pubmedqa", "bioasq", "mmlu"])
    p.add_argument("--per-dataset", type=int, default=200,
                   help="first N of the seed-0 sample used by rag.eval_mirage --max-per-dataset 1000")
    p.add_argument("--max-calls", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--max-chars", type=int, default=1200)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--corpus", default="data/medrag/textbooks/chunk")
    p.add_argument("--index", default="data/indexes/textbooks")
    p.add_argument("--query-model", default="weights/MedCPT-Query-Encoder")
    p.add_argument("--reranker-model", default="weights/MedCPT-Cross-Encoder")
    p.add_argument("--compare", default=None, help="rag.eval_mirage result JSON to compare against")
    p.add_argument("--label", default="agent")
    p.add_argument("--out-dir", default="results/agent")
    args = p.parse_args()

    pool = load_mirage(args.mirage, args.datasets, 1000)
    questions = [q for ds in args.datasets for q in [x for x in pool if x["dataset"] == ds][:args.per_dataset]]
    print(f"{len(questions)} questions")
    search = LockedRetriever(HybridRetriever.from_index(args.corpus, args.index, args.query_model,
                                                        args.reranker_model), args.k, args.max_chars)
    before = asyncio.run(get_metrics(args.base_url))
    t = time.perf_counter()
    items = asyncio.run(run_all(args.base_url, args.model, questions, search, args.max_calls,
                                args.max_tokens, args.concurrency))
    wall = time.perf_counter() - t
    after = asyncio.run(get_metrics(args.base_url))

    per_ds: dict[str, list[bool]] = {}
    for it in items:
        per_ds.setdefault(it["dataset"], []).append(it["pred"] == it["answer"])
    summary = {ds: accuracy(c) for ds, c in per_ds.items()}
    summary["avg"] = {"acc": sum(r["acc"] for r in summary.values()) / len(summary)}
    prompt_tok = after["prompt_tokens"] - before["prompt_tokens"]
    cached_tok = after["cached_prompt_tokens"] - before["cached_prompt_tokens"]
    stats = {
        "wall_s": wall,
        "avg_tool_calls": statistics.mean(len(it["queries"]) for it in items),
        "no_search_ratio": sum(not it["queries"] for it in items) / len(items),
        "avg_turns": statistics.mean(it["turns"] for it in items),
        "latency_median_s": statistics.median(it["latency_s"] for it in items),
        "unparsed": sum(it["pred"] is None for it in items),
        "errors": sum("error" in it for it in items),
        "prompt_tokens": prompt_tok, "cached_prompt_tokens": cached_tok,
        "prefix_cache_hit_rate": cached_tok / prompt_tok if prompt_tok else 0.0,
    }
    print("agent         " + "  ".join(f"{ds} {r['acc']:.3f}" for ds, r in summary.items()))
    print(json.dumps(stats, indent=1))

    comparison = None
    if args.compare:
        rag = json.loads(Path(args.compare).read_text())["items"]
        keys = {(it["dataset"], it["qid"]) for it in items}
        comparison = {}
        for mode, rows in rag.items():
            sub = [r for r in rows if (r["dataset"], r["qid"]) in keys]
            by_ds: dict[str, list[bool]] = {}
            for r in sub:
                by_ds.setdefault(r["dataset"], []).append(r["pred"] == r["answer"])
            comparison[mode] = {ds: accuracy(c)["acc"] for ds, c in by_ds.items()}
            comparison[mode]["avg"] = sum(comparison[mode].values()) / len(comparison[mode])
            print(f"{mode:14}" + "  ".join(f"{ds} {a:.3f}" for ds, a in comparison[mode].items()))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps(
        {"config": vars(args), "summary": summary, "stats": stats, "same_questions_rag": comparison,
         "items": items}, indent=1))
    print(f"saved -> {out / f'{args.label}.json'}")


if __name__ == "__main__":
    main()
