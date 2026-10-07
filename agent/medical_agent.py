"""Agentic RAG：由模型自己决定要不要检索、检索什么，然后再作答。

循环过程（通过本项目 OpenAI 兼容服务，使用 Qwen3 原生的函数调用格式）：

    system + 工具定义 + 题目 ──► 模型 ──► <tool_call>{"name": "search_medical_kb", ...}</tool_call>
                ▲                               │
                └──── 工具结果（top-3 片段） ◄────┘    …… 直到模型回答 "Answer: X" 或用完检索次数

每一轮都会把整段对话重新发给服务端，第 n 轮和第 n-1 轮有相同的前缀，
引擎的前缀缓存可以跳过这部分的 prefill。命中率通过服务端的 /metrics 统计。

用法：
    uv run python -m agent.medical_agent --per-dataset 200 --label agent-qwen3-8b \
        --compare results/rag/qwen3-8b-textbooks-k5.json
    uv run python -m agent.medical_agent --per-dataset 200 --corpus pubmed --label agent-qwen3-8b-pubmed
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
from rag.search import CORPORA, SearchFn, load_search

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_medical_kb",
        "description": "",   # 按语料填写，见 TOOL_DESCRIPTIONS
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "A focused search query about one fact."}},
            "required": ["query"],
        },
    },
}]
TOOL_DESCRIPTIONS = {
    "textbooks": "Search medical textbooks (anatomy, physiology, pathology, pharmacology, "
                 "internal medicine, ...) and return the most relevant passages.",
    "pubmed": "Search PubMed abstracts of biomedical research articles and return the most "
              "relevant ones.",
}
SYSTEM = ("You are a medical expert answering a multiple-choice question. You may call "
          "search_medical_kb (at most {max_calls} times) to look up facts you are unsure about. "
          "When you are confident, reply with 'Answer: <letter>' and nothing else.")
TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
FINAL = re.compile(r"Answer:\s*\(?([A-E])\b")


class LockedRetriever:
    """很多题同时在跑，检索模型在 GPU 上，用一把锁让检索一次只做一个。"""

    def __init__(self, search: SearchFn, k: int, max_chars: int):
        self.search = search
        self.k = k
        self.max_chars = max_chars
        self.lock = threading.Lock()

    def __call__(self, query: str) -> str:
        with self.lock:
            hits = self.search([query], self.k)[0]
        return "\n".join(f"[{i}] ({h['title']}) {h['content'][:self.max_chars]}"
                         for i, h in enumerate(hits, 1))


async def chat(session: aiohttp.ClientSession, url: str, model: str, messages: list[dict],
               max_tokens: int) -> str:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": messages,
        "tools": TOOLS,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with session.post(url, json=payload) as resp:
        body = await resp.json()
    if "choices" not in body:
        raise RuntimeError(str(body.get("error")))
    return body["choices"][0]["message"]["content"]


async def run_tool_calls(calls: list[str], queries: list[str], search: LockedRetriever,
                         max_calls: int) -> str:
    """执行模型这一轮发出的所有工具调用，把结果拼成一条 tool 消息。

    用完检索次数后不删除工具定义（那样会改变 system prompt，前缀缓存就失效了），
    而是直接告诉模型"已达上限，请作答"。
    """
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
    return "\n\n".join(results)


async def solve(session: aiohttp.ClientSession, url: str, model: str, q: dict,
                search: LockedRetriever, max_calls: int, max_tokens: int) -> dict:
    """对一道题跑完整个工具调用循环，返回预测和过程统计。"""
    options = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    messages = [{"role": "system", "content": SYSTEM.format(max_calls=max_calls)},
                {"role": "user", "content": f"Question: {q['question']}\nOptions:\n{options}"}]
    queries: list[str] = []
    start = time.perf_counter()
    reply = ""
    for _ in range(max_calls + 2):
        reply = await chat(session, url, model, messages, max_tokens)
        messages.append({"role": "assistant", "content": reply})
        calls = TOOL_CALL.findall(reply)
        if not calls:
            break
        tool_result = await run_tool_calls(calls, queries, search, max_calls)
        messages.append({"role": "tool", "content": tool_result})

    valid = set(q["options"])
    final = FINAL.search(reply)
    pred = final.group(1) if final and final.group(1) in valid else parse_letter(reply, valid)
    turns = sum(m["role"] == "assistant" for m in messages)
    return {"dataset": q["dataset"], "qid": q["qid"], "pred": pred, "answer": q["answer"],
            "queries": queries, "turns": turns, "latency_s": time.perf_counter() - start}


async def run_all(base_url: str, model: str, questions: list[dict], search: LockedRetriever,
                  max_calls: int, max_tokens: int, concurrency: int) -> list[dict]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    sem = asyncio.Semaphore(concurrency)

    async def one(session, q: dict) -> dict:
        async with sem:
            try:
                return await solve(session, url, model, q, search, max_calls, max_tokens)
            except (RuntimeError, aiohttp.ClientError) as e:
                return {"dataset": q["dataset"], "qid": q["qid"], "pred": None, "answer": q["answer"],
                        "queries": [], "turns": 0, "latency_s": 0.0, "error": str(e)}

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6 * 3600)) as session:
        return await asyncio.gather(*(one(session, q) for q in questions))


async def get_metrics(base_url: str) -> dict:
    async with aiohttp.ClientSession() as session, session.get(base_url.rstrip("/") + "/metrics") as r:
        return await r.json()


# ---------------------------------------------------------------------- 统计

def select_questions(mirage: str, datasets: list[str], per_dataset: int) -> list[dict]:
    """取 rag.eval_mirage（--max-per-dataset 1000）同一抽样里每个数据集的前 N 题，方便对比。"""
    pool = load_mirage(mirage, datasets, 1000)
    questions = []
    for name in datasets:
        questions.extend([q for q in pool if q["dataset"] == name][:per_dataset])
    return questions


def accuracy_by_dataset(rows: list[dict]) -> dict:
    by_dataset: dict[str, list[bool]] = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], []).append(r["pred"] == r["answer"])
    return {name: accuracy(correct) for name, correct in by_dataset.items()}


def run_stats(items: list[dict], wall: float, before: dict, after: dict) -> dict:
    """Agent 的过程统计：检索次数、轮数、延迟，以及这段时间里的前缀缓存命中率。"""
    prompt_tokens = after["prompt_tokens"] - before["prompt_tokens"]
    cached_tokens = after["cached_prompt_tokens"] - before["cached_prompt_tokens"]
    return {
        "wall_s": wall,
        "avg_tool_calls": statistics.mean(len(it["queries"]) for it in items),
        "no_search_ratio": sum(not it["queries"] for it in items) / len(items),
        "avg_turns": statistics.mean(it["turns"] for it in items),
        "latency_median_s": statistics.median(it["latency_s"] for it in items),
        "unparsed": sum(it["pred"] is None for it in items),
        "errors": sum("error" in it for it in items),
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "prefix_cache_hit_rate": cached_tokens / prompt_tokens if prompt_tokens else 0.0,
    }


def compare_with_rag(items: list[dict], rag_result_path: str) -> dict:
    """在完全相同的题目上，取出 rag.eval_mirage 各模式的准确率。"""
    rag_items = json.loads(Path(rag_result_path).read_text())["items"]
    asked = {(it["dataset"], it["qid"]) for it in items}
    comparison = {}
    for mode, rows in rag_items.items():
        same = [r for r in rows if (r["dataset"], r["qid"]) in asked]
        scores = {name: r["acc"] for name, r in accuracy_by_dataset(same).items()}
        scores["avg"] = sum(scores.values()) / len(scores)
        comparison[mode] = scores
    return comparison


def main() -> None:
    p = argparse.ArgumentParser(description="Agentic RAG 评测")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen3-8B")
    p.add_argument("--mirage", default="data/mirage/benchmark.json")
    p.add_argument("--datasets", nargs="+", default=["medqa", "medmcqa", "pubmedqa", "bioasq", "mmlu"])
    p.add_argument("--per-dataset", type=int, default=200)
    p.add_argument("--max-calls", type=int, default=3, help="每题最多检索几次")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--k", type=int, default=3, help="每次检索返回几段")
    p.add_argument("--max-chars", type=int, default=1200)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--corpus", choices=CORPORA, default="textbooks", help="检索哪个语料")
    p.add_argument("--query-model", default="weights/MedCPT-Query-Encoder")
    p.add_argument("--article-model", default="weights/MedCPT-Article-Encoder")
    p.add_argument("--reranker-model", default="weights/MedCPT-Cross-Encoder")
    p.add_argument("--compare", default=None, help="rag.eval_mirage 的结果文件，用于同题对比")
    p.add_argument("--label", default="agent")
    p.add_argument("--out-dir", default="results/agent")
    args = p.parse_args()

    questions = select_questions(args.mirage, args.datasets, args.per_dataset)
    print(f"{len(questions)} 道题")
    TOOLS[0]["function"]["description"] = TOOL_DESCRIPTIONS[args.corpus]
    search = LockedRetriever(load_search(args.corpus, args.query_model, args.article_model, args.reranker_model),
                             args.k, args.max_chars)

    before = asyncio.run(get_metrics(args.base_url))
    start = time.perf_counter()
    items = asyncio.run(run_all(args.base_url, args.model, questions, search, args.max_calls,
                                args.max_tokens, args.concurrency))
    wall = time.perf_counter() - start
    after = asyncio.run(get_metrics(args.base_url))

    summary = accuracy_by_dataset(items)
    summary["avg"] = {"acc": sum(r["acc"] for r in summary.values()) / len(summary)}
    stats = run_stats(items, wall, before, after)
    print("agent         " + "  ".join(f"{name} {r['acc']:.3f}" for name, r in summary.items()))
    print(json.dumps(stats, indent=1))

    comparison = compare_with_rag(items, args.compare) if args.compare else None
    for mode, scores in (comparison or {}).items():
        print(f"{mode:14}" + "  ".join(f"{name} {acc:.3f}" for name, acc in scores.items()))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.label}.json"
    out_file.write_text(json.dumps({"config": vars(args), "summary": summary, "stats": stats,
                                    "same_questions_rag": comparison, "items": items}, indent=1))
    print(f"结果已保存到 {out_file}")


if __name__ == "__main__":
    main()
