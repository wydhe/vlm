"""有据作答 + 闭卷兜底：把 GRPO 对齐后的模型接回推理服务，放进检索流程里评测。

每道题的流程：
    1. 检索：从语料（教材或 PubMed）取 top-k 文档；对 PubMedQA 另外构造一组"原始摘要 + 检索到的
       其他文档"，模拟"一定检索到了正确文献"的情况（检索结果里的原文按 PMID 去掉，避免重复）。
    2. 有据作答：按 GRPO 训练时的格式，让模型引用某篇文档作答，或回答 insufficient（资料不足）。
    3. 闭卷兜底：如果模型认为资料不足，就不给资料，让它直接作答。

比较三种策略：
    closed    闭卷作答，不看任何资料
    rag       只根据资料作答，回答 insufficient 记为答错
    rag+fb    根据资料作答，资料不足时改为闭卷作答

用法（先用 openai_server 启动要评测的模型）：
    uv run python -m agent.grounded_qa --model Qwen3-1.7B-GRPO --label grpo
    uv run python -m agent.grounded_qa --model Qwen3-1.7B-GRPO --corpus pubmed --label grpo-pubmed

同一语料、k、seed 的检索结果缓存在 results/grounded/docs-<语料>-k<k>-seed<seed>.json，换模型时直接复用。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

import aiohttp
import pyarrow.parquet as pq

from align.task import SYSTEM as GROUNDED_SYSTEM
from align.task import parse, render_user
from rag.search import CORPORA, load_search

CLOSED_SYSTEM = (
    "You answer biomedical yes/no questions.\n"
    "Reply in exactly this format:\n"
    "Reasoning: <one or two sentences>\n"
    "Answer: yes | no | maybe"
)
CLOSED_ANSWER = re.compile(r"Answer:\s*(yes|no|maybe)", re.IGNORECASE)


# ----------------------------------------------------------------------------- 数据

def load_questions(mirage_path: str, pubmedqa_dir: str) -> list[dict]:
    """读取 MIRAGE 的 PubMedQA 和 BioASQ，答案转成 yes/no/maybe 文本。

    PubMedQA 的题目额外带上原始摘要（gold），用于"检索到正确文献"的对照条件。
    """
    bench = json.loads(Path(mirage_path).read_text())
    table = pq.read_table(next(Path(pubmedqa_dir, "pqa_labeled").glob("*.parquet")))
    abstracts = {str(r["pubid"]): " ".join(r["context"]["contexts"]) for r in table.to_pylist()}

    questions = []
    for dataset in ("pubmedqa", "bioasq"):
        for qid, q in bench[dataset].items():
            item = {
                "dataset": dataset,
                "qid": qid,
                "question": q["question"],
                "label": q["options"][q["answer"]],
                "pmids": [str(pmid) for pmid in q["PMID"]],
            }
            if dataset == "pubmedqa":
                item["gold_abstract"] = abstracts[str(q["PMID"][0])]
            questions.append(item)
    return questions


def attach_documents(questions: list[dict], corpus: str, k: int, seed: int, cache_dir: Path,
                     models: tuple[str, str, str]) -> None:
    """为每道题准备两组文档：检索结果（docs_retrieved），和"原始摘要 + 检索结果"（docs_gold，仅 PubMedQA）。"""
    cache_file = cache_dir / f"docs-{corpus}-k{k}-seed{seed}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
    else:
        search = load_search(corpus, *models)
        # 多取一条：PubMedQA 的原文可能在检索结果里，去掉后仍有 k-1 篇干扰文档
        hits = search([q["question"] for q in questions], k + 1)
        rng = random.Random(seed)
        cached = {}
        for q, hit_list in zip(questions, hits):
            entry = {"retrieved": [h["content"] for h in hit_list[:k]]}
            if "gold_abstract" in q:
                others = [h["content"] for h in hit_list if str(h["id"]) not in q["pmids"]]
                docs = [q["gold_abstract"]] + others[:k - 1]
                rng.shuffle(docs)
                entry["gold"] = docs
            cached[f"{q['dataset']}/{q['qid']}"] = entry
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cached))
    for q in questions:
        entry = cached[f"{q['dataset']}/{q['qid']}"]
        q["docs_retrieved"] = entry["retrieved"]
        if "gold" in entry:
            q["docs_gold"] = entry["gold"]
            q["gold_index"] = entry["gold"].index(q["gold_abstract"])


# ----------------------------------------------------------------------------- 调用模型

async def chat(session: aiohttp.ClientSession, url: str, model: str,
               system: str, user: str, sem: asyncio.Semaphore) -> str:
    """一次贪心对话补全（关闭 Qwen3 的 thinking）。"""
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 256,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with sem, session.post(url, json=payload) as resp:
        body = await resp.json()
    return body["choices"][0]["message"]["content"]


async def answer_closed(session, url, model, q, sem) -> str | None:
    """闭卷作答，返回 yes/no/maybe，解析失败返回 None。"""
    reply = await chat(session, url, model, CLOSED_SYSTEM, f"Question: {q['question']}", sem)
    match = CLOSED_ANSWER.search(reply)
    return match.group(1).lower() if match else None


async def answer_grounded(session, url, model, q, docs, sem) -> dict:
    """按 GRPO 训练格式、根据文档作答；返回答案和引用的文档编号（1 开始）。"""
    user = render_user({"question": q["question"], "docs": docs})
    reply = await chat(session, url, model, GROUNDED_SYSTEM, user, sem)
    parsed = parse(reply)
    return {"answer": parsed.answer, "cites": sorted(parsed.cites)}


async def evaluate_one(session, url, model, q, sem) -> dict:
    """对一道题跑完所有条件，返回逐题记录。"""
    record = {"dataset": q["dataset"], "qid": q["qid"], "label": q["label"]}
    record["closed"] = await answer_closed(session, url, model, q, sem)
    record["retrieved"] = await answer_grounded(session, url, model, q, q["docs_retrieved"], sem)
    if "docs_gold" in q:
        record["gold"] = await answer_grounded(session, url, model, q, q["docs_gold"], sem)
        record["gold_index"] = q["gold_index"]
    return record


async def evaluate_all(base_url: str, model: str, questions: list[dict], concurrency: int) -> list[dict]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [evaluate_one(session, url, model, q, sem) for q in questions]
        return await asyncio.gather(*tasks)


# ----------------------------------------------------------------------------- 统计

def strategy_answers(record: dict, docs_key: str) -> dict[str, str | None]:
    """把一道题的原始结果换算成三种策略各自的最终答案。"""
    grounded = record[docs_key]["answer"]
    fallback = record["closed"] if grounded in (None, "insufficient") else grounded
    return {"closed": record["closed"], "rag": grounded, "rag+fb": fallback}


def summarize(records: list[dict]) -> dict:
    """按 数据集 × 文档条件 汇总三种策略的准确率，以及拒答率、引用准确率。"""
    summary = {}
    for dataset in ("pubmedqa", "bioasq"):
        rows = [r for r in records if r["dataset"] == dataset]
        doc_conditions = ["retrieved", "gold"] if dataset == "pubmedqa" else ["retrieved"]
        for docs_key in doc_conditions:
            name = f"{dataset}/{docs_key}"
            correct = {"closed": 0, "rag": 0, "rag+fb": 0}
            refused = 0
            for r in rows:
                answers = strategy_answers(r, docs_key)
                for strategy, answer in answers.items():
                    correct[strategy] += answer == r["label"]
                refused += r[docs_key]["answer"] == "insufficient"
            n = len(rows)
            summary[name] = {"n": n, "refusal_rate": refused / n}
            summary[name].update({s: c / n for s, c in correct.items()})
            if docs_key == "gold":
                cited_gold = sum(r["gold"]["cites"] == [r["gold_index"] + 1] for r in rows)
                summary[name]["citation_acc"] = cited_gold / n
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="有据作答 + 闭卷兜底评测")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True, help="服务端的模型名")
    p.add_argument("--mirage", default="data/mirage/benchmark.json")
    p.add_argument("--pubmedqa", default="data/pubmedqa")
    p.add_argument("--corpus", choices=CORPORA, default="textbooks", help="检索哪个语料")
    p.add_argument("--query-model", default="weights/MedCPT-Query-Encoder")
    p.add_argument("--article-model", default="weights/MedCPT-Article-Encoder")
    p.add_argument("--reranker-model", default="weights/MedCPT-Cross-Encoder")
    p.add_argument("--k", type=int, default=4, help="每题给模型的文档数")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="grounded")
    p.add_argument("--out-dir", default="results/grounded")
    args = p.parse_args()

    questions = load_questions(args.mirage, args.pubmedqa)
    attach_documents(questions, args.corpus, args.k, args.seed, Path(args.out_dir),
                     (args.query_model, args.article_model, args.reranker_model))
    print(f"{len(questions)} 道题，开始评测模型 {args.model}")

    records = asyncio.run(evaluate_all(args.base_url, args.model, questions, args.concurrency))
    summary = summarize(records)
    for name, row in summary.items():
        print(f"{name:18} closed {row['closed']:.3f}  rag {row['rag']:.3f}  rag+fb {row['rag+fb']:.3f}"
              f"  拒答率 {row['refusal_rate']:.3f}"
              + (f"  引用准确率 {row['citation_acc']:.3f}" if "citation_acc" in row else ""))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.label}.json"
    out_file.write_text(json.dumps({"config": vars(args), "summary": summary, "records": records}, indent=1))
    print(f"结果已保存到 {out_file}")


if __name__ == "__main__":
    main()
