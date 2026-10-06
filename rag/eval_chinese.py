"""中文医学考试题评测：MedQA-MCMLE（执业医师考试）+ CMB 验证集，比较不同检索方式。

模式：
    none            不检索，直接作答
    bm25            jieba 分词 BM25
    dense           bge-m3 向量检索
    hybrid_rerank   BM25 + 向量 → RRF 融合 → bge-reranker 重排

多选题必须选出全部正确选项才算对。生成模型由本项目的推理服务提供（例如 Qwen3-8B）。

用法：
    uv run python -m rag.eval_chinese --model Qwen3-8B --label qwen3-8b-zh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import aiohttp
import pyarrow.parquet as pq

from rag.chinese import ChineseRetriever
from rag.eval_mirage import accuracy

SYSTEM = "你是一名医学专家。如果提供的资料与题目相关，请参考资料；请只输出选项字母，不要解释。"
LETTERS = re.compile(r"[A-E]")


def load_questions(mcmle_path: str, cmb_path: str) -> list[dict]:
    """统一成 {dataset, question, options{字母: 内容}, answer(字母串), multi(是否多选)}。"""
    questions = []
    for row in pq.read_table(mcmle_path).to_pylist():
        questions.append({
            "dataset": "mcmle",
            "question": row["question"],
            "options": {o["key"]: o["value"] for o in row["options"]},
            "answer": row["answer_idx"],
            "multi": False,
        })
    for row in json.loads(Path(cmb_path).read_text()):
        questions.append({
            "dataset": "cmb",
            "question": row["question"],
            "options": {k: v for k, v in row["option"].items() if v},
            "answer": row["answer"],
            "multi": row["question_type"] == "多项选择题",
        })
    return questions


def build_prompt(q: dict, passages: list[str] | None) -> str:
    parts = []
    if passages:
        docs = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
        parts.append(f"参考资料：\n{docs}\n")
    options = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    kind = "多项选择题，请输出所有正确选项的字母（例如 ACD）" if q["multi"] else "单项选择题，请只输出一个字母"
    parts.append(f"题目（{kind}）：{q['question']}\n选项：\n{options}\n答案：")
    return "\n".join(parts)


def parse_answer(reply: str, q: dict) -> str | None:
    """单选取第一个合法字母；多选取所有合法字母（排序去重）。"""
    letters = [c for c in LETTERS.findall(reply) if c in q["options"]]
    if not letters:
        return None
    return "".join(sorted(set(letters))) if q["multi"] else letters[0]


async def ask(session, url: str, model: str, prompt: str, sem: asyncio.Semaphore) -> str:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 16,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with sem, session.post(url, json=payload) as resp:
        body = await resp.json()
    return body["choices"][0]["message"]["content"] if "choices" in body else ""


async def ask_all(base_url: str, model: str, prompts: list[str], concurrency: int) -> list[str]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as session:
        return await asyncio.gather(*(ask(session, url, model, p, sem) for p in prompts))


def main() -> None:
    p = argparse.ArgumentParser(description="中文医学考试题 RAG 评测")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen3-8B")
    p.add_argument("--mcmle", default="data/medqa_zh/med_qa_zh_4options_source/test-00000-of-00001.parquet")
    p.add_argument("--cmb", default="data/cmb/CMB-Exam/CMB-val/CMB-val-merge.json")
    p.add_argument("--modes", nargs="+", default=["none", "bm25", "dense", "hybrid_rerank"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--index", default="data/indexes/huatuo")
    p.add_argument("--encoder", default="weights/bge-m3")
    p.add_argument("--reranker", default="weights/bge-reranker-v2-m3")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--label", default="zh")
    p.add_argument("--out-dir", default="results/rag")
    args = p.parse_args()

    questions = load_questions(args.mcmle, args.cmb)
    retriever = ChineseRetriever(args.index, args.encoder, args.reranker)
    print(f"{len(questions)} 道题")

    record = {"config": vars(args), "results": {}, "items": {}}
    for mode in args.modes:
        start = time.perf_counter()
        if mode == "none":
            passages = [None] * len(questions)
        else:
            passages = retriever.search([q["question"] for q in questions], k=args.k, mode=mode)
        prompts = [build_prompt(q, ps) for q, ps in zip(questions, passages)]
        replies = asyncio.run(ask_all(args.base_url, args.model, prompts, args.concurrency))
        predictions = [parse_answer(r, q) for r, q in zip(replies, questions)]

        by_dataset: dict[str, list[bool]] = {}
        for q, pred in zip(questions, predictions):
            by_dataset.setdefault(q["dataset"], []).append(pred == q["answer"])
        result = {name: accuracy(c) for name, c in by_dataset.items()}
        result["unparsed"] = sum(pred is None for pred in predictions)
        result["seconds"] = round(time.perf_counter() - start, 1)
        record["results"][mode] = result
        record["items"][mode] = [{"dataset": q["dataset"], "pred": pred, "answer": q["answer"]}
                                 for q, pred in zip(questions, predictions)]
        print(f"{mode:14} " + "  ".join(f"{name} {r['acc']:.3f}" for name, r in result.items()
                                       if isinstance(r, dict))
              + f"  | 未解析 {result['unparsed']}  用时 {result['seconds']:.0f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.label}.json"
    out_file.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    print(f"结果已保存到 {out_file}")


if __name__ == "__main__":
    main()
