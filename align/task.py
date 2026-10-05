"""Grounded-answer task for GRPO: cite the supporting abstract, or refuse when none supports the answer.

Each sample shows ``n_docs`` PubMed abstracts: the question's own abstract ("gold")
plus BM25-nearest distractors, shuffled. With probability ``p_unanswerable`` the
gold abstract is swapped for another distractor and the correct reply becomes
``insufficient``. Everything is automatically verifiable, so no reward model or
LLM judge is needed.

Train samples come from ``pqa_artificial``; eval samples from ``pqa_labeled``
(disjoint from training, and the source of MIRAGE's PubMedQA split).

Sample (one JSON line)::

    {"question": str, "docs": [str, ...], "gold": int (0-based, -1 if unanswerable),
     "label": "yes" | "no" | "maybe" | "insufficient"}
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import bm25s
import pyarrow.parquet as pq
import Stemmer

SYSTEM = (
    "You answer biomedical yes/no questions using ONLY the given documents.\n"
    "Reply in exactly this format:\n"
    "Reasoning: <one or two sentences>\n"
    "Citation: [n]\n"
    "Answer: yes | no | maybe\n"
    "If none of the documents addresses the question, reply:\n"
    "Reasoning: <why>\n"
    "Citation: none\n"
    "Answer: insufficient"
)
FORMAT = re.compile(r"Reasoning:.+\nCitation:\s*(none|\[\d+\](?:\s*,?\s*\[\d+\])*)\s*\n"
                    r"Answer:\s*(yes|no|maybe|insufficient)\s*$", re.DOTALL | re.IGNORECASE)
CITES = re.compile(r"\[(\d+)\]")


# --------------------------------------------------------------------------- data

def _abstract(row: dict) -> str:
    return " ".join(row["context"]["contexts"])


def build_samples(rows: list[dict], n: int, n_docs: int, p_unanswerable: float,
                  rng: random.Random, balance: bool = False) -> list[dict]:
    """Assemble ``n`` samples from PubMedQA rows with BM25-nearest distractor abstracts.

    With ``balance`` the gold questions are drawn equally from each label, so a
    policy cannot collect reward by always answering the majority label.
    """
    stemmer = Stemmer.Stemmer("english")
    abstracts = [_abstract(r) for r in rows]
    index = bm25s.BM25()
    index.index(bm25s.tokenize(abstracts, stopwords="en", stemmer=stemmer, show_progress=False),
                show_progress=False)
    if balance:
        by_label: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            by_label.setdefault(r["final_decision"], []).append(i)
        per = n // len(by_label)
        picked = [i for idx in by_label.values() for i in rng.sample(idx, min(per, len(idx)))]
        rng.shuffle(picked)
    else:
        picked = rng.sample(range(len(rows)), n)
    q_tokens = bm25s.tokenize([rows[i]["question"] for i in picked], stopwords="en",
                              stemmer=stemmer, show_progress=False)
    neighbors, _ = index.retrieve(q_tokens, k=n_docs + 2, show_progress=False, n_threads=8)
    samples = []
    for i, nb in zip(picked, neighbors):
        distractors = [int(j) for j in nb if int(j) != i][:n_docs]
        unanswerable = rng.random() < p_unanswerable
        docs = distractors[:n_docs] if unanswerable else [i] + distractors[:n_docs - 1]
        rng.shuffle(docs)
        samples.append({
            "question": rows[i]["question"],
            "docs": [abstracts[j] for j in docs],
            "gold": -1 if unanswerable else docs.index(i),
            "label": "insufficient" if unanswerable else rows[i]["final_decision"],
        })
    return samples


def render_user(sample: dict, max_chars: int = 1500) -> str:
    """User turn: numbered documents then the question."""
    docs = "\n\n".join(f"[{k}] {d[:max_chars]}" for k, d in enumerate(sample["docs"], 1))
    return f"Documents:\n{docs}\n\nQuestion: {sample['question']}"


def messages(sample: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": render_user(sample)}]


# --------------------------------------------------------------------------- reward

@dataclass(slots=True)
class Parsed:
    """Structured view of a completion (``answer`` is None when the format is broken)."""

    well_formed: bool
    answer: str | None
    cites: set[int]          # 1-based document numbers


def parse(text: str) -> Parsed:
    m = FORMAT.search(text.strip())
    if not m:
        return Parsed(False, None, set())
    cites = {int(c) for c in CITES.findall(m.group(1))}
    return Parsed(True, m.group(2).lower(), cites)


def reward(text: str, sample: dict) -> dict[str, float]:
    """Verifiable reward components; ``total`` = format 0.1 + answer 1.0 + citation 0.5.

    - answerable: answer must equal the label, citation must be exactly the gold doc;
      refusing ("insufficient") earns nothing — over-refusal is penalised implicitly.
    - unanswerable: answer must be "insufficient" and citation "none".
    """
    p = parse(text)
    r = {"format": 0.1 if p.well_formed else 0.0, "answer": 0.0, "citation": 0.0}
    if p.well_formed:
        if sample["gold"] < 0:
            r["answer"] = 1.0 if p.answer == "insufficient" else 0.0
            r["citation"] = 0.5 if not p.cites else 0.0
        else:
            r["answer"] = 1.0 if p.answer == sample["label"] else 0.0
            r["citation"] = 0.5 if p.cites == {sample["gold"] + 1} else 0.0
    r["total"] = r["format"] + r["answer"] + r["citation"]
    return r


def score(texts: list[str], samples: list[dict]) -> dict[str, float]:
    """Aggregate eval metrics over (completion, sample) pairs."""
    ans = [(parse(t), s) for t, s in zip(texts, samples)]
    a = [(p, s) for p, s in ans if s["gold"] >= 0]
    u = [(p, s) for p, s in ans if s["gold"] < 0]

    def frac(xs: list, pred) -> float:
        return sum(map(pred, xs)) / len(xs) if xs else float("nan")

    return {
        "n": len(ans),
        "format_rate": frac(ans, lambda x: x[0].well_formed),
        "answer_acc": frac(a, lambda x: x[0].answer == x[1]["label"]),
        "citation_acc": frac(a, lambda x: x[0].cites == {x[1]["gold"] + 1}),
        "over_refusal": frac(a, lambda x: x[0].answer == "insufficient"),
        "refusal_recall": frac(u, lambda x: x[0].answer == "insufficient"),
        "mean_reward": frac([(t, s) for t, s in zip(texts, samples)],
                            lambda x: reward(x[0], x[1])["total"]),
    }


def main() -> None:
    """Write ``train.jsonl`` (pqa_artificial) and ``eval.jsonl`` (pqa_labeled)."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pubmedqa", default="data/pubmedqa")
    p.add_argument("--out", default="data/grpo")
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--n-docs", type=int, default=4)
    p.add_argument("--p-unanswerable", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, sub, n in (("train", "pqa_artificial", args.n_train), ("eval", "pqa_labeled", args.n_eval)):
        rows = pq.read_table(next(Path(args.pubmedqa, sub).glob("*.parquet"))).to_pylist()
        samples = build_samples(rows, n, args.n_docs, args.p_unanswerable, rng,
                                balance=(split == "train"))
        with open(out / f"{split}.jsonl", "w") as f:
            f.writelines(json.dumps(s) + "\n" for s in samples)
        labels = {}
        for s in samples:
            labels[s["label"]] = labels.get(s["label"], 0) + 1
        print(f"{split}: {len(samples)} samples from {sub} ({len(rows)} rows) labels={labels}")


if __name__ == "__main__":
    main()
