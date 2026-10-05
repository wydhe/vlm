"""Build BM25 + MedCPT dense indexes for a MedRAG corpus.

Usage:
    uv run python -m rag.build_index --corpus data/medrag/textbooks/chunk \
        --out data/indexes/textbooks --article-model weights/MedCPT-Article-Encoder
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rag.retrievers import BM25Retriever, MedCPTEncoder, load_chunks


def main() -> None:
    """Parse CLI arguments and write ``ids.json``, ``dense_fp16.npy`` and ``bm25/``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", required=True, help="directory of *.jsonl chunks")
    p.add_argument("--out", required=True, help="index output directory")
    p.add_argument("--query-model", default="weights/MedCPT-Query-Encoder")
    p.add_argument("--article-model", default="weights/MedCPT-Article-Encoder")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks(args.corpus)
    print(f"{len(chunks)} chunks from {args.corpus}")
    (out / "ids.json").write_text(json.dumps([c["id"] for c in chunks]))

    t = time.perf_counter()
    BM25Retriever.build(chunks).save(out / "bm25")
    print(f"bm25 index: {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    # sort by length so each batch pads little; restore the original order afterwards
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]["content"]))
    enc = MedCPTEncoder(args.query_model, args.article_model)
    emb_sorted = enc.encode_articles([chunks[i] for i in order], args.batch_size)
    emb = np.empty_like(emb_sorted)
    emb[order] = emb_sorted
    np.save(out / "dense_fp16.npy", emb)
    print(f"dense index {emb.shape}: {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
