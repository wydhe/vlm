"""Qwen3-VL on nano-vllm: correctness vs HF transformers, SLAKE accuracy and throughput.

1. Consistency: greedy outputs of our engine vs ``Qwen3VLForConditionalGeneration`` on the
   first ``--hf-n`` questions (token-exact match rate and first-divergence positions).
2. Accuracy on SLAKE (English): closed questions by exact match, open questions by
   exact match / answer-in-output.
3. Throughput: our engine (batched) vs HF ``generate`` (one request at a time).
4. Multimodal prefix cache: the same image asked twice reuses KV; a different image with
   identical text must not.

Usage:
    uv run python -m bench.vqa_eval --model weights/Qwen3-VL-4B-Instruct --label qwen3vl-4b
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from nanovllm import LLM, SamplingParams
from PIL import Image, ImageOps
from transformers import AutoProcessor

PROMPT = "{q} Answer with a single word or short phrase."


def load_slake(root: str, lang: str, n: int | None) -> list[dict]:
    items = [x for x in json.loads((Path(root) / "test.json").read_text()) if x["q_lang"] == lang]
    return items[:n] if n else items


def encode(processor, root: str, item: dict, mirror: bool = False) -> dict:
    """Chat-template one image + question; returns token ids and vision inputs."""
    image = Image.open(Path(root) / "imgs" / item["img_name"]).convert("RGB")
    if mirror:                             # new pixels, identical token ids
        image = ImageOps.mirror(image)
    msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": PROMPT.format(q=item["question"])}]}]
    enc = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                        return_dict=True, return_tensors="pt")
    return {"input_ids": enc["input_ids"][0].tolist(), "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc["image_grid_thw"]}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def correct(pred: str, item: dict) -> bool:
    p, a = norm(pred), norm(str(item["answer"]))
    if item["answer_type"] == "CLOSED":
        return p.split()[:1] == a.split()[:1]
    return p == a or bool(a and a in p)


@torch.inference_mode()
def hf_generate(model_dir: str, encoded: list[dict], max_tokens: int) -> tuple[list[list[int]], float]:
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    outs, start = [], time.perf_counter()
    for e in encoded:
        ids = torch.tensor([e["input_ids"]], device="cuda")
        o = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                           pixel_values=e["pixel_values"].cuda(), image_grid_thw=e["image_grid_thw"].cuda(),
                           max_new_tokens=max_tokens, do_sample=False)
        outs.append(o[0, ids.size(1):].tolist())
    elapsed = time.perf_counter() - start
    del model
    torch.cuda.empty_cache()
    return outs, elapsed


def strip_eos(ids: list[int], eos: set[int]) -> list[int]:
    return ids[:-1] if ids and ids[-1] in eos else ids


def main() -> None:
    """Run consistency, accuracy, throughput and prefix-cache checks; save a JSON record."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="weights/Qwen3-VL-4B-Instruct")
    p.add_argument("--slake", default="data/slake")
    p.add_argument("--lang", default="en")
    p.add_argument("--n", type=int, default=None, help="limit questions (default: all)")
    p.add_argument("--hf-n", type=int, default=64, help="questions compared against HF (0 = skip)")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--label", default="vqa")
    p.add_argument("--out-dir", default="results/vqa")
    args = p.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    items = load_slake(args.slake, args.lang, args.n)
    t = time.perf_counter()
    encoded = [encode(processor, args.slake, it) for it in items]
    print(f"{len(items)} SLAKE-{args.lang} questions, preprocessing {time.perf_counter() - t:.1f}s")

    hf_outs, hf_time = None, None
    if args.hf_n:
        hf_outs, hf_time = hf_generate(args.model, encoded[:args.hf_n], args.max_tokens)

    llm = LLM(args.model, max_model_len=4096, gpu_memory_utilization=args.gpu_memory_utilization)
    eos = {llm.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<|im_end|>")}
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    bm = llm.scheduler.block_manager
    warm = encoded[-1]                     # warmup: compile the 3-D M-RoPE / vision paths first
    llm.generate([warm["input_ids"]], sp, use_tqdm=False,
                 mm_inputs=[{k: warm[k] for k in ("pixel_values", "image_grid_thw")}])
    llm.scheduler.block_manager.reset_prefix_cache()
    t = time.perf_counter()
    res = llm.generate([e["input_ids"] for e in encoded], sp, use_tqdm=False,
                       mm_inputs=[{k: e[k] for k in ("pixel_values", "image_grid_thw")} for e in encoded])
    nano_time = time.perf_counter() - t
    outs = [r["token_ids"] for r in res]
    texts = [processor.tokenizer.decode(o, skip_special_tokens=True) for o in outs]

    record: dict = {"config": vars(args), "n": len(items)}
    if hf_outs is not None:
        k = len(hf_outs)
        a = [strip_eos(o, eos) for o in outs[:k]]
        b = [strip_eos(o, eos) for o in hf_outs]
        first = [next((i for i, (x, y) in enumerate(zip(u, v)) if x != y), None) for u, v in zip(a, b)]
        exact = sum(u == v for u, v in zip(a, b))
        record["consistency"] = {"n": k, "exact": exact,
                                 "divergence_positions": sorted(f for f in first if f is not None)}
        # answers are a few tokens long, so requests/s (incl. vision encoding) is the meaningful unit
        record["throughput"] = {"hf_req_per_s": k / hf_time, "hf_s": hf_time,
                                "nano_req_per_s": len(items) / nano_time, "nano_s": nano_time}
        print(f"consistency: {exact}/{k} token-exact vs HF; divergence at "
              f"{record['consistency']['divergence_positions']}")
        print(f"throughput: HF {record['throughput']['hf_req_per_s']:.2f} req/s (one at a time) | "
              f"nano {record['throughput']['nano_req_per_s']:.2f} req/s ({len(items)} reqs batched)")

    acc: dict[str, list[bool]] = {}
    for it, txt in zip(items, texts):
        acc.setdefault(it["answer_type"], []).append(correct(txt, it))
    record["accuracy"] = {k: sum(v) / len(v) for k, v in acc.items()} | {
        "overall": sum(sum(v) for v in acc.values()) / len(items)}
    print("accuracy:", {k: round(v, 3) for k, v in record["accuracy"].items()})

    # multimodal prefix cache, from an empty cache: a request -> no hit; the same request again
    # -> hit; the mirrored image with identical token ids -> must not hit (image hash in block hash)
    bm.reset_prefix_cache()
    mirrored = encode(processor, args.slake, items[0], mirror=True)
    assert mirrored["input_ids"] == encoded[0]["input_ids"]
    probes = (("first_time", encoded[0]), ("same_image_same_text", encoded[0]),
              ("mirrored_image_same_tokens", mirrored))
    hits = {}
    for name, e in probes:
        before = bm.num_cached_prompt_tokens
        llm.generate([e["input_ids"]], sp, use_tqdm=False,
                     mm_inputs=[{k: e[k] for k in ("pixel_values", "image_grid_thw")}])
        hits[name] = {"prompt_tokens": len(e["input_ids"]), "cached": bm.num_cached_prompt_tokens - before}
    record["mm_prefix_cache"] = hits
    print("mm prefix cache:", hits)

    record["samples"] = [{"q": it["question"], "answer": it["answer"], "pred": txt, "type": it["answer_type"]}
                         for it, txt in zip(items[:50], texts[:50])]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
    print(f"saved -> {out / f'{args.label}.json'}")


if __name__ == "__main__":
    main()
