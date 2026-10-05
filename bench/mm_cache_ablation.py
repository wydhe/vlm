"""Ablation: image content hash in prefix-cache block hashes.

Pairs of SLAKE images with identical token ids (same grid size) are asked the same
question. With the hash, image B never reuses image A's KV; without it, B silently hits
A's cached blocks and may answer about the wrong image.

Usage:
    uv run python -m bench.mm_cache_ablation
"""
from collections import defaultdict

from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager
from PIL import Image
from transformers import AutoProcessor

from bench.vqa_eval import encode, load_slake

model = "weights/Qwen3-VL-4B-Instruct"
proc = AutoProcessor.from_pretrained(model)
items = load_slake("data/slake", "en", None)
by_size = defaultdict(dict)
for x in items:
    by_size[Image.open(f"data/slake/imgs/{x['img_name']}").size].setdefault(x["location"], x)
pairs = [list(d.values())[i:i + 2] for d in by_size.values() for i in range(len(d) - 1)][:12]
llm = LLM(model, max_model_len=4096, gpu_memory_utilization=0.6)
sp = SamplingParams(temperature=0, max_tokens=12)
bm = llm.scheduler.block_manager
real_extra = BlockManager._extra
def ask(e):
    before = bm.num_cached_prompt_tokens
    out = llm.generate([e["input_ids"]], sp, use_tqdm=False,
                       mm_inputs=[{k: e[k] for k in ("pixel_values", "image_grid_thw")}])[0]["text"]
    return out.replace("<|im_end|>", "").strip(), bm.num_cached_prompt_tokens - before
q = {"question": "Which body part is shown in this image?", "answer_type": "OPEN"}
n_pairs = n_hit_bug = n_wrong = 0
for x, y in pairs:
    a, b = encode(proc, "data/slake", {**x, **q}), encode(proc, "data/slake", {**y, **q})
    if a["input_ids"] != b["input_ids"]:
        continue
    n_pairs += 1
    res = {}
    for mode, extra in (("hash", real_extra), ("nohash", lambda self, seq, i: 0)):
        BlockManager._extra = extra
        bm.reset_prefix_cache()
        ask(a)
        res[mode] = ask(b)
    BlockManager._extra = real_extra
    bm.reset_prefix_cache()
    alone = ask(b)                       # B with a cold cache = reference answer for B
    n_hit_bug += res["nohash"][1] > 0
    n_wrong += res["nohash"][0] != alone[0]
    print(f"{x['location']:>8} -> {y['location']:<8} | B alone {alone[0]!r:14} | B after A (hash) "
          f"{res['hash'][0]!r:14} cached {res['hash'][1]:3} | (no hash) {res['nohash'][0]!r:14} cached {res['nohash'][1]:3}")
print(f"pairs {n_pairs}: without image hash, B reused A's KV in {n_hit_bug}, answer changed in {n_wrong}")
