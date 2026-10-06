"""Qwen3-VL 在本项目引擎上的评测：和 HF 的一致性、SLAKE 准确率、吞吐、多模态前缀缓存。

    1. 一致性：前 --hf-n 道题，本引擎和 HF transformers 的贪心输出逐 token 对比
    2. 准确率：SLAKE 英文测试集；封闭题精确匹配，开放题精确匹配或答案包含在输出里
    3. 吞吐：本引擎批量处理 vs HF 逐条 generate（答案很短，用"请求/秒"衡量）
    4. 多模态前缀缓存：同一张图再问一次应命中；像素不同但 token 完全相同的图不能命中

用法：
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


# ---------------------------------------------------------------------- 数据

def load_slake(root: str, lang: str, n: int | None) -> list[dict]:
    items = [x for x in json.loads((Path(root) / "test.json").read_text()) if x["q_lang"] == lang]
    return items[:n] if n else items


def encode(processor, root: str, item: dict, mirror: bool = False) -> dict:
    """用 chat template 编码一张图 + 一个问题，返回 token id 和视觉输入。

    mirror=True 时把图片水平翻转：像素变了，但 token id 完全不变（用于测前缀缓存）。
    """
    image = Image.open(Path(root) / "imgs" / item["img_name"]).convert("RGB")
    if mirror:
        image = ImageOps.mirror(image)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT.format(q=item["question"])},
    ]}]
    enc = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        return_dict=True, return_tensors="pt")
    return {"input_ids": enc["input_ids"][0].tolist(), "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc["image_grid_thw"]}


def vision_inputs(encoded: dict) -> dict:
    return {"pixel_values": encoded["pixel_values"], "image_grid_thw": encoded["image_grid_thw"]}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def is_correct(pred: str, item: dict) -> bool:
    """封闭题比第一个词；开放题精确匹配，或标准答案包含在回答里。"""
    pred, answer = normalize(pred), normalize(str(item["answer"]))
    if item["answer_type"] == "CLOSED":
        return pred.split()[:1] == answer.split()[:1]
    return pred == answer or bool(answer and answer in pred)


# ---------------------------------------------------------------------- 两种推理方式

@torch.inference_mode()
def hf_generate(model_dir: str, encoded: list[dict], max_tokens: int) -> tuple[list[list[int]], float]:
    """HF transformers 逐条贪心生成，作为正确性参考和速度基线。"""
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    outputs = []
    start = time.perf_counter()
    for e in encoded:
        ids = torch.tensor([e["input_ids"]], device="cuda")
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             pixel_values=e["pixel_values"].cuda(), image_grid_thw=e["image_grid_thw"].cuda(),
                             max_new_tokens=max_tokens, do_sample=False)
        outputs.append(out[0, ids.size(1):].tolist())
    seconds = time.perf_counter() - start
    del model
    torch.cuda.empty_cache()
    return outputs, seconds


def generate_with_engine(llm: LLM, encoded: list[dict], params: SamplingParams) -> tuple[list[list[int]], float]:
    """本引擎批量生成。先用一个请求预热（触发编译），再清空前缀缓存后正式计时。"""
    warm = encoded[-1]
    llm.generate([warm["input_ids"]], params, use_tqdm=False, mm_inputs=[vision_inputs(warm)])
    llm.scheduler.block_manager.reset_prefix_cache()

    start = time.perf_counter()
    results = llm.generate([e["input_ids"] for e in encoded], params, use_tqdm=False,
                           mm_inputs=[vision_inputs(e) for e in encoded])
    return [r["token_ids"] for r in results], time.perf_counter() - start


# ---------------------------------------------------------------------- 各项检查

def compare_with_hf(ours: list[list[int]], reference: list[list[int]], eos: set[int]) -> dict:
    """逐 token 对比（去掉结尾的 EOS），返回完全一致的条数和每条第一次分歧的位置。"""
    def strip_eos(ids: list[int]) -> list[int]:
        return ids[:-1] if ids and ids[-1] in eos else ids

    exact = 0
    divergence = []
    for a, b in zip(ours, reference):
        a, b = strip_eos(a), strip_eos(b)
        if a == b:
            exact += 1
            continue
        first_diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        divergence.append(first_diff)
    return {"n": len(reference), "exact": exact, "divergence_positions": sorted(divergence)}


def accuracy_by_type(items: list[dict], texts: list[str]) -> dict:
    correct: dict[str, list[bool]] = {}
    for item, text in zip(items, texts):
        correct.setdefault(item["answer_type"], []).append(is_correct(text, item))
    result = {kind: sum(v) / len(v) for kind, v in correct.items()}
    result["overall"] = sum(sum(v) for v in correct.values()) / len(items)
    return result


def probe_prefix_cache(llm: LLM, params: SamplingParams, first: dict, mirrored: dict) -> dict:
    """从空缓存开始：首次请求 → 同一请求再来一次 → 像素不同但 token 相同的图。"""
    bm = llm.scheduler.block_manager
    bm.reset_prefix_cache()
    hits = {}
    for name, e in (("first_time", first), ("same_image_same_text", first),
                    ("mirrored_image_same_tokens", mirrored)):
        before = bm.num_cached_prompt_tokens
        llm.generate([e["input_ids"]], params, use_tqdm=False, mm_inputs=[vision_inputs(e)])
        hits[name] = {"prompt_tokens": len(e["input_ids"]), "cached": bm.num_cached_prompt_tokens - before}
    return hits


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-VL 评测")
    p.add_argument("--model", default="weights/Qwen3-VL-4B-Instruct")
    p.add_argument("--slake", default="data/slake")
    p.add_argument("--lang", default="en")
    p.add_argument("--n", type=int, default=None, help="只评测前 n 题（默认全部）")
    p.add_argument("--hf-n", type=int, default=64, help="和 HF 对比的题数（0 表示不对比）")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--label", default="vqa")
    p.add_argument("--out-dir", default="results/vqa")
    args = p.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    items = load_slake(args.slake, args.lang, args.n)
    encoded = [encode(processor, args.slake, item) for item in items]
    print(f"SLAKE-{args.lang}：{len(items)} 道题")

    hf_outputs = hf_seconds = None
    if args.hf_n:
        hf_outputs, hf_seconds = hf_generate(args.model, encoded[:args.hf_n], args.max_tokens)

    llm = LLM(args.model, max_model_len=4096, gpu_memory_utilization=args.gpu_memory_utilization)
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    outputs, seconds = generate_with_engine(llm, encoded, params)
    texts = [processor.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    record: dict = {"config": vars(args), "n": len(items)}
    if hf_outputs is not None:
        eos = {llm.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<|im_end|>")}
        record["consistency"] = compare_with_hf(outputs[:len(hf_outputs)], hf_outputs, eos)
        record["throughput"] = {"hf_req_per_s": len(hf_outputs) / hf_seconds, "hf_s": hf_seconds,
                                "nano_req_per_s": len(items) / seconds, "nano_s": seconds}
        print(f"一致性：{record['consistency']['exact']}/{len(hf_outputs)} 逐 token 一致，"
              f"分歧位置 {record['consistency']['divergence_positions']}")
        print(f"吞吐：HF 逐条 {record['throughput']['hf_req_per_s']:.2f} req/s，"
              f"本引擎批量 {record['throughput']['nano_req_per_s']:.2f} req/s")

    record["accuracy"] = accuracy_by_type(items, texts)
    print("准确率：", {k: round(v, 3) for k, v in record["accuracy"].items()})

    mirrored = encode(processor, args.slake, items[0], mirror=True)
    assert mirrored["input_ids"] == encoded[0]["input_ids"]
    record["mm_prefix_cache"] = probe_prefix_cache(llm, params, encoded[0], mirrored)
    print("多模态前缀缓存：", record["mm_prefix_cache"])

    record["samples"] = [{"q": it["question"], "answer": it["answer"], "pred": text, "type": it["answer_type"]}
                         for it, text in zip(items[:50], texts[:50])]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.label}.json"
    out_file.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    print(f"结果已保存到 {out_file}")


if __name__ == "__main__":
    main()
