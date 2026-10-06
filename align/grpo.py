"""GRPO 训练器：训练和 rollout 在同一张卡、同一个进程里，rollout 用本项目的推理引擎。

每一步做的事：
    1. 采样：随机取 P 道题，每题用推理引擎生成 G 个回答（温度 1.0）
    2. 打分：用 align/task.py 里可自动判定的奖励函数给每个回答打分
    3. 优势：同一道题的 G 个回答之间做归一化  A = (r - 组均值) / (组标准差 + eps)
       一组回答得分完全相同时没有学习信号，直接跳过
    4. 更新：损失 = -A * logp + beta * KL(策略 || 参考模型)，每批只更新一次
    5. 同步：把新权重原地拷进推理引擎，并清空引擎的前缀缓存

几个容易踩坑的地方：
    - 前缀缓存：所有题目共用同一个 system prompt，如果不清空，下一步的 rollout 会复用
      旧权重算出来的 KV，训练就悄悄变成了 off-policy，而且不会报错。
    - 显存：策略模型、参考模型、梯度、优化器状态要在推理引擎初始化之前分配好，
      因为引擎会按"当前剩余显存"划分 KV cache。
    - 精度：策略模型保留 fp32 主权重、用 bf16 autocast 计算。学习率 1e-6 的更新量
      在纯 bf16 下大多会被舍入掉。

用法：
    uv run python -m align.grpo --model weights/Qwen3-1.7B --steps 150 --seed 0 --label seed0
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import time
from pathlib import Path

import torch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.loader import load_weights
from transformers import AutoModelForCausalLM, AutoTokenizer

from align.task import messages, reward, score

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "generation_config.json")


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


class Trainer:
    """策略模型、参考模型、优化器，以及同卡的 rollout 推理引擎。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.tok = AutoTokenizer.from_pretrained(args.model)

        # 策略模型：fp32 主权重 + gradient checkpointing（只有 train 模式下才生效）
        self.policy = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, attn_implementation="flash_attention_2").cuda()
        self.policy.gradient_checkpointing_enable()
        self.policy.config.use_cache = False
        self.policy.train()

        # 参考模型：冻结，只用来算 KL
        self.ref = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda().eval()
        self.ref.requires_grad_(False)

        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=args.lr, betas=(0.9, 0.99),
                                     weight_decay=0.0, fused=True)
        self.preallocate_optimizer_state()

        # 推理引擎最后创建，它会按剩余显存划分 KV cache。
        # rollout 几乎全是 decode，不需要为混合步录制分段 CUDA Graph。
        self.engine = LLMEngine(args.model, gpu_memory_utilization=args.engine_mem,
                                max_model_len=args.max_model_len, max_num_seqs=256,
                                max_num_batched_tokens=8192, piecewise_cudagraph=False)

    def preallocate_optimizer_state(self) -> None:
        """提前分配梯度和 AdamW 状态：用全零梯度执行一次 step，参数不会变化。"""
        for param in self.policy.parameters():
            param.grad = torch.zeros_like(param)
        self.opt.step()
        for state in self.opt.state.values():
            state["step"].zero_()
        self.opt.zero_grad(set_to_none=False)

    # ------------------------------------------------------------------ 采样与评测

    def prompt_ids(self, samples: list[dict]) -> list[list[int]]:
        return [self.tok.apply_chat_template(messages(s), add_generation_prompt=True,
                                             tokenize=True, enable_thinking=False)
                for s in samples]

    def generate(self, prompts: list[list[int]], temperature: float) -> list[list[int]]:
        params = SamplingParams(temperature=temperature, max_tokens=self.args.max_new_tokens)
        outputs = self.engine.generate(prompts, params, use_tqdm=False)
        return [o["token_ids"] for o in outputs]

    def evaluate(self, samples: list[dict]) -> dict[str, float]:
        """贪心解码评测，指标见 align.task.score。"""
        outputs = self.generate(self.prompt_ids(samples), temperature=0.0)
        texts = [self.tok.decode(o, skip_special_tokens=True) for o in outputs]
        metrics = score(texts, samples)
        metrics["mean_len"] = statistics.mean(len(o) for o in outputs)
        return metrics

    @torch.no_grad()
    def sync_weights(self) -> None:
        """把策略模型的权重原地拷进推理引擎，然后清空前缀缓存。"""
        load_weights(self.engine.model_runner.model, self.policy.named_parameters())
        self.engine.scheduler.block_manager.reset_prefix_cache()

    # ------------------------------------------------------------------ 训练

    def completion_logprobs(self, model, input_ids, attention_mask, spans) -> list[torch.Tensor]:
        """计算每条序列中"回答部分"每个 token 的 log 概率。

        只在回答位置上算 LM head，避免生成 [batch, 长度, 词表] 这样巨大的 logits 张量。
        spans[i] = (回答起点, 回答终点)。
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            result = []
            for row, (start, end) in enumerate(spans):
                logits = model.lm_head(hidden[row, start - 1:end - 1]).float()
                targets = input_ids[row, start:end]
                logp = logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
                result.append(logp)
        return result

    def pad_batch(self, seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """右侧补齐成矩阵，返回 input_ids 和 attention_mask。"""
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), self.tok.pad_token_id, dtype=torch.long, device="cuda")
        attention_mask = torch.zeros_like(input_ids)
        for row, s in enumerate(seqs):
            input_ids[row, :len(s)] = torch.tensor(s, device="cuda")
            attention_mask[row, :len(s)] = 1
        return input_ids, attention_mask

    def train_step(self, seqs: list[list[int]], spans: list[tuple[int, int]],
                   advantages: list[float]) -> dict[str, float]:
        """按 micro-batch 累积梯度，最后做一次优化器更新。"""
        beta = self.args.beta
        n = len(seqs)
        total_loss, total_kl = 0.0, 0.0
        for begin in range(0, n, self.args.micro_batch):
            rows = range(begin, min(begin + self.args.micro_batch, n))
            input_ids, attention_mask = self.pad_batch([seqs[i] for i in rows])
            batch_spans = [spans[i] for i in rows]
            logp = self.completion_logprobs(self.policy, input_ids, attention_mask, batch_spans)
            with torch.no_grad():
                ref_logp = self.completion_logprobs(self.ref, input_ids, attention_mask, batch_spans)

            loss = 0.0
            for j, i in enumerate(rows):
                # 每批只更新一次，重要性比率恒为 1，策略梯度项就是 -A * logp
                policy_loss = -advantages[i] * logp[j]
                # k3 估计的 KL：exp(d) - d - 1，d = 参考 logp - 策略 logp，恒为非负
                d = ref_logp[j] - logp[j]
                kl = torch.exp(d) - d - 1
                loss = loss + (policy_loss + beta * kl).mean() / n
                total_kl += kl.mean().item()
            loss.backward()
            total_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
        self.opt.step()
        self.opt.zero_grad(set_to_none=False)
        return {"loss": total_loss, "kl": total_kl / n, "grad_norm": grad_norm.item()}

    def save_bf16(self, out_dir: Path) -> None:
        """另存一份 bf16 权重，可以直接用 openai_server 部署；训练用的 fp32 权重不受影响。"""
        bf16_state = {name: t.to(torch.bfloat16) for name, t in self.policy.state_dict().items()}
        self.policy.config.dtype = torch.bfloat16
        self.policy.save_pretrained(out_dir, state_dict=bf16_state, safe_serialization=True)
        self.policy.config.dtype = torch.float32
        for name in TOKENIZER_FILES:            # tokenizer 没有被训练，直接从原模型复制
            src = Path(self.args.model) / name
            if src.exists():
                shutil.copy(src, out_dir / name)


# ---------------------------------------------------------------------- 一步 GRPO 的各个环节

def score_rollouts(trainer: Trainer, batch: list[dict], completions: list[list[int]],
                   group_size: int) -> tuple[list[float], dict[str, float]]:
    """给每个回答打分，返回总分列表和各分项的平均值。"""
    totals = []
    parts = {"format": [], "answer": [], "citation": []}
    for k, completion in enumerate(completions):
        text = trainer.tok.decode(completion, skip_special_tokens=True)
        r = reward(text, batch[k // group_size])
        totals.append(r["total"])
        for name, values in parts.items():
            values.append(r[name])
    return totals, {f"r_{name}": statistics.mean(v) for name, v in parts.items()}


def group_advantages(prompts: list[list[int]], completions: list[list[int]],
                     rewards: list[float], group_size: int):
    """组内归一化优势。返回训练序列、回答位置、优势值，以及被跳过的组数。"""
    seqs, spans, advantages = [], [], []
    skipped = 0
    for g, prompt in enumerate(prompts):
        group = rewards[g * group_size:(g + 1) * group_size]
        mean, std = statistics.mean(group), statistics.pstdev(group)
        if std < 1e-6:                      # 全组得分一样，没有学习信号
            skipped += 1
            continue
        for k in range(group_size):
            completion = completions[g * group_size + k]
            seqs.append(prompt + completion)
            spans.append((len(prompt), len(prompt) + len(completion)))
            advantages.append((group[k] - mean) / (std + 1e-4))
    return seqs, spans, advantages, skipped


def main() -> None:
    p = argparse.ArgumentParser(description="GRPO 训练（同卡 rollout）")
    p.add_argument("--model", default="weights/Qwen3-1.7B")
    p.add_argument("--train", default="data/grpo/train.jsonl")
    p.add_argument("--eval", default="data/grpo/eval.jsonl")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--prompts-per-step", type=int, default=16)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.04, help="KL 系数")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--engine-mem", type=float, default=0.6,
                   help="推理引擎的 gpu_memory_utilization，必须大于训练部分占用的比例")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="grpo")
    p.add_argument("--out-dir", default="results/grpo")
    args = p.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out_dir) / args.label
    out.mkdir(parents=True, exist_ok=True)
    train_set = read_jsonl(args.train)
    eval_set = read_jsonl(args.eval)[:args.eval_n]
    trainer = Trainer(args)

    best = {"mean_reward": float("-inf"), "step": -1}

    def evaluate_and_maybe_save(step: int) -> None:
        nonlocal best
        metrics = trainer.evaluate(eval_set)
        metrics["step"] = step
        with open(out / "eval.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
        print(f"[eval step {step}] " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k != "step"),
              flush=True)
        if step > 0 and metrics["mean_reward"] > best["mean_reward"]:
            best = metrics
            trainer.save_bf16(out / "best")
            print(f"  新的最好结果，已保存到 {out / 'best'}", flush=True)

    (out / "eval.jsonl").unlink(missing_ok=True)
    (out / "log.jsonl").unlink(missing_ok=True)
    evaluate_and_maybe_save(0)

    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        batch = rng.sample(train_set, args.prompts_per_step)
        prompts = trainer.prompt_ids(batch)
        repeated = [prompt for prompt in prompts for _ in range(args.group_size)]
        completions = trainer.generate(repeated, temperature=1.0)
        t1 = time.perf_counter()

        rewards, reward_parts = score_rollouts(trainer, batch, completions, args.group_size)
        seqs, spans, advantages, skipped = group_advantages(prompts, completions, rewards, args.group_size)
        if seqs:
            stats = trainer.train_step(seqs, spans, advantages)
        else:
            stats = {"loss": 0.0, "kl": 0.0, "grad_norm": 0.0}
        trainer.sync_weights()
        t2 = time.perf_counter()

        record = {"step": step, "reward": statistics.mean(rewards), **reward_parts,
                  "zero_var_groups": skipped, "resp_len": statistics.mean(len(c) for c in completions),
                  **stats, "t_rollout": t1 - t0, "t_train": t2 - t1}
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        if step % 5 == 0:
            print(f"step {step} reward {record['reward']:.3f} kl {record['kl']:.4f} "
                  f"len {record['resp_len']:.0f} skip {skipped} t {t2 - t0:.1f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            evaluate_and_maybe_save(step)

    trainer.save_bf16(out / "final")
    (out / "best_step.json").write_text(json.dumps(best, indent=1))
    print(f"训练结束：最好的一次在 step {best['step']}，最终权重保存在 {out / 'final'}")


if __name__ == "__main__":
    main()
