"""GRPO with in-process nano-vllm rollouts and live weight sync (single GPU).

One step::

    sample P prompts ──► nano-vllm engine: G completions each (T=1.0)
          │                         ▲
          ▼                         │ in-place weight sync + prefix-cache reset
    verifiable rewards ──► group-normalised advantages ──► policy-gradient update (HF model)

- Advantage: ``A_i = (r_i - mean(r_group)) / (std(r_group) + eps)``; groups whose rewards are
  all equal carry no signal and are skipped (saves backward compute).
- Loss (one update per batch, so the PPO ratio is 1 and clipping is inactive):
  ``-mean_seq mean_tok [ A * logp - beta * KL_k3(pi || pi_ref) ]``.
- Policy keeps fp32 master weights (bf16 autocast for compute): at lr ~1e-6 most updates
  would round away in pure bf16. The engine receives bf16 copies.
- The engine shares the GPU with the trainer. Trainer memory (policy, reference, grads,
  AdamW states) is allocated *before* the engine sizes its KV cache.
- After every update the engine's prefix cache is reset: all prompts share the system
  prompt, and KV cached under the old weights would make rollouts silently off-policy.

Usage:
    uv run python -m align.grpo --model weights/Qwen3-1.7B --steps 150 --label grpo-1.7b
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.loader import load_weights
from transformers import AutoModelForCausalLM, AutoTokenizer

from align.task import messages, reward, score


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


class Trainer:
    """Policy / reference models, optimizer and the co-located rollout engine."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.tok = AutoTokenizer.from_pretrained(args.model)
        self.policy = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, attn_implementation="flash_attention_2").cuda()
        self.policy.gradient_checkpointing_enable()
        self.policy.config.use_cache = False
        self.policy.train()                   # checkpointing is only active in train mode
        self.ref = None
        if args.beta > 0:
            self.ref = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda().eval()
            self.ref.requires_grad_(False)
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=args.lr, betas=(0.9, 0.99),
                                     weight_decay=0.0, fused=True)
        self._preallocate_optimizer()
        self.engine = LLMEngine(args.model, gpu_memory_utilization=args.engine_mem,
                                max_model_len=args.max_model_len, max_num_seqs=256,
                                max_num_batched_tokens=8192,
                                # rollouts are decode-dominated; skip capturing hundreds of graphs
                                piecewise_cudagraph=False)

    def _preallocate_optimizer(self) -> None:
        """Materialise grads and AdamW states now, so the engine's KV sizing sees them."""
        for p in self.policy.parameters():
            p.grad = torch.zeros_like(p)
        self.opt.step()                       # zero grads, zero weight decay: a no-op update
        for state in self.opt.state.values():
            state["step"].zero_()
        self.opt.zero_grad(set_to_none=False)

    # ------------------------------------------------------------------ rollout

    def prompt_ids(self, samples: list[dict]) -> list[list[int]]:
        return [self.tok.apply_chat_template(messages(s), add_generation_prompt=True, tokenize=True,
                                             enable_thinking=False) for s in samples]

    def generate(self, prompts: list[list[int]], temperature: float, max_tokens: int) -> list[list[int]]:
        outs = self.engine.generate(prompts, SamplingParams(temperature=temperature, max_tokens=max_tokens),
                                    use_tqdm=False)
        return [o["token_ids"] for o in outs]

    @torch.no_grad()
    def sync_weights(self) -> None:
        """Copy the policy into the engine in place and drop stale prefix-cache entries."""
        load_weights(self.engine.model_runner.model, self.policy.named_parameters())
        self.engine.scheduler.block_manager.reset_prefix_cache()

    # ------------------------------------------------------------------ training

    def _completion_logp(self, model: torch.nn.Module, input_ids: torch.Tensor, attn: torch.Tensor,
                         spans: list[tuple[int, int]]) -> list[torch.Tensor]:
        """Per-token log-probs of each row's completion span ``[start, end)``.

        The LM head runs only on completion positions, avoiding a [B, T, vocab] logits tensor.
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(input_ids=input_ids, attention_mask=attn).last_hidden_state
            out = []
            for b, (s, e) in enumerate(spans):
                logits = model.lm_head(hidden[b, s - 1:e - 1]).float()
                out.append(logits.log_softmax(-1).gather(-1, input_ids[b, s:e, None]).squeeze(-1))
        return out

    def train_on(self, seqs: list[list[int]], spans: list[tuple[int, int]],
                 advantages: list[float]) -> dict[str, float]:
        """Accumulate the GRPO loss over micro-batches and take one optimizer step."""
        a = self.args
        n = len(seqs)
        kl_sum, loss_sum = 0.0, 0.0
        for i in range(0, n, a.micro_batch):
            idx = range(i, min(i + a.micro_batch, n))
            width = max(len(seqs[j]) for j in idx)
            ids = torch.full((len(idx), width), self.tok.pad_token_id, dtype=torch.long, device="cuda")
            attn = torch.zeros_like(ids)
            for r, j in enumerate(idx):
                ids[r, :len(seqs[j])] = torch.tensor(seqs[j], device="cuda")
                attn[r, :len(seqs[j])] = 1
            mb_spans = [spans[j] for j in idx]
            logp = self._completion_logp(self.policy, ids, attn, mb_spans)
            ref_logp = None
            if self.ref is not None:
                with torch.no_grad():
                    ref_logp = self._completion_logp(self.ref, ids, attn, mb_spans)
            loss = 0.0
            for r, j in enumerate(idx):
                per_tok = -advantages[j] * logp[r]                         # ratio = 1 at mu = 1
                if ref_logp is not None:
                    d = ref_logp[r] - logp[r]
                    kl = torch.exp(d) - d - 1                              # k3 estimator, >= 0
                    per_tok = per_tok + a.beta * kl
                    kl_sum += kl.mean().item()
                loss = loss + per_tok.mean() / n
            loss.backward()
            loss_sum += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), a.max_grad_norm).item()
        self.opt.step()
        self.opt.zero_grad(set_to_none=False)
        return {"loss": loss_sum, "kl": kl_sum / n, "grad_norm": grad_norm}

    # ------------------------------------------------------------------ eval

    def evaluate(self, samples: list[dict]) -> dict[str, float]:
        """Greedy decoding through the engine; task metrics from ``align.task.score``."""
        outs = self.generate(self.prompt_ids(samples), temperature=0.0, max_tokens=self.args.max_new_tokens)
        texts = [self.tok.decode(o, skip_special_tokens=True) for o in outs]
        return score(texts, samples) | {"mean_len": statistics.mean(map(len, outs))}


def main() -> None:
    """Run GRPO, logging per-step stats and periodic greedy evals."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="weights/Qwen3-1.7B")
    p.add_argument("--train", default="data/grpo/train.jsonl")
    p.add_argument("--eval", default="data/grpo/eval.jsonl")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--prompts-per-step", type=int, default=16)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--engine-mem", type=float, default=0.6,
                   help="engine gpu_memory_utilization; must exceed the trainer's share")
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
    train, evalset = read_jsonl(args.train), read_jsonl(args.eval)[:args.eval_n]
    t = Trainer(args)
    # kept open for the whole run (flushed per record), closed at the end
    log_f = open(out / "log.jsonl", "w")  # noqa: SIM115
    eval_f = open(out / "eval.jsonl", "w")  # noqa: SIM115

    def do_eval(step: int) -> None:
        m = t.evaluate(evalset) | {"step": step}
        eval_f.write(json.dumps(m) + "\n")
        eval_f.flush()
        print(f"[eval step {step}] " + " ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "step"),
              flush=True)

    do_eval(0)
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        batch = rng.sample(train, args.prompts_per_step)
        prompts = t.prompt_ids(batch)
        rep_prompts = [pr for pr in prompts for _ in range(args.group_size)]
        completions = t.generate(rep_prompts, temperature=1.0, max_tokens=args.max_new_tokens)
        t1 = time.perf_counter()

        rewards, comps = [], {"format": [], "answer": [], "citation": []}
        for k, c in enumerate(completions):
            r = reward(t.tok.decode(c, skip_special_tokens=True), batch[k // args.group_size])
            rewards.append(r["total"])
            for key, values in comps.items():
                values.append(r[key])
        seqs, spans, adv, skipped = [], [], [], 0
        for g in range(args.prompts_per_step):
            rs = rewards[g * args.group_size:(g + 1) * args.group_size]
            mu, sd = statistics.mean(rs), statistics.pstdev(rs)
            if sd < 1e-6:
                skipped += 1
                continue
            for k in range(args.group_size):
                c = completions[g * args.group_size + k]
                pr = prompts[g]
                seqs.append(pr + c)
                spans.append((len(pr), len(pr) + len(c)))
                adv.append((rs[k] - mu) / (sd + 1e-4))
        stats = t.train_on(seqs, spans, adv) if seqs else {"loss": 0.0, "kl": 0.0, "grad_norm": 0.0}
        t.sync_weights()
        t2 = time.perf_counter()
        rec = {"step": step, "reward": statistics.mean(rewards),
               **{f"r_{k}": statistics.mean(v) for k, v in comps.items()},
               "zero_var_groups": skipped, "resp_len": statistics.mean(map(len, completions)),
               **stats, "t_rollout": t1 - t0, "t_train": t2 - t1}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if step % 5 == 0:
            print(f"step {step} reward {rec['reward']:.3f} (fmt {rec['r_format']:.2f} ans {rec['r_answer']:.2f} "
                  f"cite {rec['r_citation']:.2f}) kl {rec['kl']:.4f} len {rec['resp_len']:.0f} "
                  f"skip {skipped} t {t2 - t0:.1f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            do_eval(step)
    log_f.close()
    eval_f.close()
    t.policy.save_pretrained(out / "policy", safe_serialization=True)
    t.tok.save_pretrained(out / "policy")
    print(f"saved policy -> {out / 'policy'}")


if __name__ == "__main__":
    main()
