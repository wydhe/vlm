import atexit
from dataclasses import fields
from itertools import accumulate

import xxhash
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import MMData, Sequence
from nanovllm.models.qwen3_vl import mrope_positions
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams,
                    mm_inputs: dict | None = None):
        """Queue a request. ``mm_inputs`` (Qwen3-VL): ``pixel_values`` and ``image_grid_thw``
        from the HF processor; ``prompt`` must already contain the expanded image tokens."""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        if mm_inputs is not None:
            seq.mm = self._prepare_mm(prompt, mm_inputs)
        self.scheduler.add(seq)
        return seq

    def _prepare_mm(self, prompt: list[int], mm_inputs: dict) -> MMData:
        """Encode images once and precompute M-RoPE positions / prefix-cache hash."""
        model = self.model_runner.model
        pixel_values, grid = mm_inputs["pixel_values"], mm_inputs["image_grid_thw"]
        embeds, deepstack = self.model_runner.encode_images(pixel_values, grid)
        # park features in pinned host memory: thousands of queued requests would otherwise pin
        # tens of GB of GPU memory; prefill copies only the rows of the current chunk back
        embeds = embeds.cpu().pin_memory()
        deepstack = [d.cpu().pin_memory() for d in deepstack]
        positions, delta = mrope_positions(prompt, [tuple(g) for g in grid.tolist()],
                                           model.spatial_merge_size, model.image_token_id)
        h = xxhash.xxh64(pixel_values.float().cpu().numpy().tobytes())
        h.update(grid.cpu().numpy().tobytes())
        is_img = [t == model.image_token_id for t in prompt]
        assert sum(is_img) == embeds.size(0), "image tokens in prompt != vision features"
        return MMData(embeds=embeds, deepstack=list(deepstack), positions=positions, delta=delta,
                      image_hash=h.intdigest() or 1, first_image_pos=is_img.index(True),
                      img_prefix=list(accumulate(is_img, initial=0)))

    def step_updates(self) -> tuple[list[Sequence], int]:
        """Run one engine step.

        Returns:
            The sequences that sampled a new token in this step (partially
            chunk-prefilled sequences are excluded), and the number of tokens
            processed (positive for a prefill step, negative for decode).
        """
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        # mirrors Scheduler.postprocess: a prefill emits a token only once the whole prompt is cached
        emitted = [seq for seq in seqs
                   if not is_prefill or seq.num_cached_tokens + seq.num_scheduled_tokens >= seq.num_tokens]
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        return emitted, num_tokens

    def step(self):
        emitted, num_tokens = self.step_updates()
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in emitted if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        mm_inputs: list[dict | None] | None = None,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        mm_inputs = mm_inputs or [None] * len(prompts)
        for prompt, sp, mm in zip(prompts, sampling_params, mm_inputs):
            self.add_request(prompt, sp, mm)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
