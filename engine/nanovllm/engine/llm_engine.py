import atexit
from dataclasses import fields
from itertools import accumulate

import torch
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
                    mm_inputs: dict | None = None, vision: tuple | None = None):
        """把一个请求加入调度队列。

        多模态（Qwen3-VL）请求需要传 mm_inputs：HF processor 输出的 pixel_values 和
        image_grid_thw，并且 prompt 里已经展开了图像占位 token。
        vision 是已经算好的视觉特征 (embeds, deepstack)；不传就在这里现算。
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        if mm_inputs is not None:
            if vision is None:
                vision = self.encode_images([mm_inputs])[0]
            seq.mm = self._prepare_mm(prompt, mm_inputs, vision)
        self.scheduler.add(seq)
        return seq

    def encode_images(self, mm_inputs: list[dict], max_patches: int = 32768) -> list[tuple]:
        """批量视觉编码：把多个请求的图片拼成一批，跑完再按请求切开。

        每批的 patch 总数不超过 max_patches，避免显存峰值过高。
        返回每个请求的 (embeds, deepstack)，放在 CPU 锁页内存里：成千上万个排队请求的
        视觉特征如果都留在 GPU 上会占几十 GB，prefill 时只把当前分块要用的行拷回 GPU。
        """
        merge = self.model_runner.model.spatial_merge_size
        results = []
        batch = []
        batch_patches = 0

        def flush():
            if not batch:
                return
            pixel_values = torch.cat([m["pixel_values"] for m in batch])
            grid = torch.cat([m["image_grid_thw"] for m in batch])
            embeds, deepstack = self.model_runner.encode_images(pixel_values, grid)
            # 每个请求占多少行 = 它所有图片的 t*h*w / merge^2
            rows = [int(m["image_grid_thw"].prod(dim=1).sum()) // (merge * merge) for m in batch]
            embed_parts = embeds.split(rows)
            deep_parts = [layer.split(rows) for layer in deepstack]   # deep_parts[层][请求]
            for i in range(len(batch)):
                request_embeds = embed_parts[i].cpu().pin_memory()
                request_deep = [deep_parts[layer][i].cpu().pin_memory() for layer in range(len(deepstack))]
                results.append((request_embeds, request_deep))
            batch.clear()

        for m in mm_inputs:
            patches = m["pixel_values"].size(0)
            if batch and batch_patches + patches > max_patches:
                flush()
                batch_patches = 0
            batch.append(m)
            batch_patches += patches
        flush()
        return results

    def _prepare_mm(self, prompt: list[int], mm_inputs: dict, vision: tuple) -> MMData:
        """准备多模态请求的状态：M-RoPE 位置、图片内容哈希（用于前缀缓存）等。"""
        model = self.model_runner.model
        pixel_values, grid = mm_inputs["pixel_values"], mm_inputs["image_grid_thw"]
        embeds, deepstack = vision
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
        # 先把所有图片批量编码，再逐个加入请求
        with_images = [m for m in mm_inputs if m is not None]
        encoded = iter(self.encode_images(with_images)) if with_images else iter(())
        for prompt, sp, mm in zip(prompts, sampling_params, mm_inputs):
            vision = next(encoded) if mm is not None else None
            self.add_request(prompt, sp, mm, vision)
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
