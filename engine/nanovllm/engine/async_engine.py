"""Asyncio front-end for ``LLMEngine``: one engine thread, per-request token streams.

Threading model::

    asyncio loop (HTTP)                         engine thread
    ───────────────────                         ─────────────
    generate() ──("add", req)──► inbox ──► drain inbox → add_request / abort
                                              │
                                              ▼
                                        step_updates()  (GPU)
                                              │  one call_soon_threadsafe per step
    req.queue ◄──────── _deliver(batch) ◄─────┘

The engine is constructed *inside* its thread because ``torch.set_default_device``
and ``torch.cuda.set_device`` are thread-local.
"""

import asyncio
import queue
import threading
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


@dataclass(slots=True)
class RequestOutput:
    """One streaming update of a request."""

    request_id: str
    token_ids: list[int]          # tokens sampled since the previous update
    finished: bool
    finish_reason: str | None     # "stop" (EOS) | "length" (max_tokens) | None
    num_prompt_tokens: int
    num_completion_tokens: int


@dataclass(slots=True)
class _Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    loop: asyncio.AbstractEventLoop
    mm_inputs: dict | None = None       # Qwen3-VL: pixel_values + image_grid_thw
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    seq: Sequence | None = None


def _deliver(items: list[tuple[asyncio.Queue, Any]]) -> None:
    """Runs on the event loop: fan one engine step's outputs out to request queues."""
    for q, item in items:
        q.put_nowait(item)


class EngineDeadError(RuntimeError):
    """The engine thread has crashed or been shut down."""


class AsyncLLMEngine:
    """Serve concurrent async requests from a single ``LLMEngine``."""

    def __init__(self, model: str, **engine_kwargs: Any):
        """Start the engine thread and block until the model is loaded.

        Args:
            model: Local checkpoint directory.
            **engine_kwargs: Forwarded to ``LLMEngine`` / ``Config``.
        """
        self._inbox: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        self._by_id: dict[str, _Request] = {}      # engine-thread only
        self._by_seq: dict[int, _Request] = {}     # engine-thread only
        self._error: BaseException | None = None
        self._ready = threading.Event()
        self._engine: LLMEngine | None = None
        self._thread = threading.Thread(target=self._run, args=(model, engine_kwargs),
                                        name="nanovllm-engine", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            raise EngineDeadError("engine failed to start") from self._error
        config = self._engine.model_runner.config
        self.tokenizer = self._engine.tokenizer
        self.is_multimodal: bool = config.mm_config is not None
        self.max_model_len: int = config.max_model_len
        self._max_seq_tokens: int = config.num_kvcache_blocks * config.kvcache_block_size

    # ---------------------------------------------------------------- public API

    @property
    def is_alive(self) -> bool:
        """Whether the engine thread is running and healthy."""
        return self._error is None and self._thread.is_alive()

    async def generate(self, prompt_token_ids: list[int], sampling_params: SamplingParams,
                       request_id: str | None = None,
                       mm_inputs: dict | None = None) -> AsyncIterator[RequestOutput]:
        """Submit a request and stream its outputs.

        Closing the iterator early (e.g. client disconnect) aborts the request and
        frees its KV cache blocks. ``mm_inputs`` carries image inputs for Qwen3-VL.

        Raises:
            ValueError: The request can never be scheduled.
            EngineDeadError: The engine thread is not running.
        """
        self._validate(prompt_token_ids, sampling_params)
        if mm_inputs is not None and not self.is_multimodal:
            raise ValueError("this model does not accept images")
        req = _Request(request_id or uuid.uuid4().hex, list(prompt_token_ids), sampling_params,
                       asyncio.get_running_loop(), mm_inputs)
        self._inbox.put(("add", req))
        finished = False
        try:
            while True:
                item = await req.queue.get()
                if isinstance(item, BaseException):
                    raise EngineDeadError("engine thread crashed") from item
                yield item
                if item.finished:
                    finished = True
                    return
        finally:
            if not finished:
                self._inbox.put(("abort", req.request_id))

    def stats(self) -> dict[str, float]:
        """Cumulative prefix-cache counters (read racily from the engine thread; ints only)."""
        bm = self._engine.scheduler.block_manager
        total, cached = bm.num_prompt_tokens, bm.num_cached_prompt_tokens
        return {"prompt_tokens": total, "cached_prompt_tokens": cached,
                "prefix_cache_hit_rate": cached / total if total else 0.0,
                "free_kv_blocks": len(bm.free_block_ids), "total_kv_blocks": len(bm.blocks)}

    def shutdown(self) -> None:
        """Stop the engine loop and wait for the thread to exit."""
        self._inbox.put(("stop", None))
        self._thread.join()

    # ------------------------------------------------------------ validation

    def _validate(self, prompt_token_ids: list[int], sp: SamplingParams) -> None:
        if not self.is_alive:
            raise EngineDeadError("engine is not running") from self._error
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        total = len(prompt_token_ids) + sp.max_tokens
        if total > self.max_model_len:
            raise ValueError(f"prompt ({len(prompt_token_ids)}) + max_tokens ({sp.max_tokens}) "
                             f"exceeds max_model_len ({self.max_model_len})")
        if total > self._max_seq_tokens:
            # would sit at the head of the waiting queue forever and block everyone
            raise ValueError(f"request needs {total} tokens of KV cache, "
                             f"only {self._max_seq_tokens} available")

    # ------------------------------------------------------------ engine thread

    def _run(self, model: str, engine_kwargs: dict[str, Any]) -> None:
        try:
            self._engine = LLMEngine(model, **engine_kwargs)
        except BaseException as e:  # noqa: BLE001 - surfaced to the constructor
            self._error = e
            self._ready.set()
            return
        self._ready.set()
        engine = self._engine
        try:
            while self._drain_inbox(block=engine.is_finished()):
                if engine.is_finished():
                    continue
                emitted, _ = engine.step_updates()
                self._publish(emitted)
        except BaseException as e:  # noqa: BLE001 - propagate to all waiters
            self._error = e
            self._fail_all(e)

    def _drain_inbox(self, block: bool) -> bool:
        """Apply pending add/abort commands. Returns False on shutdown."""
        try:
            cmd, arg = self._inbox.get() if block else self._inbox.get_nowait()
        except queue.Empty:
            return True
        while True:
            if cmd == "stop":
                return False
            if cmd == "add":
                arg.seq = self._engine.add_request(arg.prompt_token_ids, arg.sampling_params,
                                                   arg.mm_inputs)
                self._by_id[arg.request_id] = arg
                self._by_seq[arg.seq.seq_id] = arg
            elif cmd == "abort":
                req = self._by_id.pop(arg, None)
                if req is not None:
                    del self._by_seq[req.seq.seq_id]
                    self._engine.scheduler.abort(req.seq)
            try:
                cmd, arg = self._inbox.get_nowait()
            except queue.Empty:
                return True

    def _publish(self, emitted: list[Sequence]) -> None:
        eos = self._engine.scheduler.eos
        batches: dict[asyncio.AbstractEventLoop, list] = defaultdict(list)
        for seq in emitted:
            req = self._by_seq.get(seq.seq_id)
            if req is None:
                continue
            reason = None
            if seq.is_finished:
                reason = "stop" if (not seq.ignore_eos and seq.last_token == eos) else "length"
                del self._by_seq[seq.seq_id]
                del self._by_id[req.request_id]
            out = RequestOutput(req.request_id, [seq.last_token], seq.is_finished, reason,
                                seq.num_prompt_tokens, seq.num_completion_tokens)
            batches[req.loop].append((req.queue, out))
        for loop, items in batches.items():
            try:
                loop.call_soon_threadsafe(_deliver, items)
            except RuntimeError:  # event loop already closed
                pass

    def _fail_all(self, error: BaseException) -> None:
        batches: dict[asyncio.AbstractEventLoop, list] = defaultdict(list)
        for req in self._by_id.values():
            batches[req.loop].append((req.queue, error))
        self._by_id.clear()
        self._by_seq.clear()
        for loop, items in batches.items():
            try:
                loop.call_soon_threadsafe(_deliver, items)
            except RuntimeError:
                pass
