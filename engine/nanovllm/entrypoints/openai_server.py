"""OpenAI-compatible HTTP server (``/v1/completions``, ``/v1/chat/completions``).

Usage:
    python -m nanovllm.entrypoints.openai_server --model weights/Qwen3-8B --port 8000

Supported request fields: ``prompt`` (str or token ids, single prompt) / ``messages``,
``max_tokens``, ``temperature`` (0 = greedy), ``stream``,
``stream_options.include_usage``, ``ignore_eos`` and, for chat,
``chat_template_kwargs`` (e.g. ``{"enable_thinking": false}`` for Qwen3).
"""

import argparse
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from nanovllm.engine.async_engine import AsyncLLMEngine, EngineDeadError, RequestOutput
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.detokenizer import IncrementalDetokenizer



class StreamOptions(BaseModel):
    include_usage: bool = False


class _SamplingFields(BaseModel):
    model: str | None = None
    max_tokens: int = Field(16, ge=1)
    temperature: float = Field(1.0, ge=0.0)
    stream: bool = False
    stream_options: StreamOptions | None = None
    ignore_eos: bool = False

    def sampling_params(self) -> SamplingParams:
        return SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens,
                              ignore_eos=self.ignore_eos)


class CompletionRequest(_SamplingFields):
    prompt: str | list[int]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(_SamplingFields):
    messages: list[ChatMessage]
    tools: list[dict[str, Any]] | None = None   # OpenAI function schemas, rendered by the chat template
    chat_template_kwargs: dict[str, Any] | None = None


def _sse(payload: dict[str, Any] | str) -> str:
    return f"data: {payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)}\n\n"


def _usage(out: RequestOutput) -> dict[str, int]:
    return {"prompt_tokens": out.num_prompt_tokens,
            "completion_tokens": out.num_completion_tokens,
            "total_tokens": out.num_prompt_tokens + out.num_completion_tokens}


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "code": status}}, status_code=status)


def build_app(engine: AsyncLLMEngine, model_name: str) -> FastAPI:
    """Create the FastAPI app bound to a running engine."""
    app = FastAPI(title="nano-vllm")
    tokenizer = engine.tokenizer

    async def run(prompt_ids: list[int], req: _SamplingFields, chat: bool):
        rid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}"
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"
        try:
            stream = engine.generate(prompt_ids, req.sampling_params(), rid)
            # pull the first update eagerly so validation errors become HTTP errors
            first = await anext(stream)
        except ValueError as e:
            return _error(400, str(e))
        except EngineDeadError as e:
            return _error(503, str(e))
        detok = IncrementalDetokenizer(tokenizer, prompt_ids)

        def choice(text: str, reason: str | None, final: bool) -> dict[str, Any]:
            if chat:
                key = "message" if final else "delta"
                return {"index": 0, key: {"role": "assistant", "content": text}, "finish_reason": reason}
            return {"index": 0, "text": text, "finish_reason": reason}

        async def updates() -> AsyncIterator[tuple[str, RequestOutput]]:
            out = first
            while True:
                text = "".join(detok.add(t) for t in out.token_ids)
                if out.finished:
                    text += detok.flush()
                yield text, out
                if out.finished:
                    return
                out = await anext(stream)

        if not req.stream:
            parts, last = [], first
            async for text, last in updates():
                parts.append(text)
            return {"id": rid, "object": obj, "created": created, "model": model_name,
                    "choices": [choice("".join(parts), last.finish_reason, final=True)],
                    "usage": _usage(last)}

        include_usage = bool(req.stream_options and req.stream_options.include_usage)

        async def sse() -> AsyncIterator[str]:
            chunk_obj = f"{obj}.chunk" if chat else obj
            try:
                # one chunk per sampled token (text may be "" while a multi-byte char is pending)
                async for text, out in updates():
                    yield _sse({"id": rid, "object": chunk_obj, "created": created,
                                "model": model_name,
                                "choices": [choice(text, out.finish_reason, final=False)]})
                if include_usage:
                    yield _sse({"id": rid, "object": chunk_obj, "created": created,
                                "model": model_name, "choices": [], "usage": _usage(out)})
            except EngineDeadError as e:
                yield _sse({"error": {"message": str(e), "code": 503}})
            finally:
                await stream.aclose()  # aborts the request if the client went away
            yield _sse("[DONE]")

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok"} if engine.is_alive else {"status": "dead"},
                            status_code=200 if engine.is_alive else 503)

    @app.get("/metrics")
    async def metrics():
        return engine.stats()

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model",
                                            "owned_by": "nano-vllm",
                                            "max_model_len": engine.max_model_len}]}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        ids = tokenizer.encode(req.prompt) if isinstance(req.prompt, str) else req.prompt
        return await run(ids, req, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        ids = tokenizer.apply_chat_template([m.model_dump() for m in req.messages],
                                            tools=req.tools, tokenize=True,
                                            add_generation_prompt=True,
                                            **(req.chat_template_kwargs or {}))
        return await run(list(ids), req, chat=True)

    return app


def main() -> None:
    """Parse CLI arguments, load the model and serve."""
    p = argparse.ArgumentParser(description="nano-vllm OpenAI-compatible server")
    p.add_argument("--model", default=os.environ.get("MODEL_DIR"), help="checkpoint dir (default: $MODEL_DIR)")
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--scheduling-policy", choices=["mixed", "prefill_first"], default="mixed")
    p.add_argument("--no-piecewise-cudagraph", action="store_true",
                   help="run mixed/prefill steps eagerly (ablation)")
    args = p.parse_args()
    if not args.model:
        p.error("pass --model or set MODEL_DIR")

    engine = AsyncLLMEngine(args.model, max_model_len=args.max_model_len,
                            max_num_seqs=args.max_num_seqs,
                            max_num_batched_tokens=args.max_num_batched_tokens,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                            enforce_eager=args.enforce_eager,
                            scheduling_policy=args.scheduling_policy,
                            piecewise_cudagraph=not args.no_piecewise_cudagraph)
    name = args.served_model_name or os.path.basename(os.path.normpath(args.model))
    uvicorn.run(build_app(engine, name), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
