"""OpenAI 兼容的 HTTP 服务（/v1/completions、/v1/chat/completions）。

用法：
    python -m nanovllm.entrypoints.openai_server --model weights/Qwen3-8B --port 8000

支持的请求字段：prompt（字符串或 token id，单条）/ messages（可以带 image_url 图片分片，
仅 Qwen3-VL）、max_tokens、temperature（0 表示贪心）、stream、stream_options.include_usage、
ignore_eos；聊天接口还支持 tools 和 chat_template_kwargs（例如 Qwen3 的 {"enable_thinking": false}）。
"""

import argparse
import base64
import io
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoProcessor

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
    # 字符串，或 OpenAI 格式的分片列表：{"type": "text", ...} / {"type": "image_url", ...}
    content: str | list[dict[str, Any]]


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


def load_image(url: str) -> Image.Image:
    """读取图片。支持 base64 的 data URL 和 file:// 本地路径。

    故意不支持 http(s)：服务端去抓任意网址有 SSRF 风险，需要的话由客户端下载后转成 base64。
    """
    if url.startswith("data:"):
        _, encoded = url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    if url.startswith("file://"):
        return Image.open(url[len("file://"):]).convert("RGB")
    raise ValueError("image_url must be a data: URL or a file:// path")


def has_images(messages: list[ChatMessage]) -> bool:
    return any(isinstance(m.content, list) and any(p.get("type") == "image_url" for p in m.content)
               for m in messages)


def text_only(messages: list[ChatMessage]) -> list[dict]:
    """把分片列表形式的纯文本消息拼成字符串，交给普通的 chat template。"""
    result = []
    for m in messages:
        content = m.content
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if p.get("type") == "text")
        result.append({"role": m.role, "content": content})
    return result


def processor_messages(messages: list[ChatMessage]) -> list[dict]:
    """转成 HF processor 认识的格式：图片分片换成 PIL 图片。"""
    result = []
    for m in messages:
        if isinstance(m.content, str):
            parts = [{"type": "text", "text": m.content}]
        else:
            parts = []
            for p in m.content:
                if p.get("type") == "image_url":
                    parts.append({"type": "image", "image": load_image(p["image_url"]["url"])})
                elif p.get("type") == "text":
                    parts.append({"type": "text", "text": p["text"]})
        result.append({"role": m.role, "content": parts})
    return result


def build_app(engine: AsyncLLMEngine, model_name: str, processor=None) -> FastAPI:
    """创建绑定到运行中引擎的 FastAPI 应用（processor 只有 Qwen3-VL 需要）。"""
    app = FastAPI(title="nano-vllm")
    tokenizer = engine.tokenizer

    async def run(prompt_ids: list[int], req: _SamplingFields, chat: bool, mm_inputs: dict | None = None):
        rid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}"
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"
        try:
            stream = engine.generate(prompt_ids, req.sampling_params(), rid, mm_inputs)
            # 先取第一条输出，这样参数校验失败能变成 HTTP 错误码返回
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
                # 每生成一个 token 发一个分片（多字节字符还没拼完整时，text 可能是空字符串）
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
                await stream.aclose()  # 客户端断开时，会在这里取消请求
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
        template_kwargs = req.chat_template_kwargs or {}
        if not has_images(req.messages):
            ids = tokenizer.apply_chat_template(text_only(req.messages), tools=req.tools,
                                                tokenize=True, add_generation_prompt=True,
                                                **template_kwargs)
            return await run(list(ids), req, chat=True)

        if processor is None:
            return _error(400, "this model does not accept images")
        try:
            encoded = processor.apply_chat_template(processor_messages(req.messages), tokenize=True,
                                                    add_generation_prompt=True, return_dict=True,
                                                    return_tensors="pt", **template_kwargs)
        except (ValueError, OSError) as e:
            return _error(400, f"bad image input: {e}")
        mm_inputs = {"pixel_values": encoded["pixel_values"], "image_grid_thw": encoded["image_grid_thw"]}
        return await run(encoded["input_ids"][0].tolist(), req, chat=True, mm_inputs=mm_inputs)

    return app


def main() -> None:
    """解析命令行参数，加载模型，启动服务。"""
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
    processor = AutoProcessor.from_pretrained(args.model) if engine.is_multimodal else None
    uvicorn.run(build_app(engine, name, processor), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
