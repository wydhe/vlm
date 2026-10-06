"""演示界面（Gradio）：医学影像问答 + 有据医学问答。

需要先启动两个服务（两个模型共用一张卡，第一个服务占的显存会被第二个计入"已用"）：
    uv run python -m nanovllm.entrypoints.openai_server --model weights/Qwen3-VL-4B-Instruct \
        --port 8001 --gpu-memory-utilization 0.35
    uv run python -m nanovllm.entrypoints.openai_server --model weights/Qwen3-1.7B-GRPO \
        --port 8002 --gpu-memory-utilization 0.75
然后：
    uv run --extra demo python -m agent.demo --port 7860

页面：
    医学影像问答   上传图片 + 提问，由 Qwen3-VL 回答
    有据医学问答   先检索医学教材，再让 GRPO 模型引用资料作答；资料不足时改为闭卷作答并明确提示
"""

from __future__ import annotations

import argparse
import base64
import io
import os

import gradio as gr
import requests
from PIL import Image

from align.task import SYSTEM as GROUNDED_SYSTEM
from align.task import parse, render_user
from rag.retrievers import HybridRetriever

CLOSED_SYSTEM = "You are a careful medical assistant. Answer briefly."


def chat(base_url: str, model: str, messages: list[dict], max_tokens: int = 256) -> str:
    """调用 OpenAI 兼容服务，返回回复文本。"""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120,
                         proxies={"http": None, "https": None})
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_ui(args: argparse.Namespace, retriever: HybridRetriever) -> gr.Blocks:

    def answer_image(image: Image.Image | None, question: str) -> str:
        if image is None or not question.strip():
            return "请上传图片并输入问题。"
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
            {"type": "text", "text": question},
        ]}]
        return chat(args.vl_url, args.vl_model, messages)

    def answer_grounded(question: str) -> tuple[str, str]:
        if not question.strip():
            return "请输入问题。", ""
        hits = retriever.search([question], k=4, mode="hybrid_rerank")[0]
        docs = [h.content for h in hits]
        sources = "\n\n".join(f"[{i}] ({h.title}) {h.content[:600]}" for i, h in enumerate(hits, 1))

        user = render_user({"question": question, "docs": docs})
        reply = chat(args.text_url, args.text_model,
                     [{"role": "system", "content": GROUNDED_SYSTEM}, {"role": "user", "content": user}])
        parsed = parse(reply)
        if parsed.well_formed and parsed.answer != "insufficient":
            return f"{reply}\n\n（依据：文档 {sorted(parsed.cites)}）", sources

        # 资料不足：闭卷作答，并明确告诉用户这部分没有资料支撑
        fallback = chat(args.text_url, args.text_model,
                        [{"role": "system", "content": CLOSED_SYSTEM}, {"role": "user", "content": question}])
        note = "检索到的资料不足以回答该问题。以下是模型自身知识，未经资料核实，仅供参考：\n\n"
        return note + fallback, sources

    with gr.Blocks(title="医疗问答演示") as demo:
        gr.Markdown("# 医疗问答演示\n推理服务均由本项目的 nano-vllm 引擎提供。仅供技术演示，不构成医疗建议。")
        with gr.Tab("医学影像问答"):
            image = gr.Image(type="pil", label="医学影像")
            question = gr.Textbox(label="问题", placeholder="例如：Which organ is shown in this image?")
            answer = gr.Textbox(label="回答")
            gr.Button("提问").click(answer_image, [image, question], answer)
        with gr.Tab("有据医学问答"):
            question = gr.Textbox(label="问题", placeholder="例如：Does metformin cause lactic acidosis?")
            answer = gr.Textbox(label="回答", lines=6)
            sources = gr.Textbox(label="检索到的资料", lines=12)
            gr.Button("提问").click(answer_grounded, question, [answer, sources])
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description="医疗问答演示界面")
    p.add_argument("--vl-url", default="http://127.0.0.1:8001")
    p.add_argument("--vl-model", default="Qwen3-VL-4B-Instruct")
    p.add_argument("--text-url", default="http://127.0.0.1:8002")
    p.add_argument("--text-model", default="Qwen3-1.7B-GRPO")
    p.add_argument("--corpus", default="data/medrag/textbooks/chunk")
    p.add_argument("--index", default="data/indexes/textbooks")
    p.add_argument("--query-model", default="weights/MedCPT-Query-Encoder")
    p.add_argument("--reranker-model", default="weights/MedCPT-Cross-Encoder")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    # gradio 启动时会访问自己的本地地址做自检；设置了 http_proxy 的环境里这个请求会被转给代理而失败
    for name in ("NO_PROXY", "no_proxy"):
        os.environ[name] = ",".join(filter(None, [os.environ.get(name), "localhost", "127.0.0.1"]))

    retriever = HybridRetriever.from_index(args.corpus, args.index, args.query_model, args.reranker_model)
    build_ui(args, retriever).launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
