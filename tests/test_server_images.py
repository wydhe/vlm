"""服务端图片输入的辅助函数（只测 CPU 逻辑，不启动模型）。"""

import base64
import io

import pytest
from nanovllm.entrypoints.openai_server import (
    ChatMessage,
    has_images,
    load_image,
    processor_messages,
    text_only,
)
from PIL import Image


def png_data_url(size=(8, 6)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(255, 0, 0)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_plain_string_message_has_no_images():
    messages = [ChatMessage(role="user", content="hello")]
    assert not has_images(messages)
    assert text_only(messages) == [{"role": "user", "content": "hello"}]


def test_text_parts_are_joined():
    messages = [ChatMessage(role="user", content=[{"type": "text", "text": "a "},
                                                  {"type": "text", "text": "b"}])]
    assert text_only(messages) == [{"role": "user", "content": "a b"}]


def test_image_part_is_detected_and_decoded():
    messages = [ChatMessage(role="user", content=[
        {"type": "image_url", "image_url": {"url": png_data_url()}},
        {"type": "text", "text": "what is this?"},
    ])]
    assert has_images(messages)
    parts = processor_messages(messages)[0]["content"]
    assert parts[0]["type"] == "image" and parts[0]["image"].size == (8, 6)
    assert parts[1] == {"type": "text", "text": "what is this?"}


def test_local_file_path(tmp_path):
    path = tmp_path / "x.png"
    Image.new("RGB", (4, 4)).save(path)
    assert load_image(f"file://{path}").size == (4, 4)


def test_remote_urls_are_rejected():
    with pytest.raises(ValueError):
        load_image("http://example.com/x.png")
