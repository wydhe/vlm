"""IncrementalDetokenizer: concatenated deltas must equal a one-shot decode."""

import os
from pathlib import Path

import pytest
from nanovllm.utils.detokenizer import IncrementalDetokenizer
from transformers import AutoTokenizer

TOKENIZER_DIR = os.environ.get(
    "MODEL_DIR", str(Path(__file__).resolve().parents[1] / "weights" / "Qwen3-0.6B"))

pytestmark = pytest.mark.skipif(not os.path.isdir(TOKENIZER_DIR),
                                reason=f"tokenizer not found at {TOKENIZER_DIR}")

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "患者男性，65岁，主诉胸痛3小时，心电图示ST段抬高。",
    "Emoji test 🩺🧬💊 and mixed 中英 text with  double  spaces.",
    "def f(x):\n    return x ** 2  # 平方\n\n\tindented\n",
    "Ünïcödé àccents, Ελληνικά, русский, العربية, 日本語のテキスト。",
]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR)


@pytest.mark.parametrize("text", TEXTS)
def test_stream_equals_full_decode(tokenizer, text):
    prompt = tokenizer.encode("Answer:")
    ids = tokenizer.encode(text)
    detok = IncrementalDetokenizer(tokenizer, prompt)
    streamed = "".join(detok.add(t) for t in ids) + detok.flush()
    assert streamed == tokenizer.decode(ids, skip_special_tokens=True)


def test_never_emits_replacement_char(tokenizer):
    ids = tokenizer.encode("🩺🧬💊中文")
    detok = IncrementalDetokenizer(tokenizer, tokenizer.encode("x"))
    assert all("�" not in detok.add(t) for t in ids)


def test_special_tokens_skipped(tokenizer):
    ids = tokenizer.encode("done") + [tokenizer.eos_token_id]
    detok = IncrementalDetokenizer(tokenizer, tokenizer.encode("x"))
    streamed = "".join(detok.add(t) for t in ids) + detok.flush()
    assert streamed == "done"
