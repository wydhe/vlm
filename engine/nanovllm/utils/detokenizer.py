"""Incremental detokenization for streaming output."""

from transformers import PreTrainedTokenizerBase


class IncrementalDetokenizer:
    """Turn a token stream into text deltas without re-decoding the whole output.

    Only a short sliding window ``ids[prefix_offset:]`` is decoded per token, so
    the cost is O(1) per token instead of O(n). Text is held back while the
    window ends in an incomplete multi-byte character (decoded as U+FFFD), which
    byte-level BPE produces for e.g. Chinese characters and emoji split across
    tokens.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, prompt_token_ids: list[int],
                 skip_special_tokens: bool = True, context: int = 5):
        """
        Args:
            tokenizer: Tokenizer of the served model.
            prompt_token_ids: Prompt tokens; the last ``context`` of them seed the
                window so the first completion token is decoded in context.
            skip_special_tokens: Drop special tokens (e.g. EOS) from the text.
            context: Number of prompt tokens used as initial context.
        """
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.ids: list[int] = list(prompt_token_ids[-context:])
        self.prefix_offset = 0
        self.read_offset = len(self.ids)

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=self.skip_special_tokens)

    def add(self, token_id: int) -> str:
        """Append one token and return the newly finalized text (may be empty)."""
        self.ids.append(token_id)
        prefix_text = self._decode(self.ids[self.prefix_offset:self.read_offset])
        new_text = self._decode(self.ids[self.prefix_offset:])
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.ids)
            return new_text[len(prefix_text):]
        return ""

    def flush(self) -> str:
        """Return any held-back text at the end of the stream."""
        prefix_text = self._decode(self.ids[self.prefix_offset:self.read_offset])
        new_text = self._decode(self.ids[self.prefix_offset:])
        self.prefix_offset = self.read_offset = len(self.ids)
        return new_text[len(prefix_text):]
