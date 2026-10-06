"""流式输出用的增量反解码：把 token 流逐步转成文本片段。"""

from transformers import PreTrainedTokenizerBase


class IncrementalDetokenizer:
    """每来一个 token，返回新增的那一小段文本，不重复解码整个输出。

    每次只解码一个很短的滑动窗口 ids[prefix_offset:]，所以每个 token 的开销是 O(1)，
    而不是随输出长度增长的 O(n)。

    多字节字符（中文、emoji 等）可能被字节级 BPE 拆到好几个 token 里。窗口末尾是半个字符时
    会解码成 U+FFFD（乱码符号），这时先不输出，等后面的 token 把字符补完整。
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, prompt_token_ids: list[int],
                 skip_special_tokens: bool = True, context: int = 5):
        """
        Args:
            tokenizer: 当前模型的 tokenizer。
            prompt_token_ids: prompt 的 token；取最后 context 个作为窗口的初始上下文，
                这样回答的第一个 token 也能在上下文里正确解码（比如前导空格）。
            skip_special_tokens: 是否去掉特殊 token（例如 EOS）。
            context: 用作初始上下文的 prompt token 数。
        """
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.ids: list[int] = list(prompt_token_ids[-context:])
        self.prefix_offset = 0              # 窗口起点
        self.read_offset = len(self.ids)    # 这之前的 token 都已经输出过

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=self.skip_special_tokens)

    def add(self, token_id: int) -> str:
        """加入一个 token，返回新确定下来的文本（可能为空）。"""
        self.ids.append(token_id)
        prefix_text = self._decode(self.ids[self.prefix_offset:self.read_offset])
        new_text = self._decode(self.ids[self.prefix_offset:])
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.ids)
            return new_text[len(prefix_text):]
        return ""

    def flush(self) -> str:
        """流结束时，把还没输出的文本全部返回。"""
        prefix_text = self._decode(self.ids[self.prefix_offset:self.read_offset])
        new_text = self._decode(self.ids[self.prefix_offset:])
        self.prefix_offset = self.read_offset = len(self.ids)
        return new_text[len(prefix_text):]
