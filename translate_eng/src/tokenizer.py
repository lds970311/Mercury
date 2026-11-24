"""中英分词器：统一词表、定长编码和特殊 token 处理。"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

from nltk import TreebankWordDetokenizer, TreebankWordTokenizer
from tqdm import tqdm


class BaseTokenizer(ABC):
    pad_token = "<pad>"
    unk_token = "<unk>"
    sos_token = "<sos>"
    eos_token = "<eos>"
    special_tokens = (pad_token, unk_token, sos_token, eos_token)

    def __init__(self, vocab_list: Sequence[str]):
        if len(vocab_list) != len(set(vocab_list)):
            raise ValueError("词表中存在重复 token")
        if tuple(vocab_list[:4]) != self.special_tokens:
            raise ValueError("词表前四项必须依次为 <pad>、<unk>、<sos>、<eos>")

        self.vocab_list = list(vocab_list)
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = dict(enumerate(vocab_list))
        self.pad_token_index = self.word2index[self.pad_token]
        self.unk_token_index = self.word2index[self.unk_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]

    @classmethod
    @abstractmethod
    def tokenize(cls, text: str) -> list[str]:
        """将文本切分为 token。"""

    @abstractmethod
    def detokenize(self, tokens: list[str]) -> str:
        """将 token 列表还原为文本。"""

    @classmethod
    def build_vocab(cls, sentences: Iterable[str], vocab_file: str | Path) -> None:
        unique_words = {
            token
            for sentence in tqdm(sentences, desc=f"构建{cls.__name__}词表")
            for token in cls.tokenize(sentence)
            if token.strip() and token not in cls.special_tokens
        }
        vocab_list = list(cls.special_tokens) + sorted(unique_words)
        Path(vocab_file).write_text(
            "\n".join(vocab_list) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_vocab(cls, vocab_file: str | Path):
        vocab_list = Path(vocab_file).read_text(encoding="utf-8").splitlines()
        return cls(vocab_list)

    def encode(
            self, sentence: str, seq_len: int, add_sos_eos: bool = False
    ) -> list[int]:
        """编码并截断/填充到 seq_len；目标序列可自动添加首尾标记。"""
        if seq_len < (2 if add_sos_eos else 1):
            raise ValueError("seq_len 不足以容纳序列和特殊 token")

        tokens = self.tokenize(sentence)
        if add_sos_eos:
            tokens = [self.sos_token] + tokens[: seq_len - 2] + [self.eos_token]
        else:
            tokens = tokens[:seq_len]

        indexes = [
            self.word2index.get(token, self.unk_token_index) for token in tokens
        ]
        return indexes + [self.pad_token_index] * (seq_len - len(indexes))

    def to_tokens(self, indexes: Iterable[int]) -> list[str]:
        """去除首尾标记和填充，遇到 <eos> 即停止。"""
        tokens = []
        for raw_index in indexes:
            index = int(raw_index)
            if index == self.eos_token_index:
                break
            if index in (self.pad_token_index, self.sos_token_index):
                continue
            tokens.append(self.index2word[index])
        return tokens

    def decode(self, indexes: Iterable[int]) -> str:
        return self.detokenize(self.to_tokens(indexes))


class ChineseTokenizer(BaseTokenizer):
    """中文按非空白字符切分。"""

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        return [char for char in text if not char.isspace()]

    def detokenize(self, tokens: list[str]) -> str:
        return "".join(tokens)


class EnglishTokenizer(BaseTokenizer):
    """英文使用 Penn Treebank 规则切分并统一转为小写。"""

    _tokenizer = TreebankWordTokenizer()
    _detokenizer = TreebankWordDetokenizer()

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        return cls._tokenizer.tokenize(text.lower())

    def detokenize(self, tokens: list[str]) -> str:
        return self._detokenizer.detokenize(tokens)
