# 自定义分词器(源语言: 中文 jieba 分词; 目标语言: 英文空格分词)
from abc import abstractmethod

from nltk import TreebankWordTokenizer, TreebankWordDetokenizer
from tqdm import tqdm


class BaseTokenizer:

    @staticmethod
    @abstractmethod
    def tokenize(text):
        pass


class ChineseTokenizer(BaseTokenizer):
    """基于 jieba 的中文分词器, 用于源语言(中文)的分词、编码和词表管理"""

    unk_token = '<unk>'
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]

    @staticmethod
    def tokenize(sentence):
        """对中文句子进行 jieba 分词"""
        return [char for char in sentence if char.strip() != '']

    @classmethod
    def build_vocab(cls, sentences, vocab_file):
        """构建中文词表并保存到文件"""
        unique_words = set()
        for sentence in tqdm(sentences, desc='中文分词'):
            for word in cls.tokenize(sentence):
                if word.strip() != "":
                    unique_words.add(word)

        # 特殊 token 放在词表开头
        vocab_list = [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token] + list(unique_words)

        with open(vocab_file, 'w', encoding='utf-8') as f:
            for word in vocab_list:
                f.write(word + '\n')

    @classmethod
    def from_vocab(cls, vocab_file):
        """从文件加载词表"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)

    def encode(self, sentence, seq_len, add_sos_eos=False):
        """将中文句子编码为定长索引序列"""
        tokens = self.tokenize(sentence)
        if add_sos_eos:
            if len(tokens) == seq_len - 2:
                tokens = [self.sos_token] + tokens + [self.eos_token]
            elif len(tokens) > seq_len - 2:
                tokens = [self.sos_token] + tokens[:seq_len - 2] + [self.eos_token]
            else:
                tokens = [self.sos_token] + tokens + [self.eos_token] + [self.pad_token] * (seq_len - 2 - len(tokens))

        indexes = [self.word2index.get(token, self.unk_token_index) for token in tokens]

        if len(indexes) >= seq_len:
            return indexes[:seq_len]
        else:
            return indexes + [self.pad_token_index] * (seq_len - len(indexes))

    def decode(self, indexes):
        """将索引序列解码为中文句子，自动过滤特殊 token"""
        special = {self.pad_token_index, self.sos_token_index, self.eos_token_index}
        tokens = [self.index2word[i] for i in indexes if i not in special]
        return ''.join(tokens)


class EnglishTokenizer:
    """基于空格分割的英文分词器, 用于目标语言(英文)的分词、编码和词表管理"""

    unk_token = '<unk>'
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]

    @staticmethod
    def tokenize(sentence):

        eng_tokenizer = TreebankWordTokenizer()
        return eng_tokenizer.tokenize(sentence.lower())

    @classmethod
    def build_vocab(cls, sentences, vocab_file):
        """构建英文词表并保存到文件"""
        unique_words = set()
        for sentence in tqdm(sentences, desc='英文分词'):
            for word in cls.tokenize(sentence):
                if word.strip() != "":
                    unique_words.add(word)

        # 特殊 token 放在词表开头
        vocab_list = [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token] + list(unique_words)

        with open(vocab_file, 'w', encoding='utf-8') as f:
            for word in vocab_list:
                f.write(word + '\n')

    @classmethod
    def from_vocab(cls, vocab_file):
        """从文件加载词表"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)

    def encode(self, sentence, seq_len, add_sos_eos=True):
        """将英文句子编码为定长索引序列(前缀加 <sos>, 后缀加 <eos>)"""
        tokens = self.tokenize(sentence)

        if add_sos_eos:
            if len(tokens) == seq_len - 2:
                tokens = [self.sos_token] + tokens + [self.eos_token]
            elif len(tokens) > seq_len - 2:
                tokens = [self.sos_token] + tokens[:seq_len - 2] + [self.eos_token]
            else:
                tokens = [self.sos_token] + tokens + [self.eos_token] + [self.pad_token] * (seq_len - 2 - len(tokens))

        return [self.word2index.get(word, self.unk_token_index) for word in tokens]

    def decode(self, indexes):
        """将索引序列解码为英文句子，自动过滤 <pad>/<sos>/<eos> 等特殊 token"""
        special = {self.pad_token_index, self.sos_token_index, self.eos_token_index}
        tokens = [self.index2word[i] for i in indexes if i not in special]
        detokenizer = TreebankWordDetokenizer()
        return detokenizer.detokenize(tokens)

    def to_tokens(self, indexes):
        """将索引序列转为过滤掉特殊 token 的字符串列表，适用于 BLEU 等指标计算"""
        special = {self.pad_token_index, self.sos_token_index, self.eos_token_index}
        return [self.index2word[i] for i in indexes if i not in special]


if __name__ == '__main__':
    chn_token = ChineseTokenizer.tokenize('今天坐地铁去上班')
    print(chn_token)
    print(EnglishTokenizer.tokenize("I'm a student"))
    detokenizer = TreebankWordDetokenizer()
    print(detokenizer.detokenize(['i', "'m", 'a', 'student']))
