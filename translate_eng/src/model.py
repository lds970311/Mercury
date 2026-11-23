# 模型结构定义(Seq2Seq: Encoder-Decoder with GRU)
import torch
from torch import nn

from config import EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT


class Encoder(nn.Module):
    """双向 GRU 编码器: Embedding -> BiGRU, 同时捕获源句前后文信息"""

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_DIM)
        # 双向 GRU: 每个方向 HIDDEN_SIZE//2, 拼接后输出维度仍为 HIDDEN_SIZE
        self.gru = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE // 2,
                          num_layers=NUM_LAYERS, batch_first=True,
                          bidirectional=True, dropout=DROPOUT)
        # 将双向隐状态合并投影为解码器所需的隐状态维度
        self.fc_hidden = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, src):
        # src.shape: [batch_size, src_seq_len]
        embedded = self.embedding(src)
        # outputs.shape: [batch_size, src_seq_len, HIDDEN_SIZE] (双向拼接)
        # hidden.shape: [num_layers*2, batch_size, HIDDEN_SIZE//2]
        outputs, hidden = self.gru(embedded)

        # 合并双向隐状态: 将每层的 forward 和 backward 拼接
        # hidden: [num_layers*2, batch_size, HIDDEN_SIZE//2] -> [num_layers, batch_size, HIDDEN_SIZE]
        # 偶数索引为 forward, 奇数索引为 backward
        hidden = torch.cat((hidden[0::2], hidden[1::2]), dim=2)
        hidden = torch.tanh(self.fc_hidden(hidden))

        return outputs, hidden


class Decoder(nn.Module):
    """解码器: Embedding -> GRU -> Linear, 逐步生成目标序列"""

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_DIM)
        self.gru = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE,
                          num_layers=NUM_LAYERS, batch_first=True, dropout=DROPOUT)
        self.linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=vocab_size)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, input_token, hidden):
        # input_token.shape: [batch_size, 1]
        embedded = self.dropout(self.embedding(input_token))
        # output.shape: [batch_size, 1, hidden_size]
        output, hidden = self.gru(embedded, hidden)
        # prediction.shape: [batch_size, 1, tgt_vocab_size]
        prediction = self.linear(output)
        return prediction, hidden


class Seq2Seq(nn.Module):
    """Seq2Seq 翻译模型: 编码器将源语言(中文)编码为上下文向量, 解码器逐步生成目标语言(英文)"""

    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.encoder = Encoder(src_vocab_size)
        self.decoder = Decoder(tgt_vocab_size)
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src: 源序列 [batch_size, src_seq_len]
            tgt: 目标序列 [batch_size, tgt_seq_len] (含 <sos> 和 <eos>)
            teacher_forcing_ratio: 使用 teacher forcing 的概率
        Returns:
            outputs: [batch_size, tgt_seq_len, tgt_vocab_size] 每步的预测 logits
        """
        batch_size = src.size(0)
        tgt_seq_len = tgt.size(1)

        # 存储所有时间步的预测
        outputs = torch.zeros(batch_size, tgt_seq_len, self.tgt_vocab_size).to(src.device)

        # 编码源序列
        _, hidden = self.encoder(src)

        # 解码: 从 <sos> 开始逐步生成
        input_token = tgt[:, 0:1]  # <sos> token, shape [batch_size, 1]

        for t in range(tgt_seq_len):
            prediction, hidden = self.decoder(input_token, hidden)
            outputs[:, t:t + 1, :] = prediction

            # teacher forcing: 以一定概率使用真实目标作为下一步输入
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tgt[:, t:t + 1]
            else:
                input_token = prediction.argmax(dim=-1)

        return outputs
