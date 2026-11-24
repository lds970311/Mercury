"""独立的 GRU 编码器、解码器及检查点读写。"""
from pathlib import Path

import torch
from torch import nn

from config import DROPOUT, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS


class Encoder(nn.Module):
    """使用双向 GRU 编码中文序列，并将隐状态投影到解码器维度。"""

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = EMBEDDING_DIM,
            hidden_size: int = HIDDEN_SIZE,
            num_layers: int = NUM_LAYERS,
            dropout: float = DROPOUT,
    ):
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError("双向编码器的 hidden_size 必须为偶数")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """src: [batch, src_len]；hidden: [layers, batch, hidden_size]。"""
        outputs, hidden = self.gru(self.embedding(src))
        hidden = torch.cat((hidden[0::2], hidden[1::2]), dim=2)
        return outputs, torch.tanh(self.fc_hidden(hidden))


class Decoder(nn.Module):
    """每次接收一个目标 token，输出下一步词表 logits。"""

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = EMBEDDING_DIM,
            hidden_size: int = HIDDEN_SIZE,
            num_layers: int = NUM_LAYERS,
            dropout: float = DROPOUT,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
            self, input_token: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """input_token: [batch, 1]；返回 logits: [batch, vocab_size]。"""
        embedded = self.output_dropout(self.embedding(input_token))
        output, hidden = self.gru(embedded, hidden)
        return self.linear(output.squeeze(1)), hidden


def save_checkpoint(
        path: str | Path,
        encoder: Encoder,
        decoder: Decoder,
        epoch: int,
        loss: float,
) -> None:
    """将两套权重和重建模型所需参数保存为一个检查点。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "epoch": epoch,
            "loss": loss,
            "model_config": {
                "src_vocab_size": encoder.vocab_size,
                "tgt_vocab_size": decoder.vocab_size,
                "embedding_dim": encoder.embedding_dim,
                "hidden_size": encoder.hidden_size,
                "num_layers": encoder.num_layers,
                "dropout": encoder.dropout,
            },
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
        },
        path,
    )


def load_checkpoint(
        path: str | Path, device: torch.device
) -> tuple[Encoder, Decoder]:
    """从检查点恢复编码器和解码器，并切换到评估模式。"""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if "model_config" not in checkpoint:
        raise ValueError("检查点格式过旧，请重新运行训练脚本生成新版检查点")

    config = checkpoint["model_config"]
    common = {
        "embedding_dim": config["embedding_dim"],
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "dropout": config["dropout"],
    }
    encoder = Encoder(config["src_vocab_size"], **common).to(device)
    decoder = Decoder(config["tgt_vocab_size"], **common).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder
