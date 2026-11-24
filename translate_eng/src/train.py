"""独立 Encoder/Decoder 的训练入口。"""
import argparse
import random
import time
from itertools import chain

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    ADAM_LEARNING_RATE,
    CHECKPOINT_PATH,
    DEVICE,
    EPOCHS,
    GRAD_CLIP_NORM,
    LOG_DIR,
    RANDOM_STATE,
    SRC_VOCAB_PATH,
    TF_RATIO_END,
    TF_RATIO_START,
    TGT_VOCAB_PATH,
)
from dataset import get_dataloader
from model import Decoder, Encoder, save_checkpoint
from tokenizer import ChineseTokenizer, EnglishTokenizer


def count_parameters(*models: nn.Module) -> int:
    return sum(
        parameter.numel()
        for model in models
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def get_teacher_forcing_ratio(epoch: int, total_epochs: int) -> float:
    if total_epochs <= 1:
        return TF_RATIO_START
    progress = epoch / (total_epochs - 1)
    return TF_RATIO_START - (TF_RATIO_START - TF_RATIO_END) * progress


def forward_batch(
        encoder: Encoder,
        decoder: Decoder,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回每一步 logits 及右移后的目标，二者时间维严格对齐。"""
    _, hidden = encoder(src)
    input_token = tgt[:, 0:1]
    targets = tgt[:, 1:]
    step_logits = []

    for step in range(targets.size(1)):
        logits, hidden = decoder(input_token, hidden)
        step_logits.append(logits)

        if step + 1 == targets.size(1):
            break
        if random.random() < teacher_forcing_ratio:
            input_token = tgt[:, step + 1: step + 2]
        else:
            input_token = logits.argmax(dim=-1, keepdim=True)

    return torch.stack(step_logits, dim=1), targets


def train_one_epoch(
        encoder: Encoder,
        decoder: Decoder,
        dataloader: DataLoader,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        teacher_forcing_ratio: float,
) -> float:
    encoder.train()
    decoder.train()
    total_loss = 0.0

    for src, tgt in tqdm(dataloader, desc="训练", leave=False):
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        outputs, targets = forward_batch(
            encoder, decoder, src, tgt, teacher_forcing_ratio
        )
        loss = loss_function(
            outputs.reshape(-1, decoder.vocab_size), targets.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            chain(encoder.parameters(), decoder.parameters()), GRAD_CLIP_NORM
        )
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(epochs: int = EPOCHS) -> None:
    if epochs < 1:
        raise ValueError("epochs 必须大于等于 1")

    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    device = DEVICE
    print(f"使用设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}")

    src_tokenizer = ChineseTokenizer.from_vocab(SRC_VOCAB_PATH)
    tgt_tokenizer = EnglishTokenizer.from_vocab(TGT_VOCAB_PATH)
    dataloader = get_dataloader("train")
    encoder = Encoder(src_tokenizer.vocab_size).to(device)
    decoder = Decoder(tgt_tokenizer.vocab_size).to(device)
    print(f"可训练参数量：{count_parameters(encoder, decoder):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.pad_token_index)
    optimizer = torch.optim.Adam(
        chain(encoder.parameters(), decoder.parameters()), lr=ADAM_LEARNING_RATE
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = LOG_DIR / time.strftime("%Y-%m-%d_%H-%M-%S")
    best_loss = float("inf")
    train_start = time.time()

    with SummaryWriter(log_dir=str(log_dir)) as writer:
        for epoch in range(epochs):
            tf_ratio = get_teacher_forcing_ratio(epoch, epochs)
            print(f"===== Epoch {epoch + 1}/{epochs}，TF={tf_ratio:.2f} =====")
            epoch_start = time.time()
            average_loss = train_one_epoch(
                encoder,
                decoder,
                dataloader,
                criterion,
                optimizer,
                device,
                tf_ratio,
            )
            epoch_cost = time.time() - epoch_start
            print(f"Loss：{average_loss:.4f}，耗时：{epoch_cost:.2f}s")

            writer.add_scalar("Loss/train", average_loss, epoch)
            writer.add_scalar("TeacherForcingRatio", tf_ratio, epoch)
            writer.add_scalar("EpochTime", epoch_cost, epoch)

            if average_loss < best_loss:
                best_loss = average_loss
                save_checkpoint(
                    CHECKPOINT_PATH, encoder, decoder, epoch + 1, best_loss
                )
                print(f"已保存最优检查点：{CHECKPOINT_PATH}")

        total_cost = time.time() - train_start
        writer.add_scalar("TotalTrainTime", total_cost, 0)

    print(f"训练完成，最优训练损失：{best_loss:.4f}，总耗时：{total_cost:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="训练中译英 Encoder/Decoder")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="训练轮数")
    args = parser.parse_args()
    train(epochs=args.epochs)


if __name__ == "__main__":
    main()
