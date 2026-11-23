# 模型训练脚本
import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (PROCESSED_DIR, ADAM_LEARNING_RATE, EPOCHS,
                    LOG_DIR, MODELS_DIR, DEVICE_TYPE)
from dataset import get_dataloader
from model import Seq2Seq
from tokenizer import ChineseTokenizer, EnglishTokenizer

# Teacher Forcing 衰减参数: 初期高比例加速收敛, 后期低比例缓解 exposure bias
TF_RATIO_START = 0.9
TF_RATIO_END = 0.1


def count_parameters(model):
    """统计模型可训练参数总量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_teacher_forcing_ratio(epoch, total_epochs):
    """线性衰减: 从 TF_RATIO_START 逐步降至 TF_RATIO_END"""
    return TF_RATIO_START - (TF_RATIO_START - TF_RATIO_END) * (epoch / (total_epochs - 1))


def train_one_epoch(model, dataloader, loss_function, optimizer, device, tf_ratio):
    """
    在给定的数据加载器上训练模型一个完整的轮次, 返回本轮所有 batch 的平均损失。

    Args:
        model: Seq2Seq 翻译模型
        dataloader: 每次迭代返回 (src, tgt) 的元组
        loss_function: 损失函数(CrossEntropyLoss, 忽略 <pad>)
        optimizer: 优化器实例
        device: 运行设备
        tf_ratio: 当前 epoch 的 Teacher Forcing 比例

    Returns:
        float: 当前 epoch 的平均损失
    """
    epoch_total_loss = 0
    model.train()
    for src, tgt in tqdm(dataloader, desc='训练'):
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        # outputs.shape: [batch_size, tgt_seq_len, tgt_vocab_size]
        outputs = model(src, tgt, teacher_forcing_ratio=tf_ratio)

        # 计算损失时跳过第一个 token(<sos>), 将输出展平为 [batch_size * tgt_seq_len, vocab_size]
        output_flat = outputs[:, 1:, :].reshape(-1, model.tgt_vocab_size)
        tgt_flat = tgt[:, 1:].reshape(-1)

        loss = loss_function(output_flat, tgt_flat)
        loss.backward()
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_total_loss += loss.item()
    return epoch_total_loss / len(dataloader)


def train():
    device = DEVICE_TYPE
    print(f"使用设备: {device}")

    dataloader = get_dataloader(is_train=True)
    # 加载双语词表获取词表大小
    src_tokenizer = ChineseTokenizer.from_vocab(PROCESSED_DIR / 'src_vocab.txt')
    tgt_tokenizer = EnglishTokenizer.from_vocab(PROCESSED_DIR / 'tgt_vocab.txt')
    print(f"中文词表大小: {src_tokenizer.vocab_size}")
    print(f"英文词表大小: {tgt_tokenizer.vocab_size}")

    model = Seq2Seq(src_vocab_size=src_tokenizer.vocab_size,
                    tgt_vocab_size=tgt_tokenizer.vocab_size).to(device)
    print(f"可训练参数量: {count_parameters(model)}")

    # 损失函数: 交叉熵, 忽略 <pad> token (index=0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_LEARNING_RATE)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    # 训练模型
    best_loss = float('inf')
    train_start = time.time()
    for epoch in range(EPOCHS):
        tf_ratio = get_teacher_forcing_ratio(epoch, EPOCHS)
        print(f'========== Epoch {epoch} (TF ratio: {tf_ratio:.2f}) ==========')
        epoch_start = time.time()
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, device, tf_ratio)
        epoch_cost = time.time() - epoch_start
        print(f'Loss: {avg_loss:.4f}  本轮耗时: {epoch_cost:.2f}s')

        # 记录训练结果
        writer.add_scalar('Loss', avg_loss, epoch)
        writer.add_scalar('TeacherForcingRatio', tf_ratio, epoch)
        writer.add_scalar('EpochTime', epoch_cost, epoch)

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODELS_DIR / 'model.pt')
            print("保存模型成功")
        else:
            print("无需保存模型")

    total_cost = time.time() - train_start
    print(f'========== 训练完成 ==========')
    print(f'最优训练损失: {best_loss:.4f}  总耗时: {total_cost:.2f}s')
    writer.add_scalar('TotalTrainTime', total_cost, 0)
    writer.close()


if __name__ == '__main__':
    train()
