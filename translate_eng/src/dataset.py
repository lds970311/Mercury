"""JSONL 翻译数据集与 DataLoader 构建。"""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from config import BATCH_SIZE, PROCESSED_DIR


class TranslationDataset(Dataset):
    """每条样本返回定长的中文源序列和英文目标序列。"""

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"数据集不存在，请先运行预处理：{path}")

        with path.open("r", encoding="utf-8") as file:
            self.samples = [json.loads(line) for line in file if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        return (
            torch.tensor(sample["src"], dtype=torch.long),
            torch.tensor(sample["tgt"], dtype=torch.long),
        )


def get_dataloader(split: str = "train", batch_size: int = BATCH_SIZE) -> DataLoader:
    if split not in {"train", "test"}:
        raise ValueError("split 仅支持 'train' 或 'test'")

    dataset = TranslationDataset(PROCESSED_DIR / f"indexed_{split}_dataset.jsonl")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        pin_memory=torch.cuda.is_available(),
    )
