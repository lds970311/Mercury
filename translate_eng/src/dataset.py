# 自定义DataSet
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import PROCESSED_DIR, BATCH_SIZE


class TranslationDataset(Dataset):
    """封装预处理后的 jsonl 平行语料数据集, 按索引返回 (源序列, 目标序列) 张量"""

    def __init__(self, path):
        self.data = pd.read_json(path, lines=True).to_dict("records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_tensor = torch.tensor(self.data[idx]["src"], dtype=torch.long)
        tgt_tensor = torch.tensor(self.data[idx]["tgt"], dtype=torch.long)
        return src_tensor, tgt_tensor


def get_dataloader(is_train=True):
    """按 is_train 加载训练集或测试集的 DataLoader"""
    if is_train:
        path = PROCESSED_DIR / "indexed_train_dataset.jsonl"
    else:
        path = PROCESSED_DIR / "indexed_test_dataset.jsonl"
    dataset = TranslationDataset(path)

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=is_train)


if __name__ == '__main__':
    dataloader = get_dataloader(is_train=True)
    test_data_loader = get_dataloader(is_train=False)
    print(f"训练集 batch 数: {len(dataloader)}")
    print(f"测试集 batch 数: {len(test_data_loader)}")
