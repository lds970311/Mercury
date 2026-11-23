# 超参数配置
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).parent.parent

# 原始数据集路径(Tatoeba 英中平行语料, Tab 分隔: 英文\t中文\t版权信息)
DATA_FILE_PATH = ROOT_DIR.parent / "data" / "cmn.txt"

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
LOG_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"

# 数据集划分比例(测试集占比)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 源语言(中文)与目标语言(英文)定长序列长度
MAX_SRC_SEQ_LEN = 36
MAX_TGT_SEQ_LEN = 36

BATCH_SIZE = 64
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.1

# 学习率与训练轮数
ADAM_LEARNING_RATE = 0.001
EPOCHS = 20

DEVICE_TYPE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
