"""项目路径与训练超参数。"""
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE_PATH = ROOT_DIR.parent / "data" / "cmn.txt"

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
LOG_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINT_PATH = MODELS_DIR / "model.pt"
SRC_VOCAB_PATH = PROCESSED_DIR / "src_vocab.txt"
TGT_VOCAB_PATH = PROCESSED_DIR / "tgt_vocab.txt"

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_SRC_SEQ_LEN = 36
MAX_TGT_SEQ_LEN = 36

BATCH_SIZE = 64
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.1

ADAM_LEARNING_RATE = 0.001
EPOCHS = 20
TF_RATIO_START = 0.9
TF_RATIO_END = 0.1
GRAD_CLIP_NORM = 1.0

BEAM_SIZE = 5
LENGTH_PENALTY = 0.6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
