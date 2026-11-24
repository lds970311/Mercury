"""读取 Tatoeba 平行语料并生成词表和定长索引数据集。"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    DATA_FILE_PATH,
    MAX_SRC_SEQ_LEN,
    MAX_TGT_SEQ_LEN,
    PROCESSED_DIR,
    RANDOM_STATE,
    SRC_VOCAB_PATH,
    TEST_SIZE,
    TGT_VOCAB_PATH,
)
from tokenizer import ChineseTokenizer, EnglishTokenizer

TRAIN_DATASET_PATH = PROCESSED_DIR / "indexed_train_dataset.jsonl"
TEST_DATASET_PATH = PROCESSED_DIR / "indexed_test_dataset.jsonl"


def load_parallel_corpus() -> pd.DataFrame:
    """读取 Tab 分隔的英文、中文两列并清理无效行。"""
    if not DATA_FILE_PATH.is_file():
        raise FileNotFoundError(f"原始语料不存在：{DATA_FILE_PATH}")

    dataframe = pd.read_csv(
        DATA_FILE_PATH,
        sep="\t",
        header=None,
        encoding="utf-8",
        usecols=[0, 1],
        names=["eng", "chn"],
    ).dropna()
    dataframe = dataframe[
        dataframe["eng"].str.strip().ne("")
        & dataframe["chn"].str.strip().ne("")
        ].reset_index(drop=True)
    print(f"有效平行语料数：{len(dataframe)}")
    return dataframe


def split_dataset(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        dataframe, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def construct_dataset(
        dataframe: pd.DataFrame,
        src_tokenizer: ChineseTokenizer,
        tgt_tokenizer: EnglishTokenizer,
) -> list[dict[str, list[int]]]:
    samples = []
    pairs = zip(dataframe["chn"], dataframe["eng"], strict=True)
    for chinese, english in tqdm(pairs, desc="索引化", total=len(dataframe)):
        samples.append(
            {
                "src": src_tokenizer.encode(chinese, MAX_SRC_SEQ_LEN),
                "tgt": tgt_tokenizer.encode(
                    english, MAX_TGT_SEQ_LEN, add_sos_eos=True
                ),
            }
        )
    return samples


def save_dataset(samples: list[dict[str, list[int]]], path: Path) -> None:
    pd.DataFrame(samples).to_json(
        path, orient="records", lines=True, force_ascii=False
    )


def process(force: bool = False) -> None:
    """所有产物齐全时直接复用；否则统一重建，避免词表和数据不匹配。"""
    artifacts = (
        SRC_VOCAB_PATH,
        TGT_VOCAB_PATH,
        TRAIN_DATASET_PATH,
        TEST_DATASET_PATH,
    )
    if not force and all(path.is_file() for path in artifacts):
        print("预处理产物已存在，直接复用；如需重建请添加 --force。")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataframe = load_parallel_corpus()
    train_df, test_df = split_dataset(dataframe)

    ChineseTokenizer.build_vocab(train_df["chn"].tolist(), SRC_VOCAB_PATH)
    EnglishTokenizer.build_vocab(train_df["eng"].tolist(), TGT_VOCAB_PATH)
    src_tokenizer = ChineseTokenizer.from_vocab(SRC_VOCAB_PATH)
    tgt_tokenizer = EnglishTokenizer.from_vocab(TGT_VOCAB_PATH)
    print(f"中文词表大小：{src_tokenizer.vocab_size}")
    print(f"英文词表大小：{tgt_tokenizer.vocab_size}")

    save_dataset(
        construct_dataset(train_df, src_tokenizer, tgt_tokenizer),
        TRAIN_DATASET_PATH,
    )
    save_dataset(
        construct_dataset(test_df, src_tokenizer, tgt_tokenizer),
        TEST_DATASET_PATH,
    )
    print(f"预处理完成，产物目录：{PROCESSED_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建中译英训练数据")
    parser.add_argument("--force", action="store_true", help="强制重建全部产物")
    args = parser.parse_args()
    process(force=args.force)


if __name__ == "__main__":
    main()
