# 数据预处理脚本
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import DATA_FILE_PATH, MAX_SRC_SEQ_LEN, MAX_TGT_SEQ_LEN, PROCESSED_DIR, RANDOM_STATE, TEST_SIZE
from tokenizer import ChineseTokenizer, EnglishTokenizer


def load_parallel_corpus():
    """读取 Tatoeba 英中平行语料(Tab 分隔: 英文\t中文\t版权信息), 仅保留前两列"""
    dataframe = pd.read_csv(DATA_FILE_PATH, sep='\t', header=None, encoding='utf-8',
                            usecols=[0, 1], names=['eng', 'chn'])
    # 过滤空值并重置索引
    dataframe = dataframe.dropna().reset_index(drop=True)
    dataframe = dataframe[dataframe['eng'].str.strip().ne('') & dataframe['chn'].str.strip().ne('')]
    print(f"平行语料总数: {len(dataframe)}")
    return dataframe


def split_dataset(dataframe):
    """按 TEST_SIZE 比例随机划分训练集与测试集"""
    train_df, test_df = train_test_split(dataframe, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def construct_dataset(dataframe, src_tokenizer, tgt_tokenizer):
    """将平行语料编码为定长索引序列对, 生成 {'src': [...], 'tgt': [...]} 样本列表"""
    samples = []
    for chn, eng in tqdm(zip(dataframe['chn'], dataframe['eng']), desc='索引化', total=len(dataframe)):
        samples.append({
            'src': src_tokenizer.encode(chn, MAX_SRC_SEQ_LEN),
            'tgt': tgt_tokenizer.encode(eng, MAX_TGT_SEQ_LEN)
        })
    return samples


def save_dataset(samples, split_type):
    """将样本列表以 jsonl 格式保存到 PROCESSED_DIR 目录下"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(samples).to_json(PROCESSED_DIR / f'indexed_{split_type}_dataset.jsonl',
                                  orient='records', lines=True)


def process():
    """数据预处理主流程: 读取语料 -> 划分数据集 -> 构建双语词表 -> 定长编码 -> 保存"""
    print('数据预处理中...')
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataframe = load_parallel_corpus()
    train_df, test_df = split_dataset(dataframe)

    src_vocab_file = PROCESSED_DIR / 'src_vocab.txt'
    tgt_vocab_file = PROCESSED_DIR / 'tgt_vocab.txt'

    # 如果词表不存在则仅用训练集构建
    if not src_vocab_file.exists():
        ChineseTokenizer.build_vocab(train_df['chn'].tolist(), src_vocab_file)
    if not tgt_vocab_file.exists():
        EnglishTokenizer.build_vocab(train_df['eng'].tolist(), tgt_vocab_file)

    src_tokenizer = ChineseTokenizer.from_vocab(src_vocab_file)
    tgt_tokenizer = EnglishTokenizer.from_vocab(tgt_vocab_file)
    print(f"中文词表大小: {src_tokenizer.vocab_size}")
    print(f"英文词表大小: {tgt_tokenizer.vocab_size}")

    # 如果文件存在就不执行
    if not (PROCESSED_DIR / 'indexed_train_dataset.jsonl').exists():
        save_dataset(construct_dataset(train_df, src_tokenizer, tgt_tokenizer), split_type='train')

    if not (PROCESSED_DIR / 'indexed_test_dataset.jsonl').exists():
        save_dataset(construct_dataset(test_df, src_tokenizer, tgt_tokenizer), split_type='test')

    print('数据预处理完成！')


if __name__ == '__main__':
    process()
