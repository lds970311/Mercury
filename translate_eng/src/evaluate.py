# 模型评估脚本(BLEU 评分)
import torch

from config import MODELS_DIR, DEVICE_TYPE, PROCESSED_DIR, MAX_TGT_SEQ_LEN
from dataset import get_dataloader
from model import Seq2Seq
from tokenizer import ChineseTokenizer, EnglishTokenizer


def translate_sentence(model, src, tgt_tokenizer, device, max_len=MAX_TGT_SEQ_LEN):
    """
    对单条源序列进行贪心解码翻译, 返回预测的目标索引序列。

    Args:
        model: Seq2Seq 模型(已加载权重, eval 模式)
        src: 源序列张量 [1, src_seq_len]
        tgt_tokenizer: 英文分词器(用于获取 <sos>/<eos> 索引)
        device: 运行设备
        max_len: 最大生成长度

    Returns:
        list: 预测的目标索引序列(不含 <sos>)
    """
    model.eval()
    with torch.no_grad():
        _, hidden = model.encoder(src)
        input_token = torch.tensor([[tgt_tokenizer.sos_token_index]], dtype=torch.long).to(device)
        predictions = []

        for _ in range(max_len):
            prediction, hidden = model.decoder(input_token, hidden)
            top1 = prediction.argmax(dim=-1)  # [1, 1]
            predictions.append(top1.item())

            if top1.item() == tgt_tokenizer.eos_token_index:
                break
            input_token = top1

    return predictions


def compute_bleu(reference_tokens, hypothesis_tokens):
    """
    计算单句 BLEU-4 分数(简化版, 不含 brevity penalty 的多句平滑)。

    Args:
        reference_tokens: 参考译文的 token 列表
        hypothesis_tokens: 模型译文的 token 列表

    Returns:
        float: BLEU 分数
    """
    import math
    from collections import Counter

    if len(hypothesis_tokens) == 0:
        return 0.0

    # 计算 1-gram 到 4-gram 的精度
    precisions = []
    for n in range(1, 5):
        ref_ngrams = Counter(
            tuple(reference_tokens[i:i + n]) for i in range(len(reference_tokens) - n + 1)
        )
        hyp_ngrams = Counter(
            tuple(hypothesis_tokens[i:i + n]) for i in range(len(hypothesis_tokens) - n + 1)
        )
        clipped = sum(min(count, ref_ngrams[ngram]) for ngram, count in hyp_ngrams.items())
        total = max(sum(hyp_ngrams.values()), 1)
        precisions.append(clipped / total)

    # 如果任一 n-gram 精度为 0, BLEU 为 0
    if any(p == 0 for p in precisions):
        return 0.0

    # 几何平均
    log_avg = sum(math.log(p) for p in precisions) / 4

    # Brevity Penalty
    bp = 1.0
    if len(hypothesis_tokens) < len(reference_tokens):
        bp = math.exp(1 - len(reference_tokens) / len(hypothesis_tokens))

    return bp * math.exp(log_avg)


def evaluate():
    device = DEVICE_TYPE
    src_tokenizer = ChineseTokenizer.from_vocab(PROCESSED_DIR / 'src_vocab.txt')
    tgt_tokenizer = EnglishTokenizer.from_vocab(PROCESSED_DIR / 'tgt_vocab.txt')

    model = Seq2Seq(src_vocab_size=src_tokenizer.vocab_size,
                    tgt_vocab_size=tgt_tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / 'model.pt', map_location=device))
    model.eval()

    data_loader = get_dataloader(is_train=False)

    total_bleu = 0.0
    sample_count = 0

    for src_batch, tgt_batch in data_loader:
        for i in range(src_batch.size(0)):
            src = src_batch[i:i + 1].to(device)
            # 获取参考译文 token(跳过 <sos> 和 <eos>)
            ref_indexes = tgt_batch[i].tolist()
            ref_tokens = tgt_tokenizer.to_tokens(ref_indexes)

            # 模型翻译
            pred_indexes = translate_sentence(model, src, tgt_tokenizer, device)
            pred_tokens = tgt_tokenizer.to_tokens(pred_indexes)

            bleu = compute_bleu(ref_tokens, pred_tokens)
            total_bleu += bleu
            sample_count += 1

    avg_bleu = total_bleu / sample_count if sample_count > 0 else 0.0
    print(f'========== 评估结果(Seq2Seq GRU) ==========')
    print(f'测试集样本数: {sample_count}')
    print(f'平均 BLEU-4 分数: {avg_bleu:.4f}')
    print('=============================================')


if __name__ == '__main__':
    evaluate()
