"""在测试集上计算平均句级 BLEU-4。"""
import argparse
import math
from collections import Counter

from tqdm import tqdm

from .config import CHECKPOINT_PATH, DEVICE, TGT_VOCAB_PATH
from .dataset import get_dataloader
from .inference import beam_search_decode, greedy_decode
from .model import load_checkpoint
from .tokenizer import EnglishTokenizer


def compute_bleu(reference_tokens: list[str], hypothesis_tokens: list[str]) -> float:
    if not hypothesis_tokens:
        return 0.0

    precisions = []
    for n in range(1, 5):
        reference_ngrams = Counter(
            tuple(reference_tokens[i: i + n])
            for i in range(len(reference_tokens) - n + 1)
        )
        hypothesis_ngrams = Counter(
            tuple(hypothesis_tokens[i: i + n])
            for i in range(len(hypothesis_tokens) - n + 1)
        )
        clipped = sum(
            min(count, reference_ngrams[ngram])
            for ngram, count in hypothesis_ngrams.items()
        )
        total = max(sum(hypothesis_ngrams.values()), 1)
        precisions.append(clipped / total)

    if any(precision == 0 for precision in precisions):
        return 0.0

    log_average = sum(math.log(precision) for precision in precisions) / 4
    brevity_penalty = 1.0
    if len(hypothesis_tokens) < len(reference_tokens):
        brevity_penalty = math.exp(
            1 - len(reference_tokens) / len(hypothesis_tokens)
        )
    return brevity_penalty * math.exp(log_average)


def evaluate(limit: int | None = None, beam_size: int = 1) -> float:
    if limit is not None and limit < 1:
        raise ValueError("limit 必须大于等于 1")

    tgt_tokenizer = EnglishTokenizer.from_vocab(TGT_VOCAB_PATH)
    encoder, decoder = load_checkpoint(CHECKPOINT_PATH, DEVICE)
    dataloader = get_dataloader("test")

    total_bleu = 0.0
    sample_count = 0
    stop = False
    for src_batch, tgt_batch in tqdm(dataloader, desc="评估"):
        for index in range(src_batch.size(0)):
            src = src_batch[index: index + 1].to(DEVICE)
            reference_tokens = tgt_tokenizer.to_tokens(tgt_batch[index].tolist())

            common = {
                "sos_index": tgt_tokenizer.sos_token_index,
                "eos_index": tgt_tokenizer.eos_token_index,
            }
            if beam_size > 1:
                predicted_indexes = beam_search_decode(
                    encoder, decoder, src, beam_size=beam_size, **common
                )
            else:
                predicted_indexes = greedy_decode(encoder, decoder, src, **common)

            predicted_tokens = tgt_tokenizer.to_tokens(predicted_indexes)
            total_bleu += compute_bleu(reference_tokens, predicted_tokens)
            sample_count += 1
            if limit is not None and sample_count >= limit:
                stop = True
                break
        if stop:
            break

    if sample_count == 0:
        raise RuntimeError("测试集为空，无法评估")

    average_bleu = total_bleu / sample_count
    print("========== 评估结果 ==========")
    print(f"设备：{DEVICE}")
    print(f"解码方式：{'束搜索' if beam_size > 1 else '贪心'}")
    print(f"测试样本数：{sample_count}")
    print(f"平均 BLEU-4：{average_bleu:.4f}")
    return average_bleu


def main() -> None:
    parser = argparse.ArgumentParser(description="评估中译英模型")
    parser.add_argument("--limit", type=int, default=None, help="仅评估前 N 条样本")
    parser.add_argument("--beam-size", type=int, default=1, help="束宽度；1 为贪心")
    args = parser.parse_args()
    evaluate(limit=args.limit, beam_size=args.beam_size)


if __name__ == "__main__":
    main()
