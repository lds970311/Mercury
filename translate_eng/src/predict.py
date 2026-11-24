"""交互式或单句中译英推理。"""
import argparse
import sys

import torch

from config import (
    BEAM_SIZE,
    CHECKPOINT_PATH,
    DEVICE,
    MAX_SRC_SEQ_LEN,
    SRC_VOCAB_PATH,
    TGT_VOCAB_PATH,
)
from inference import beam_search_decode, greedy_decode
from model import Decoder, Encoder, load_checkpoint
from tokenizer import ChineseTokenizer, EnglishTokenizer


def configure_console() -> None:
    """尽量将 Windows 控制台切换为 UTF-8，兼容不支持该操作的 IDE。"""
    try:
        for stream in (sys.stdin, sys.stdout):
            if stream.encoding and stream.encoding.lower() != "utf-8":
                stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def translate(
        sentence: str,
        encoder: Encoder,
        decoder: Decoder,
        src_tokenizer: ChineseTokenizer,
        tgt_tokenizer: EnglishTokenizer,
        beam_size: int = BEAM_SIZE,
) -> str:
    indexes = src_tokenizer.encode(sentence, MAX_SRC_SEQ_LEN)
    src = torch.tensor([indexes], dtype=torch.long, device=DEVICE)
    common = {
        "sos_index": tgt_tokenizer.sos_token_index,
        "eos_index": tgt_tokenizer.eos_token_index,
    }
    if beam_size > 1:
        predictions = beam_search_decode(
            encoder, decoder, src, beam_size=beam_size, **common
        )
    else:
        predictions = greedy_decode(encoder, decoder, src, **common)
    return tgt_tokenizer.decode(predictions)


def predict(sentence: str | None = None, beam_size: int = BEAM_SIZE) -> None:
    configure_console()
    src_tokenizer = ChineseTokenizer.from_vocab(SRC_VOCAB_PATH)
    tgt_tokenizer = EnglishTokenizer.from_vocab(TGT_VOCAB_PATH)
    encoder, decoder = load_checkpoint(CHECKPOINT_PATH, DEVICE)
    print(f"模型加载完成，设备：{DEVICE}")

    if sentence is not None:
        print(
            translate(
                sentence,
                encoder,
                decoder,
                src_tokenizer,
                tgt_tokenizer,
                beam_size,
            )
        )
        return

    print("输入中文句子进行翻译，输入 quit 退出。")
    while True:
        try:
            sentence = input("请输入中文：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出翻译")
            break
        if sentence.lower() == "quit":
            break
        if not sentence:
            continue
        result = translate(
            sentence,
            encoder,
            decoder,
            src_tokenizer,
            tgt_tokenizer,
            beam_size,
        )
        print(f"翻译结果：{result}")


def main() -> None:
    parser = argparse.ArgumentParser(description="中译英推理")
    parser.add_argument("--sentence", type=str, default=None, help="直接翻译单句")
    parser.add_argument("--beam-size", type=int, default=BEAM_SIZE, help="束宽度")
    args = parser.parse_args()
    predict(sentence=args.sentence, beam_size=args.beam_size)


if __name__ == "__main__":
    main()
