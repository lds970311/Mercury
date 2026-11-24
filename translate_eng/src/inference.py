"""编码器/解码器共用的贪心解码与束搜索。"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from config import BEAM_SIZE, LENGTH_PENALTY, MAX_TGT_SEQ_LEN
from model import Decoder, Encoder


@dataclass
class _Beam:
    tokens: list[int]
    score: float
    hidden: torch.Tensor


def _validate_source(src: torch.Tensor) -> None:
    if src.ndim != 2 or src.size(0) != 1:
        raise ValueError("当前解码函数要求 src 形状为 [1, src_len]")


def _normalized_score(beam: _Beam, length_penalty: float) -> float:
    return beam.score / max(len(beam.tokens), 1) ** length_penalty


@torch.no_grad()
def greedy_decode(
        encoder: Encoder,
        decoder: Decoder,
        src: torch.Tensor,
        sos_index: int,
        eos_index: int,
        max_len: int = MAX_TGT_SEQ_LEN,
) -> list[int]:
    """对单条源序列进行贪心解码，结果不含 <sos>/<eos>。"""
    _validate_source(src)
    encoder.eval()
    decoder.eval()

    _, hidden = encoder(src)
    input_token = torch.tensor([[sos_index]], dtype=torch.long, device=src.device)
    predictions = []

    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        input_token = logits.argmax(dim=-1, keepdim=True)
        token_index = input_token.item()
        if token_index == eos_index:
            break
        predictions.append(token_index)

    return predictions


@torch.no_grad()
def beam_search_decode(
        encoder: Encoder,
        decoder: Decoder,
        src: torch.Tensor,
        sos_index: int,
        eos_index: int,
        beam_size: int = BEAM_SIZE,
        max_len: int = MAX_TGT_SEQ_LEN,
        length_penalty: float = LENGTH_PENALTY,
) -> list[int]:
    """逐候选扩展并保留最高分 beam，使用长度归一化缓解短句偏好。"""
    _validate_source(src)
    if beam_size < 1:
        raise ValueError("beam_size 必须大于等于 1")
    if beam_size == 1:
        return greedy_decode(
            encoder, decoder, src, sos_index, eos_index, max_len=max_len
        )

    encoder.eval()
    decoder.eval()
    _, initial_hidden = encoder(src)
    beams = [_Beam(tokens=[], score=0.0, hidden=initial_hidden)]
    completed: list[_Beam] = []

    for _ in range(max_len):
        candidates: list[_Beam] = []
        for beam in beams:
            previous = beam.tokens[-1] if beam.tokens else sos_index
            input_token = torch.tensor(
                [[previous]], dtype=torch.long, device=src.device
            )
            logits, hidden = decoder(input_token, beam.hidden)
            log_probs = F.log_softmax(logits.squeeze(0), dim=-1)
            top_scores, top_indexes = log_probs.topk(beam_size)

            for score, token_index in zip(
                    top_scores.tolist(), top_indexes.tolist(), strict=True
            ):
                candidate = _Beam(
                    tokens=beam.tokens + [token_index],
                    score=beam.score + score,
                    hidden=hidden,
                )
                if token_index == eos_index:
                    completed.append(candidate)
                else:
                    candidates.append(candidate)

        if not candidates:
            break
        candidates.sort(
            key=lambda beam: _normalized_score(beam, length_penalty), reverse=True
        )
        beams = candidates[:beam_size]

    pool = completed or beams
    if not pool:
        return []
    best = max(pool, key=lambda beam: _normalized_score(beam, length_penalty))
    if best.tokens and best.tokens[-1] == eos_index:
        return best.tokens[:-1]
    return best.tokens
