# 模型推理脚本
import sys

import torch
import torch.nn.functional as F

from config import MODELS_DIR, PROCESSED_DIR, MAX_SRC_SEQ_LEN, MAX_TGT_SEQ_LEN, DEVICE_TYPE
from model import Seq2Seq
from tokenizer import ChineseTokenizer, EnglishTokenizer

# 束搜索宽度
BEAM_SIZE = 5


def beam_search_decode(model, src_tensor, tgt_tokenizer, device, beam_size=BEAM_SIZE, max_len=MAX_TGT_SEQ_LEN):
    """
    束搜索解码: 每步保留 beam_size 个累计得分最高的候选序列,
    相比贪心解码能探索更多可能的翻译路径, 显著提升翻译质量。

    Args:
        model: Seq2Seq 模型(已加载权重, eval 模式)
        src_tensor: 源序列张量 [1, src_seq_len]
        tgt_tokenizer: 英文分词器
        device: 运行设备
        beam_size: 束宽度
        max_len: 最大生成长度

    Returns:
        list: 最优翻译的目标索引序列(不含 <sos>/<eos>)
    """
    with torch.no_grad():
        # 编码源序列, 获取初始隐状态
        _, hidden = model.encoder(src_tensor)
        # hidden.shape: [num_layers, 1, hidden_size]

        # 初始化: beam_size 个相同的起始状态
        # 将 hidden 扩展为 [num_layers, beam_size, hidden_size]
        hidden = hidden.expand(-1, beam_size, -1).contiguous()

        # 每个 beam 的当前输入 token, 初始为 <sos>
        input_tokens = torch.full((beam_size, 1), tgt_tokenizer.sos_token_index,
                                  dtype=torch.long, device=device)
        # 每个 beam 的累计 log 概率得分
        scores = torch.zeros(beam_size, device=device)
        # 每个 beam 已生成的 token 序列
        sequences = [[] for _ in range(beam_size)]
        # 已完成的候选(生成了 <eos> 的序列)
        completed = []
        # 标记哪些 beam 仍活跃
        alive = torch.ones(beam_size, dtype=torch.bool, device=device)

        for step in range(max_len):
            if not alive.any():
                break

            # 解码一步: prediction [beam_size, 1, vocab_size]
            prediction, new_hidden = model.decoder(input_tokens, hidden)
            log_probs = F.log_softmax(prediction.squeeze(1), dim=-1)
            # log_probs.shape: [beam_size, vocab_size]

            # 已结束的 beam 不再参与扩展, 将得分设为 -inf
            log_probs[~alive] = float('-inf')

            # 候选得分 = 当前累计得分 + 新 token 的 log 概率
            # candidate_scores.shape: [beam_size, vocab_size]
            candidate_scores = scores.unsqueeze(1) + log_probs

            # 展平后取 top beam_size 个候选
            flat_scores = candidate_scores.view(-1)
            top_scores, top_indices = flat_scores.topk(beam_size)

            # 解析 top_indices -> (beam_idx, token_idx)
            beam_indices = top_indices // model.tgt_vocab_size
            token_indices = top_indices % model.tgt_vocab_size

            # 更新状态
            new_sequences = []
            new_alive = torch.ones(beam_size, dtype=torch.bool, device=device)
            for i in range(beam_size):
                b_idx = beam_indices[i].item()
                t_idx = token_indices[i].item()
                seq = sequences[b_idx] + [t_idx]

                if t_idx == tgt_tokenizer.eos_token_index:
                    # 该候选已完成, 加入 completed (长度归一化得分)
                    completed.append((top_scores[i].item() / len(seq), seq))
                    new_alive[i] = False
                new_sequences.append(seq)

            sequences = new_sequences
            scores = top_scores
            alive = new_alive

            # 根据选中的 beam 重组隐状态和输入 token
            hidden = new_hidden[:, beam_indices, :]
            input_tokens = token_indices.unsqueeze(1)

        # 选择最优结果: 优先从已完成候选中选, 否则取活跃 beam 中得分最高的
        if completed:
            completed.sort(key=lambda x: x[0], reverse=True)
            best_seq = completed[0][1]
        else:
            best_idx = scores.argmax().item()
            best_seq = sequences[best_idx]

        # 移除末尾的 <eos>(如果存在)
        if best_seq and best_seq[-1] == tgt_tokenizer.eos_token_index:
            best_seq = best_seq[:-1]
        return best_seq


def predict():
    """
    交互式翻译演示: 输入中文句子, 输出英文翻译结果。

    流程:
        1. 加载双语词表, 创建与训练时结构一致的模型并加载已保存的权重;
        2. 输入的中文经 src_tokenizer 编码为定长索引序列;
        3. 使用束搜索(Beam Search)解码生成英文索引序列, 经 tgt_tokenizer 解码为英文句子输出。
    """
    # Windows 终端默认 GBK 编码，强制 UTF-8 以正确接收中文输入
    try:
        if sys.stdin.encoding and sys.stdin.encoding.lower() != 'utf-8':
            sys.stdin.reconfigure(encoding='utf-8')
        if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass  # IDE 交互窗口可能不支持 reconfigure

    device = DEVICE_TYPE
    src_tokenizer = ChineseTokenizer.from_vocab(PROCESSED_DIR / 'src_vocab.txt')
    tgt_tokenizer = EnglishTokenizer.from_vocab(PROCESSED_DIR / 'tgt_vocab.txt')

    # 用与训练相同的超参创建模型实例, 再加载 state_dict 权重
    model = Seq2Seq(src_vocab_size=src_tokenizer.vocab_size,
                    tgt_vocab_size=tgt_tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / 'model.pt', map_location=device))
    model.eval()

    print("模型加载完成, 输入中文句子进行翻译(输入 quit 退出)")
    while True:
        sentence = input("请输入中文: ").strip()
        if sentence.lower() == 'quit':
            print("退出翻译")
            break
        if not sentence:
            continue

        # 编码为定长索引序列
        src_indexes = src_tokenizer.encode(sentence, MAX_SRC_SEQ_LEN)
        src_tensor = torch.tensor([src_indexes], dtype=torch.long).to(device)

        # 束搜索解码
        predictions = beam_search_decode(model, src_tensor, tgt_tokenizer, device)

        translation = tgt_tokenizer.decode(predictions)
        print(f"翻译结果: {translation}")


if __name__ == '__main__':
    predict()
