# 中译英翻译模型 (Seq2Seq GRU)

## 1. 需求说明

本项目基于 **Seq2Seq (Encoder-Decoder)** 架构实现中文到英文的机器翻译。
使用 GRU 作为循环单元，编码器将中文源句编码为上下文向量，解码器逐步生成英文目标句。

数据集来源为 Tatoeba 英中平行语料（`data/05_nlp/cmn.txt`），格式为 Tab 分隔：

| 英文 (eng)   | 中文 (chn) | 版权信息     |
|:-----------|:---------|:---------|
| Hi.        | 嗨。       | CC-BY... |
| I know.    | 我知道。     | CC-BY... |
| Thank you. | 谢谢。      | CC-BY... |

本案例选取前两列（英文与中文）构造翻译对，方向为 **中文 → 英文**。

## 2. 模型结构设计

模型整体由 Encoder 和 Decoder 两部分组成：

### 编码器 (Encoder)

- **Embedding**: 将中文词索引映射为稠密向量
- **GRU**: 多层 GRU 建模源序列上下文，输出最终隐藏状态作为语义表示

### 解码器 (Decoder)

- **Embedding**: 将已生成的英文词索引映射为稠密向量
- **GRU**: 接收编码器隐藏状态，逐步生成目标序列
- **Linear**: 将 GRU 输出映射为目标词表大小的 logits

### 训练策略

- 使用 **Teacher Forcing**（概率 0.5）加速训练收敛
- 推理时使用 **贪心解码**（Greedy Decoding）

## 3. 训练方案

* **损失函数：** CrossEntropyLoss（忽略 `<pad>` token）
* **优化器：** Adam（lr=0.001）
* **梯度裁剪：** max_norm=1.0，防止梯度爆炸

## 4. 评估方案

模型训练完毕后，使用测试集计算平均 **BLEU-4** 分数评估翻译质量。

## 5. 目录结构与运行

沿用 smart_chat 子项目标准结构：`data/processed`（预处理数据）、`logs`（训练日志）、
`models`（模型权重）、`src`（源码）。

```bash
# 安装依赖
pip install -r translate_eng/requirements.txt

# 1) 数据预处理
python translate_eng/src/process.py

# 2) 训练
python translate_eng/src/train.py

# 3) 评估
python translate_eng/src/evaluate.py

# 4) 预测
python translate_eng/src/predict.py
```

> **注意**: 如果上面运行失败，可进入 `translate_eng/src` 目录后运行对应 .py 文件。
