# 中译英翻译项目（GRU Encoder–Decoder）

基于双向 GRU 编码器和单向 GRU 解码器实现中文到英文翻译。两个网络独立定义，训练、评估和预测直接操作 `Encoder` / `Decoder`
，不再使用额外的模型封装类。

## 数据

原始语料为仓库根目录的 `data/cmn.txt`，每行以 Tab 分隔英文、中文和版权信息。项目只读取前两列，方向为中文 → 英文。

- 中文：按非空白字符切分。
- 英文：使用 NLTK Treebank 分词并转为小写。
- 词表仅使用训练集构建，特殊 token 顺序固定为 `<pad>`、`<unk>`、`<sos>`、`<eos>`。

## 项目结构

```text
translate_eng/
├── data/processed/       # 词表与索引化 JSONL 数据
├── logs/                 # TensorBoard 日志
├── models/model.pt       # 最优检查点
└── src/
    ├── config.py         # 路径与超参数
    ├── tokenizer.py      # 通用分词器及中英文实现
    ├── process.py        # 数据预处理
    ├── dataset.py        # Dataset / DataLoader
    ├── model.py          # Encoder、Decoder、检查点读写
    ├── inference.py      # 贪心解码与束搜索
    ├── train.py          # 训练入口
    ├── evaluate.py       # BLEU-4 评估
    └── predict.py        # 单句或交互式翻译
```

## 安装

在**仓库根目录**执行：

```bash
pip install -r translate_eng/requirements.txt
```

如需 NVIDIA GPU 训练，请按本机 CUDA 环境安装对应的 PyTorch wheel。代码会自动选择 CUDA；不可用时回退到 CPU。

## 运行

所有命令均在**仓库根目录**执行。

```bash
# 预处理：产物齐全时直接复用
python -m translate_eng.src.process

# 强制统一重建词表和数据集
python -m translate_eng.src.process --force

# 训练（默认轮数见 config.py）
python -m translate_eng.src.train

# 自定义训练轮数
python -m translate_eng.src.train --epochs 20

# 贪心解码评估；--limit 可用于快速检查
python -m translate_eng.src.evaluate --limit 100

# 束搜索评估
python -m translate_eng.src.evaluate --beam-size 5

# 直接翻译一句话
python -m translate_eng.src.predict --sentence "今天天气很好"

# 交互式翻译
python -m translate_eng.src.predict --beam-size 5

# 查看训练曲线
tensorboard --logdir translate_eng/logs
```

## 训练与检查点

- 损失函数为交叉熵，忽略目标词表中的 `<pad>`。
- 优化器为 Adam，梯度裁剪阈值和 teacher forcing 衰减范围集中在 `config.py`。
- 每轮根据训练损失保存最优检查点，其中包含编码器/解码器权重、模型结构参数和训练元数据。

> 重构后的检查点包含两套独立权重，旧版 `model.pt` 不能直接加载。首次使用新代码时需要重新训练；已有预处理数据可以直接复用。
