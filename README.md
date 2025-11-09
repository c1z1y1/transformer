# Transformer 从零实现

本项目实现了完整的Transformer模型，包括Encoder和Decoder，支持语言模型任务。

## 项目结构

```
.
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── attention.py        # Multi-head Attention实现
│   ├── positional_encoding.py  # 位置编码实现
│   ├── transformer.py      # Transformer模型实现
│   ├── data_loader.py      # 数据加载模块
│   └── train.py            # 训练脚本
├── scripts/                # 运行脚本
│   ├── run.sh              # Linux/Mac运行脚本
│   └── run.bat             # Windows运行脚本
├── docs/                   # 文档目录
│   └── report.tex          # 技术报告（LaTeX）
├── results/                # 结果目录
│   └── (训练曲线和结果表格)
├── checkpoints/            # 模型检查点
├── logs/                   # 训练日志
├── data/                   # 数据集目录
├── requirements.txt        # Python依赖
└── README.md              # 本文件
```

## 环境要求

- Python >= 3.7
- PyTorch >= 1.12.0
- CUDA (可选，用于GPU加速)

## 安装

1. 克隆仓库（或下载代码）

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 快速开始

### 使用脚本运行（推荐）

**Linux/Mac:**
```bash
chmod +x scripts/run.sh
./scripts/run.sh
```

**Windows:**
```cmd
scripts\run.bat
```

### 手动运行

```bash
python -m src.train \
    --dataset wikitext2 \
    --data_dir ./data \
    --seq_len 128 \
    --batch_size 32 \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 6 \
    --d_ff 2048 \
    --dropout 0.1 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --num_epochs 10 \
    --max_grad_norm 1.0 \
    --scheduler cosine \
    --seed 42 \
    --save_dir ./checkpoints \
    --log_dir ./logs
```

## 参数说明

### 数据集参数
- `--dataset`: 数据集名称 (`wikitext2` 或 `tinyshakespeare`)
- `--data_dir`: 数据目录路径
- `--seq_len`: 序列长度（默认：128）
- `--batch_size`: 批次大小（默认：32）

### 模型参数
- `--d_model`: 模型维度（默认：512）
- `--num_heads`: 注意力头数（默认：8）
- `--num_layers`: Encoder层数（默认：6）
- `--d_ff`: 前馈网络维度（默认：2048）
- `--dropout`: Dropout比率（默认：0.1）
- `--use_decoder`: 使用Decoder（默认：False，Encoder-only模式）

### 训练参数
- `--lr`: 学习率（默认：1e-4）
- `--weight_decay`: 权重衰减（默认：0.01）
- `--num_epochs`: 训练轮数（默认：10）
- `--max_grad_norm`: 梯度裁剪阈值（默认：1.0）
- `--scheduler`: 学习率调度器类型 (`cosine`, `step`, `warmup`)
- `--seed`: 随机种子（默认：42）

### 目录参数
- `--save_dir`: 模型保存目录（默认：./checkpoints）
- `--log_dir`: 日志目录（默认：./logs）

## 数据集

### WikiText-2
- 自动下载，无需手动准备
- 下载地址：https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip

### Tiny Shakespeare
- 自动下载，无需手动准备
- 下载地址：https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

## 硬件要求

### 最低要求
- CPU: 4核心
- 内存: 8GB
- 存储: 2GB可用空间

### 推荐配置
- GPU: NVIDIA GPU with 4GB+ VRAM (CUDA支持)
- CPU: 8核心+
- 内存: 16GB+
- 存储: 5GB+可用空间

## 重现实验

### 基础实验（Encoder-only）

```bash
python -m src.train \
    --dataset wikitext2 \
    --seq_len 128 \
    --batch_size 32 \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 6 \
    --d_ff 2048 \
    --dropout 0.1 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --num_epochs 10 \
    --max_grad_norm 1.0 \
    --scheduler cosine \
    --seed 42
```

### 完整实验（Encoder+Decoder）

```bash
python -m src.train \
    --dataset wikitext2 \
    --seq_len 128 \
    --batch_size 32 \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 6 \
    --d_ff 2048 \
    --dropout 0.1 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --num_epochs 10 \
    --max_grad_norm 1.0 \
    --scheduler cosine \
    --seed 42 \
    --use_decoder
```

### Tiny Shakespeare实验

```bash
python -m src.train \
    --dataset tinyshakespeare \
    --seq_len 128 \
    --batch_size 32 \
    --d_model 256 \
    --num_heads 4 \
    --num_layers 4 \
    --d_ff 1024 \
    --dropout 0.1 \
    --lr 1e-3 \
    --num_epochs 20 \
    --seed 42
```

## 功能特性

### 已实现
- ✅ Multi-head Self-Attention机制
- ✅ Position-wise Feed-Forward Network
- ✅ 残差连接 + Layer Normalization
- ✅ 正弦位置编码
- ✅ Encoder模块
- ✅ Decoder模块（可选）
- ✅ 学习率调度（Cosine, Step, Warmup）
- ✅ 梯度裁剪
- ✅ AdamW优化器
- ✅ 模型保存/加载
- ✅ 训练曲线可视化
- ✅ 支持WikiText-2和Tiny Shakespeare数据集

### 可选扩展
- 相对位置编码
- 稀疏注意力
- 线性注意力
- 超参数敏感性分析

## 结果

训练完成后，结果将保存在以下位置：
- 模型检查点：`./checkpoints/`
- 训练日志：`./logs/training_history.json`
- 训练曲线：`./logs/training_curves.png`

## 技术报告

详细的技术报告请参见 `docs/report.tex`（LaTeX格式）。

报告包含：
1. 引言：Transformer背景、动机和目标
2. 相关工作：Transformer模型和注意力机制综述
3. 模型架构与数学推导：详细的公式推导和理论基础
4. 实现细节：框架选择、关键代码实现、超参数设置
5. 实验设置：数据集选择、训练配置、评估指标
6. 结果分析：训练曲线、预测示例、消融实验
7. 可重现性：代码结构、运行说明、环境要求
8. 结论与未来工作

## 代码说明

### 核心模块

1. **attention.py**: 实现Multi-head Attention机制
   - `MultiHeadAttention`: 多头注意力
   - `PositionwiseFeedForward`: 前馈网络

2. **positional_encoding.py**: 位置编码
   - `SinusoidalPositionalEncoding`: 正弦位置编码
   - `LearnablePositionalEncoding`: 可学习位置编码

3. **transformer.py**: Transformer模型
   - `EncoderLayer`: Encoder层
   - `DecoderLayer`: Decoder层
   - `TransformerEncoder`: Encoder堆叠
   - `TransformerDecoder`: Decoder堆叠
   - `Transformer`: 完整模型

4. **data_loader.py**: 数据加载
   - `Vocabulary`: 词汇表
   - `TextDataset`: 文本数据集
   - `prepare_data`: 数据准备函数

5. **train.py**: 训练脚本
   - `Trainer`: 训练器类
   - 支持学习率调度、梯度裁剪等

## 常见问题

### Q: 训练时内存不足？
A: 减小`batch_size`或`seq_len`，或使用更小的模型（减小`d_model`、`num_layers`等）

### Q: 如何使用GPU？
A: 如果安装了CUDA版本的PyTorch，代码会自动使用GPU

### Q: 如何继续训练？
A: 使用`trainer.load_model()`加载检查点，然后继续训练

### Q: 如何生成文本？
A: 可以基于训练好的模型实现文本生成功能（需要额外实现）

## 许可证

本项目仅用于学习和研究目的。

## 作者

深度学习课程作业

## 致谢

- 感谢PyTorch团队提供优秀的深度学习框架
- 感谢论文"Attention is All You Need"的作者







