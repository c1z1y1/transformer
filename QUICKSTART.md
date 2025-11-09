# 快速开始指南

## 1. 环境准备

### 安装Python依赖

```bash
pip install -r requirements.txt
```

### 验证安装

```bash
python test_model.py
```

如果看到 "✅ 模型测试通过！"，说明环境配置正确。

## 2. 快速训练

### 使用脚本（推荐）

**Windows:**
```cmd
scripts\run.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/run.sh
./scripts/run.sh
```

### 手动运行

```bash
python -m src.train --dataset wikitext2 --num_epochs 5 --batch_size 16
```

## 3. 查看结果

训练完成后，结果保存在：
- 模型检查点：`./checkpoints/best_model.pt`
- 训练曲线：`./logs/training_curves.png`
- 训练日志：`./logs/training_history.json`

## 4. 继续训练

如果需要继续训练，可以修改训练脚本加载检查点：

```python
from src.train import Trainer
from src.transformer import Transformer
from src.data_loader import prepare_data

# 准备数据和模型（与之前相同）
train_loader, val_loader, vocab = prepare_data(...)
model = Transformer(...)
trainer = Trainer(...)

# 加载检查点
trainer.load_model('best_model.pt')

# 继续训练
trainer.train(num_epochs=10)
```

## 5. 常见问题

### Q: 内存不足
A: 减小batch_size或seq_len：
```bash
python -m src.train --batch_size 16 --seq_len 64
```

### Q: 训练太慢
A: 使用GPU或减小模型：
```bash
python -m src.train --d_model 256 --num_layers 4
```

### Q: 数据集下载失败
A: 手动下载数据集到`./data/`目录

## 6. 下一步

- 阅读 `README.md` 了解详细说明
- 查看 `docs/report.tex` 了解技术细节
- 修改超参数进行实验
- 尝试不同的数据集







