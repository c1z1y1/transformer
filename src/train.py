"""
训练脚本
包含学习率调度、梯度裁剪、AdamW优化器等功能
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

from .transformer import Transformer
from .data_loader import prepare_data


class Trainer:
    """
    Transformer模型训练器
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        vocab_size,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lr=1e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        scheduler_type='cosine',
        warmup_steps=4000,
        save_dir='./checkpoints',
        log_dir='./logs'
    ):
        """
        Args:
            model: Transformer模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            vocab_size: 词汇表大小
            device: 设备
            lr: 学习率
            weight_decay: 权重衰减
            max_grad_norm: 梯度裁剪阈值
            scheduler_type: 学习率调度器类型 ('cosine', 'step', 'warmup')
            warmup_steps: warmup步数
            save_dir: 模型保存目录
            log_dir: 日志目录
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.vocab_size = vocab_size
        self.max_grad_norm = max_grad_norm
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # 学习率调度器
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        elif scheduler_type == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
        else:
            self.scheduler = None
        
        # 目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_ppls = []  # 训练困惑度
        self.val_ppls = []    # 验证困惑度
        self.learning_rates = []
        
    def get_lr(self, step):
        """获取当前学习率（支持warmup）"""
        if self.scheduler_type == 'warmup':
            if step < self.warmup_steps:
                return self.optimizer.param_groups[0]['lr'] * (step / self.warmup_steps)
            else:
                return self.optimizer.param_groups[0]['lr']
        else:
            return self.optimizer.param_groups[0]['lr']
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 生成mask
            src_mask, _ = self.model.generate_mask(input_ids)
            
            # 模型输出
            output = self.model(input_ids, src_mask=src_mask)
            
            # 计算损失
            output = output.view(-1, self.vocab_size)
            target_ids = target_ids.view(-1)
            loss = self.criterion(output, target_ids)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # 更新参数
            self.optimizer.step()
            
            # 学习率调度
            if self.scheduler_type == 'warmup':
                # Warmup学习率
                current_step = len(self.train_losses) * len(self.train_loader) + batch_idx
                lr = self.get_lr(current_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            elif self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids in tqdm(self.val_loader, desc='Validating'):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # 生成mask
                src_mask, _ = self.model.generate_mask(input_ids)
                
                # 模型输出
                output = self.model(input_ids, src_mask=src_mask)
                
                # 计算损失
                output = output.view(-1, self.vocab_size)
                target_ids = target_ids.view(-1)
                loss = self.criterion(output, target_ids)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs=10, save_every=5, eval_every=1):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            save_every: 每N个epoch保存一次
            eval_every: 每N个epoch验证一次
        """
        print(f"开始训练，设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 80)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*80}")
            
            # 训练
            train_loss = self.train_epoch()
            train_ppl = torch.exp(torch.tensor(train_loss)).item()  # 困惑度
            self.train_losses.append(train_loss)
            self.train_ppls.append(train_ppl)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 验证
            if epoch % eval_every == 0:
                val_loss = self.validate()
                val_ppl = torch.exp(torch.tensor(val_loss)).item()  # 困惑度
                self.val_losses.append(val_loss)
                self.val_ppls.append(val_ppl)
                self.learning_rates.append(current_lr)
                
                print(f"\n训练指标:")
                print(f"  训练损失 (Train Loss): {train_loss:.4f}")
                print(f"  训练困惑度 (Train PPL): {train_ppl:.2f}")
                print(f"  验证损失 (Val Loss):   {val_loss:.4f}")
                print(f"  验证困惑度 (Val PPL):   {val_ppl:.2f}")
                print(f"  学习率 (Learning Rate): {current_lr:.6f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(f'best_model.pt')
                    print(f"\n✓ 保存最佳模型 (最佳验证损失: {val_loss:.4f}, 困惑度: {val_ppl:.2f})")
            else:
                # 即使不验证也输出训练指标
                print(f"\n训练指标:")
                print(f"  训练损失 (Train Loss): {train_loss:.4f}")
                print(f"  训练困惑度 (Train PPL): {train_ppl:.2f}")
                print(f"  学习率 (Learning Rate): {current_lr:.6f}")
            
            # 定期保存
            if epoch % save_every == 0:
                self.save_model(f'checkpoint_epoch_{epoch}.pt')
                print(f"\n✓ 保存检查点: checkpoint_epoch_{epoch}.pt")
        
        # 保存训练历史
        self.save_training_history()
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        # 输出训练总结
        print(f"\n{'='*80}")
        print("训练完成！")
        print(f"{'='*80}")
        if len(self.train_losses) > 0:
            final_train_loss = self.train_losses[-1]
            final_train_ppl = torch.exp(torch.tensor(final_train_loss)).item()
            print(f"最终训练损失: {final_train_loss:.4f}")
            print(f"最终训练困惑度: {final_train_ppl:.2f}")
        if len(self.val_losses) > 0:
            final_val_loss = self.val_losses[-1]
            final_val_ppl = torch.exp(torch.tensor(final_val_loss)).item()
            best_val_loss = min(self.val_losses)
            best_val_ppl = torch.exp(torch.tensor(best_val_loss)).item()
            print(f"最终验证损失: {final_val_loss:.4f}")
            print(f"最终验证困惑度: {final_val_ppl:.2f}")
            print(f"最佳验证损失: {best_val_loss:.4f}")
            print(f"最佳验证困惑度: {best_val_ppl:.2f}")
        print(f"{'='*80}")
    
    def save_model(self, filename):
        """保存模型"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, filepath)
    
    def load_model(self, filename):
        """加载模型"""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"模型已加载: {filepath}")
    
    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_ppls': self.train_ppls,
            'val_ppls': self.val_ppls,
            'learning_rates': self.learning_rates
        }
        filepath = os.path.join(self.log_dir, 'training_history.json')
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 困惑度曲线
        axes[0, 1].plot(self.train_ppls, label='Train PPL')
        if self.val_ppls:
            axes[0, 1].plot(self.val_ppls, label='Val PPL')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].set_title('Training and Validation Perplexity')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_yscale('log')  # 使用对数刻度，因为困惑度通常较大
        
        # 学习率曲线
        if self.learning_rates:
            axes[1, 0].plot(self.learning_rates)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].grid(True)
        
        # 损失和困惑度对比（如果有验证数据）
        if self.val_losses and self.val_ppls:
            ax_twin = axes[1, 1]
            ax_loss = ax_twin
            ax_ppl = ax_twin.twinx()
            
            epochs = range(1, len(self.val_losses) + 1)
            line1 = ax_loss.plot(epochs, self.val_losses, 'b-', label='Val Loss', linewidth=2)
            line2 = ax_ppl.plot(epochs, self.val_ppls, 'r-', label='Val PPL', linewidth=2)
            
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Validation Loss', color='b')
            ax_ppl.set_ylabel('Validation Perplexity', color='r')
            ax_loss.tick_params(axis='y', labelcolor='b')
            ax_ppl.tick_params(axis='y', labelcolor='r')
            ax_ppl.set_yscale('log')
            
            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax_loss.legend(lines, labels, loc='upper right')
            ax_loss.set_title('Validation Loss vs Perplexity')
            ax_loss.grid(True)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.log_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存: {save_path}")
        plt.close()


def main():
    """主训练函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练Transformer模型')
    parser.add_argument('--dataset', type=str, default='wikitext2', 
                       choices=['wikitext2', 'tinyshakespeare'],
                       help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据目录')
    parser.add_argument('--seq_len', type=int, default=128,
                       help='序列长度')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--d_model', type=int, default=512,
                       help='模型维度')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Encoder层数')
    parser.add_argument('--d_ff', type=int, default=2048,
                       help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout比率')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='梯度裁剪阈值')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'warmup'],
                       help='学习率调度器类型')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--use_decoder', action='store_true',
                       help='使用Decoder')
    parser.add_argument('--pos_encoding', type=str, default='sinusoidal',
                       choices=['sinusoidal', 'learnable'],
                       help='位置编码类型')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 准备数据
    print("准备数据...")
    train_loader, val_loader, vocab = prepare_data(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size
    )
    
    # 创建模型
    print("创建模型...")
    model = Transformer(
        vocab_size=len(vocab),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers if args.use_decoder else 0,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_decoder=args.use_decoder,
        pos_encoding=args.pos_encoding
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=len(vocab),
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        scheduler_type=args.scheduler,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
    
    # 训练
    trainer.train(num_epochs=args.num_epochs)
    
    print("训练完成！")


if __name__ == '__main__':
    main()

