"""
消融实验脚本
运行多组实验，比较不同超参数配置的影响
"""
import os
import sys
import subprocess
import json
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 基础配置
BASE_CONFIG = {
    'dataset': 'wikitext2',
    'data_dir': './data',
    'seq_len': 128,
    'batch_size': 32,
    'd_model': 512,
    'd_ff': 2048,
    'dropout': 0.1,
    'lr': 1e-4,
    'weight_decay': 0.01,
    'num_epochs': 5,  # 消融实验使用较少的epoch以节省时间
    'max_grad_norm': 1.0,
    'seed': 42,
}

# 实验结果存储
results = []


def run_experiment(name, config, save_suffix=''):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"运行实验: {name}")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = ['python', '-m', 'src.train']
    
    # 添加基础配置
    for key, value in BASE_CONFIG.items():
        if key == 'num_epochs':
            cmd.extend([f'--{key}', str(config.get(key, value))])
        else:
            cmd.extend([f'--{key}', str(value)])
    
    # 添加实验特定配置
    for key, value in config.items():
        if key not in BASE_CONFIG:
            if isinstance(value, bool) and value:
                cmd.append(f'--{key}')
            elif not isinstance(value, bool):
                cmd.extend([f'--{key}', str(value)])
    
    # 设置保存目录
    save_dir = f'./checkpoints/ablation_{save_suffix}'
    log_dir = f'./logs/ablation_{save_suffix}'
    cmd.extend(['--save_dir', save_dir])
    cmd.extend(['--log_dir', log_dir])
    
    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"命令: {' '.join(cmd)}")
    print()  # 空行分隔
    
    # 运行实验（实时显示输出）
    try:
        # 不使用 capture_output，让输出实时显示
        result = subprocess.run(cmd, check=True)
        
        # 读取训练历史
        history_path = os.path.join(log_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # 提取最终结果
            final_train_loss = history['train_losses'][-1] if history.get('train_losses') else None
            final_val_loss = history['val_losses'][-1] if history.get('val_losses') else None
            
            # 计算困惑度
            final_train_ppl = None
            final_val_ppl = None
            if final_train_loss is not None:
                import math
                final_train_ppl = math.exp(final_train_loss)
            if final_val_loss is not None:
                import math
                final_val_ppl = math.exp(final_val_loss)
            
            result_data = {
                'experiment': name,
                'config': config,
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'final_train_perplexity': final_train_ppl,
                'final_val_perplexity': final_val_ppl,
                'log_dir': log_dir
            }
            results.append(result_data)
            
            print(f"✓ 实验完成")
            if final_train_loss is not None:
                print(f"  最终训练损失: {final_train_loss:.4f}")
            if final_val_loss is not None:
                print(f"  最终验证损失: {final_val_loss:.4f}")
            if final_train_ppl:
                print(f"  最终训练困惑度: {final_train_ppl:.2f}")
            if final_val_ppl:
                print(f"  最终验证困惑度: {final_val_ppl:.2f}")
        else:
            print(f"⚠ 警告: 未找到训练历史文件 {history_path}")
            
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 实验失败: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"  错误输出: {e.stderr}")
        print(f"  返回码: {e.returncode}")


def main():
    """运行所有消融实验"""
    print("开始消融实验...")
    print(f"基础配置: {BASE_CONFIG}")
    
    # 实验1: 位置编码类型
    print("\n" + "="*60)
    print("实验组1: 位置编码类型")
    print("="*60)
    
    run_experiment(
        "位置编码-正弦",
        {'pos_encoding': 'sinusoidal'},
        'pos_sinusoidal'
    )
    
    run_experiment(
        "位置编码-可学习",
        {'pos_encoding': 'learnable'},
        'pos_learnable'
    )
    
    # 实验2: 注意力头数
    print("\n" + "="*60)
    print("实验组2: 注意力头数")
    print("="*60)
    
    for num_heads in [4, 8, 16]:
        run_experiment(
            f"注意力头数-{num_heads}",
            {'num_heads': num_heads},
            f'heads_{num_heads}'
        )
    
    # 实验3: 层数
    print("\n" + "="*60)
    print("实验组3: Encoder层数")
    print("="*60)
    
    for num_layers in [3, 6, 9]:
        run_experiment(
            f"层数-{num_layers}",
            {'num_layers': num_layers},
            f'layers_{num_layers}'
        )
    
    # 实验4: 学习率调度
    print("\n" + "="*60)
    print("实验组4: 学习率调度器")
    print("="*60)
    
    # 固定学习率（不使用调度器，但需要修改代码支持）
    # 这里我们只测试可用的调度器
    for scheduler in ['cosine', 'step', 'warmup']:
        run_experiment(
            f"学习率调度-{scheduler}",
            {'scheduler': scheduler},
            f'scheduler_{scheduler}'
        )
    
    # 保存结果
    print("\n" + "="*60)
    print("保存实验结果...")
    print("="*60)
    
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)
    
    # 保存为JSON
    results_file = results_dir / 'ablation_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到: {results_file}")
    
    # 创建结果表格
    table_data = []
    for r in results:
        config_str = ', '.join([f"{k}={v}" for k, v in r['config'].items()])
        table_data.append({
            '实验名称': r['experiment'],
            '配置': config_str,
            '训练损失': f"{r['final_train_loss']:.4f}" if r['final_train_loss'] else 'N/A',
            '验证损失': f"{r['final_val_loss']:.4f}" if r['final_val_loss'] else 'N/A',
            '训练困惑度': f"{r['final_train_perplexity']:.2f}" if r.get('final_train_perplexity') else 'N/A',
            '验证困惑度': f"{r['final_val_perplexity']:.2f}" if r.get('final_val_perplexity') else 'N/A'
        })
    
    df = pd.DataFrame(table_data)
    csv_file = results_dir / 'ablation_results.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"表格已保存到: {csv_file}")
    
    # 打印摘要
    print("\n" + "="*60)
    print("实验结果摘要")
    print("="*60)
    print(df.to_string(index=False))
    
    print("\n消融实验完成！")


if __name__ == '__main__':
    main()

