"""
简单的模型测试脚本
用于验证Transformer模型是否可以正常创建和运行
"""
import sys
import torch

print("=" * 50)
print("开始测试...")
print("Python版本:", sys.version)
print("PyTorch版本:", torch.__version__)
print("=" * 50)

try:
    print("\n[1/3] 导入Transformer模块...")
    from src.transformer import Transformer
    print("✓ 导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

def test_model():
    """测试模型创建和前向传播"""
    print("\n[2/3] 测试Transformer模型...")
    
    # 模型参数
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    seq_len = 128
    batch_size = 2
    
    # 创建模型
    print("创建模型...")
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        dropout=0.1,
        use_decoder=False  # 测试Encoder-only模式
    )
    
    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")
    
    # 创建随机输入
    print("创建随机输入...")
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播
    print("前向传播...")
    model.eval()
    with torch.no_grad():
        output = model(src)
    
    print(f"输入形状: {src.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出应该是: [{batch_size}, {seq_len}, {vocab_size}]")
    
    # 验证输出形状
    assert output.shape == (batch_size, seq_len, vocab_size), \
        f"输出形状不正确: {output.shape}"
    
    print("\n✅ 模型测试通过！")
    
    # 测试Decoder模式
    print("\n测试Decoder模式...")
    model_decoder = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        dropout=0.1,
        use_decoder=True
    )
    
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        output_decoder = model_decoder(src, tgt=tgt)
    
    print(f"Decoder输出形状: {output_decoder.shape}")
    assert output_decoder.shape == (batch_size, seq_len, vocab_size), \
        f"Decoder输出形状不正确: {output_decoder.shape}"
    
    print("✅ Decoder模式测试通过！")

if __name__ == '__main__':
    try:
        print("\n[3/3] 运行测试函数...")
        test_model()
        print("\n" + "=" * 50)
        print("所有测试完成！")
        print("=" * 50)
    except Exception as e:
        print(f"\n✗ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)







