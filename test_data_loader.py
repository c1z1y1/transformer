#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据加载器
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import download_wikitext2, load_wikitext2

def test_download():
    """测试下载功能"""
    print("=" * 50)
    print("测试数据下载和解压功能")
    print("=" * 50)
    
    try:
        data_dir = './data'
        extracted_dir = download_wikitext2(data_dir)
        print(f"\n✓ 数据集目录: {extracted_dir}")
        
        # 检查文件是否存在
        train_file = os.path.join(extracted_dir, "wiki.train.tokens")
        valid_file = os.path.join(extracted_dir, "wiki.valid.tokens")
        
        if os.path.exists(train_file):
            print(f"✓ 训练文件存在: {train_file}")
        else:
            print(f"✗ 训练文件不存在: {train_file}")
            
        if os.path.exists(valid_file):
            print(f"✓ 验证文件存在: {valid_file}")
        else:
            print(f"✗ 验证文件不存在: {valid_file}")
            
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_load():
    """测试加载功能"""
    print("\n" + "=" * 50)
    print("测试数据加载功能")
    print("=" * 50)
    
    try:
        # 测试加载训练集
        print("\n加载训练集...")
        train_texts = load_wikitext2('./data', 'train')
        print(f"✓ 训练集加载成功，共 {len(train_texts)} 行")
        
        # 测试加载验证集
        print("\n加载验证集...")
        val_texts = load_wikitext2('./data', 'valid')
        print(f"✓ 验证集加载成功，共 {len(val_texts)} 行")
        
        # 显示一些样本
        if train_texts:
            print(f"\n训练集样本（前3个token）: {train_texts[0][:3]}")
        if val_texts:
            print(f"验证集样本（前3个token）: {val_texts[0][:3]}")
            
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    print("开始测试数据加载器...\n")
    
    success = True
    success &= test_download()
    success &= test_load()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败")
    print("=" * 50)







