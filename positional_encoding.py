"""
位置编码实现
支持正弦位置编码和可学习位置编码
"""
import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码
    
    实现公式：
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    其中pos是位置，i是维度索引
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: dropout比率
        """
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习位置编码
    使用嵌入层学习位置表示
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: dropout比率
        """
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb
        return x

