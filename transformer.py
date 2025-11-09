"""
完整的Transformer模型实现
包括Encoder和Decoder
"""
import torch
import torch.nn as nn
import math
from .attention import MultiHeadAttention, PositionwiseFeedForward
from .positional_encoding import SinusoidalPositionalEncoding


class EncoderLayer(nn.Module):
    """
    Transformer Encoder层
    
    包含：
    1. Multi-head Self-Attention
    2. 残差连接 + Layer Normalization
    3. Position-wise Feed-Forward Network
    4. 残差连接 + Layer Normalization
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout比率
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: 注意力mask
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    Transformer Decoder层
    
    包含：
    1. Masked Multi-head Self-Attention
    2. 残差连接 + Layer Normalization
    3. Multi-head Cross-Attention (Encoder-Decoder Attention)
    4. 残差连接 + Layer Normalization
    5. Position-wise Feed-Forward Network
    6. 残差连接 + Layer Normalization
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout比率
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: [batch_size, tgt_len, d_model] Decoder输入
            encoder_output: [batch_size, src_len, d_model] Encoder输出
            src_mask: Encoder输出的mask
            tgt_mask: Decoder输入的mask（用于防止看到未来信息）
        
        Returns:
            output: [batch_size, tgt_len, d_model]
        """
        # Masked self-attention
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention (Encoder-Decoder Attention)
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    由多个EncoderLayer堆叠而成
    """
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            num_layers: Encoder层数
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout比率
        """
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: 注意力mask
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder
    由多个DecoderLayer堆叠而成
    """
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            num_layers: Decoder层数
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout比率
        """
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: [batch_size, tgt_len, d_model]
            encoder_output: [batch_size, src_len, d_model]
            src_mask: Encoder输出的mask
            tgt_mask: Decoder输入的mask
        
        Returns:
            output: [batch_size, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    """
    完整的Transformer模型
    
    支持两种模式：
    1. Encoder-only: 用于分类、语言模型等任务
    2. Encoder-Decoder: 用于机器翻译、摘要等序列到序列任务
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_len=5000,
        dropout=0.1,
        use_decoder=False,
        pos_encoding='sinusoidal'
    ):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_encoder_layers: Encoder层数
            num_decoder_layers: Decoder层数
            d_ff: 前馈网络隐藏层维度
            max_seq_len: 最大序列长度
            dropout: dropout比率
            use_decoder: 是否使用Decoder
            pos_encoding: 位置编码类型 ('sinusoidal' 或 'learnable')
        """
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.use_decoder = use_decoder
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        if pos_encoding == 'sinusoidal':
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        else:
            from .positional_encoding import LearnablePositionalEncoding
            self.pos_encoding = LearnablePositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, num_heads, d_ff, dropout
        )
        
        # Decoder (可选)
        if use_decoder:
            self.decoder = TransformerDecoder(
                num_decoder_layers, d_model, num_heads, d_ff, dropout
            )
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_mask(self, src, tgt=None, pad_idx=0):
        """
        生成注意力mask
        
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len] (可选)
            pad_idx: padding索引
        
        Returns:
            src_mask: [batch_size, 1, src_len]
            tgt_mask: [batch_size, tgt_len, tgt_len] (下三角mask)
        """
        # Source mask: 0表示padding位置
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        
        if tgt is not None:
            # Target mask: 防止看到未来信息
            tgt_len = tgt.size(1)
            tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device))
            tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
            tgt_mask = tgt_mask.unsqueeze(0) & tgt_pad_mask
            return src_mask, tgt_mask
        else:
            return src_mask, None
    
    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        """
        Args:
            src: [batch_size, src_len] 源序列
            tgt: [batch_size, tgt_len] 目标序列（可选，用于训练）
            src_mask: 源序列mask
            tgt_mask: 目标序列mask
        
        Returns:
            output: [batch_size, seq_len, vocab_size] 或 [batch_size, tgt_len, vocab_size]
        """
        # 词嵌入 + 位置编码
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        
        # Encoder
        encoder_output = self.encoder(src_emb, src_mask)
        
        if self.use_decoder and tgt is not None:
            # Decoder模式
            tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoding(tgt_emb)
            decoder_output = self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)
            output = self.output_projection(decoder_output)
        else:
            # Encoder-only模式
            output = self.output_projection(encoder_output)
        
        return output

