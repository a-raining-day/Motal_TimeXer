import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .Add_and_Normal import AddNorm
from .Attention import MultiHeadAttention, attention
from .Decoder import Decoder
from .Embedding import Embedding, PositionalEncoding
from .Encoder import Encoder
from .FeedForward import FeedForward

def create_src_mask(src, pad_idx=0):
    """创建源序列mask，padding位置为0"""
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)

def create_tag_mask(tag, pad_idx=0):
    """创建目标序列mask（因果mask + padding mask）"""
    batch_size, seq_len = tag.size()
    # 因果mask（下三角矩阵）
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=tag.device)).bool().unsqueeze(0).unsqueeze(0)
    # padding mask
    pad_mask = (tag != pad_idx).unsqueeze(1).unsqueeze(2)
    pad_mask = pad_mask.expand(batch_size, 1, seq_len, seq_len)

    return causal_mask & pad_mask

class Transformer(nn.Module):
    def __init__(self, src_vocab, tag_vocab, d_model=512, N=6, h=8, d_ff=2048, dropout=0.1):
        super().__init__()

        self.src_embedding = nn.Sequential \
            (
                Embedding(src_vocab, d_model),
                PositionalEncoding(d_model)
            )

        self.tag_embedding = nn.Sequential \
            (
                Embedding(tag_vocab, d_model),
                PositionalEncoding(d_model)
            )

        self.encoder = nn.ModuleList \
            (
                [
                    Encoder(d_model, MultiHeadAttention(h, d_model, dropout), FeedForward(d_model, d_ff, dropout), dropout) for _ in range(N)
                ]
            )  # 接收输入，通过处理后在Decoder的交叉注意力层中接收，作为参考为Decoder的输出作参考

        self.decoder = nn.ModuleList \
            (
                [
                    Decoder(d_model, MultiHeadAttention(h, d_model, dropout), MultiHeadAttention(h, d_model, dropout), FeedForward(d_model, d_ff, dropout), dropout) for _ in range(N)
                ]
            )  # 会不断进行输出，产生一个词后，会再次循环输入进Decoder，Decoder会因此不断输出成为一个序列

        self.out = nn.Linear(d_model, tag_vocab)
        self.pad_idx = 0

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_mask(self, src, tag):
        src_mask = create_src_mask(src, self.pad_idx)
        tag_mask = create_tag_mask(tag, self.pad_idx)

        return src_mask, tag_mask

    def encode(self, src, src_mask):
        x = self.src_embedding(src)

        for layer in self.encoder:
            x = layer(x, src_mask)

        return x

    def decode(self, tag, memory, src_mask, tag_mask):
        x = self.tag_embedding(tag)

        for layer in self.decoder:
            x = layer(x, memory, src_mask, tag_mask)

        return x

    def forward(self, src, tag, src_mask=None, tag_mask=None):
        if src_mask is None or tag_mask is None:
            src_mask, tag_mask = self.create_mask(src, tag)

        memory = self.encode(src, src_mask)
        output = self.decode(tag, memory, src_mask, tag_mask)
        output = self.out(output)  # 后续没有论文中的 softmax？
        # 这里是因为softmax只有在推理时才使用，训练时不需要加这个

        return output


