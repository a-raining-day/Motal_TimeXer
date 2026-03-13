import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):  # 直接使用了 nn.Embedding 模块
    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)  # TODO: Embedding内部还要看
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=x.dtype, device=x.device))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        PE = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)  # Q: unsqueeze是什么作用？
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        PE[:, 0::2] = torch.sin(position * div_term)  # 为PE赋值
        PE[:, 1::2] = torch.cos(position * div_term)

        PE = PE.unsqueeze(0)

        self.register_buffer('PE', PE)  # 注册变量。但是为了什么作用

    def forward(self, x):
        return x + self.PE[:, :x.size(1)]  # 相加