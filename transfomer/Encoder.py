import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .Add_and_Normal import AddNorm

class Encoder(nn.Module):
    def __init__(self, d_model, self_attention, feed_forward, dropout: float):
        super().__init__()

        self.self_attention = self_attention  # 也就是一个Attention模块
        self.feed_forward = feed_forward
        self.sublayers = nn.ModuleList \
            (
                [
                    AddNorm(d_model, dropout),
                    AddNorm(d_model, dropout)
                ]
            )

    def forward(self, x, mask):
        output = self.sublayers[0](x, lambda x: self.self_attention(x, x, x, mask))
        output = self.sublayers[1](output, self.feed_forward)

        return output

