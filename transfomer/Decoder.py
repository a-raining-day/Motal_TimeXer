import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .Add_and_Normal import AddNorm


class Decoder(nn.Module):
    def  __init__(self, d_model, self_attention, cross_attention, feed_forward, dropout):
        super().__init__()

        self.self_attention = self_attention
        self.cross_attention = cross_attention

        self.feed_forward = feed_forward

        self.sublayers = nn.ModuleList \
            (
                [
                    AddNorm(d_model, dropout),
                    AddNorm(d_model, dropout),
                    AddNorm(d_model, dropout)
                ]
            )

    def forward(self, x, memory, src_mask, tag_mask):
        x = self.sublayers[0](x, lambda x: self.self_attention(x, x, x, tag_mask))
        x = self.sublayers[1](x, lambda x: self.cross_attention(x, memory, memory, src_mask))
        x = self.sublayers[2](x, self.feed_forward)

        return x

