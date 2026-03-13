import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential \
            (
                nn.Linear(d_model, d_ff),  # Q: 这不是只对一维做处理吗？
                nn.ReLU(),

                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        x = self.net(x)  # Q: 也就是说这里输入的是一个一维数据

        return x

