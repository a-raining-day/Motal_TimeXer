import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class AddNorm(nn.Module):
    # 在 Add & Normal 模块中，只是做 LayerNormal 和 dropout
    def __init__(self, size, dropout=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # Q: 我不明白，为什么要这样做，我更喜欢 forward(x, output: 对应现在的 sublayer(x))
        # Q: 为什么采用把 Module 传入之后，在forward中进行？我比较喜欢把流程全放在模块的forward中，而不是拆出部分放到子模块的forward中进行计算。这是有什么原因吗？
        # Q: 好吧，这没什么原因。只是另外一种设计思路
        output = sublayer(x)  # 这里是将之前的输入通过 sublayer
        output = self.dropout(output)  # dropout
        output = x + output  # 残差连接
        output = self.norm(output)  # 归一层

        return output
