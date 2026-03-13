import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)  # TODO: query 与 key 的矩阵乘法是计算相似度，类似向量的内积计算相似度，计算两个矩阵的相似度
    # query 和 key 的 shape 一致 -> [batch_size, h, seq_q / seq_k, d_k] -> batch_size | multi-head | query or key length | key length
    # transpose(-1, -2) 是对调 -1 和 -2 维度，或者说是对三阶张量的矩阵进行转置
    # 举例：A = [batch_size, h, w] -> transpose(-1, -2) -> [batch_size, w, h] 相当于对每个 batch 的矩阵进行了转置
    # 这样，就能将 query 的 [seq_q, d_k] 与 key 的 [seq_k, d_k] 进行矩阵乘法 -> [seq_q, seq_k]
    # 得到的矩阵作为注意力矩阵，这个矩阵 [seq_q, seq_q] 的行列分别表示：query 和 key，归一后的数值表示相似度

    if mask is not None:  # 含mask
        if mask.dtype != torch.bool:
            mask = mask.bool()  # 所以 mask 是. bool 类型

        scores = scores.masked_fill(mask == False, float('-inf'))
        # scores 是 [query, key] 的相似度矩阵，也称注意力矩阵
        # 为什么 mask == 0，mask 不是 bool 类型吗？ -> 0 等价于 False -> mask == False -> 对 False 的地方，赋值为负无穷，作为因果编码
        # 但如果 mask 为 None 呢？这时候的相似度矩阵就不是因果的了

    activate_attention = torch.softmax(scores, dim=-1)  # TODO: 归一化得到的相似度，负无穷部分会变为0

    if dropout is not None:
        activate_attention = dropout(activate_attention)

    return activate_attention @ value, activate_attention  # 得到的是
    # activate_attention 是相似度矩阵的话，那么 activate 与 value 的矩阵乘法代表着什么？
    # 为什么 query 要和 key 乘，为什么之后又要与 value 乘？这个得到的就是 attention?
    # 但是 value 不应该也是 [batch_size, h, seq_v, d_k] 吗？
    # TODO: activate_attention @ value 得到的矩阵每一行，不仅表示当前行的信息，同时还融入了上一行的信息，以确保生成的信息是连续、流动、流通的，且表示了当前的信息的权重(也就是当生成到当前词时是会重点参考现在的词权重的，但又会同时考虑前面的生成)

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()

        assert d_model % h == 0  # 断言
        # 8个头要均分 d_model

        self.d_k = d_model // h
        self.h = h

        self.linear_q = nn.Linear(d_model, d_model)  # 全是一维的线性变换层
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        def transform(x, linear):
            # 对一维输入做了线性映射后，通过 .view 转化为了4维张量
            x = linear(x)  # x -> [batch, seq_len, d_model]
            return x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # 这样拆的时候，batch_size == batch | -1 == seq_len | h * d_k = d_model
            # 在 pytorch 中，内存存储的是连续的，因此需要 -1 对应 seq_len，然后再用 transpose 进行转变
            # 将 d_model 拆为 h 个均匀的层，每一层的宽度都是 d_k
            # 这个地方就将一维转为了4维张量

        query = transform(query, self.linear_q)
        key = transform(key, self.linear_k)
        value = transform(value, self.linear_v)

        x, _ = attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        # Q: contiguous 有什么用？

        x = self.linear_out(x)

        return x
