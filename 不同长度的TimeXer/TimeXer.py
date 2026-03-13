import torch
import torch.nn as nn
import torch.nn.functional as F
from .SelfAttention_Family import FullAttention, AttentionLayer
from .Embed import PositionalEmbedding, ExogEmbedding
from typing import List, Optional

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross: List, x_mask=None, cross_mask: List = None, tau=None, delta=None):
        for layer in self.layers:
            outputs = [layer(x, cross[i], x_mask=x_mask, cross_mask=cross_mask[i], tau=tau, delta=delta)
                       for i in range(len(cross))]
            outputs = torch.stack(outputs, dim=0)
            x = torch.mean(outputs, dim=0)
        if self.norm is not None:
            x = self.norm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        residual = x
        x = self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        x = self.dropout(x)
        x = x + residual
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))

        x_glb_attn = self.cross_attention(x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0]
        x_glb_attn = self.dropout(x_glb_attn)
        x_glb_attn = torch.reshape(x_glb_attn, (B, 1, D))   # [B, 1, D]
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Model(nn.Module):
    def __init__(self, configs,
                 global_mean_en=None, global_std_en=None,
                 global_means_ex=None, global_stds_ex=None):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in

        # 注册全局归一化统计量（若提供）
        if global_mean_en is not None:
            self.register_buffer('global_mean_en', global_mean_en)
            self.register_buffer('global_std_en', global_std_en)
            self.register_buffer('global_means_ex', global_means_ex)
            self.register_buffer('global_stds_ex', global_stds_ex)
        else:
            self.global_mean_en = None   # 标记未使用全局归一化

        # Embedding
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        self.ex_embedding = ExogEmbedding(1, configs.d_model, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_en, x_ex: List[torch.Tensor], x_mark_ex, x_dec, x_mark_dec):
        if x_en.dim() == 2:
            x_en = x_en.unsqueeze(-1)

        use_global = hasattr(self, 'global_mean_en') and self.global_mean_en is not None

        if use_global:
            # 全局归一化
            x_en = (x_en - self.global_mean_en) / (self.global_std_en + 1e-5)
            for i in range(len(x_ex)):
                x_ex[i] = (x_ex[i] - self.global_means_ex[i]) / (self.global_stds_ex[i] + 1e-5)
            means = self.global_mean_en  # 标量
            stdev = self.global_std_en
        elif self.use_norm:
            # 样本级归一化（原逻辑）
            means = x_en.mean(1, keepdim=True).detach()
            x_en = x_en - means
            stdev = torch.sqrt(torch.var(x_en, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_en /= stdev
            for i in range(len(x_ex)):
                temp = x_ex[i]
                _means = temp.mean(1, keepdim=True).detach()
                temp = temp - _means
                _stdev = torch.sqrt(torch.var(temp, dim=1, keepdim=True, unbiased=False) + 1e-5)
                temp /= _stdev
                x_ex[i] = temp
        else:
            means = stdev = None

        # 嵌入、编码等（保持不变）
        en_embed, n_vars = self.en_embedding(x_en[:, :, 0].unsqueeze(-1).permute(0, 2, 1))
        ex_embed = [self.ex_embedding(i) for i in x_ex]
        enc_out = self.encoder(en_embed, ex_embed, cross_mask=x_mark_ex)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if means is not None and stdev is not None:
            # 反归一化（直接广播，无需复杂索引）
            dec_out = dec_out * stdev + means

        return dec_out

    def forward(self, x_en, x_ex, x_mark_ex, x_dec, x_mark_dec, mask=None):
        # 数据生成阶段已保证每个样本的外生变量均有效，故移除动态过滤
        dec_out = self.forecast(x_en, x_ex, x_mark_ex, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # 取最后 pred_len 步