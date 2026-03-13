"""
训练 TimeXer 模型（全局归一化 + 样本过滤 + 早停 + 掩码损失）
数据来源：get_data.py 生成的多个 CSV 文件
内生变量：GDEA（广东碳排放权成交均价）
外生变量：其他所有序列（煤炭、天然气、原油、EUA、汇率、沪深300、SZA、HBEA等）
"""

import os
import torch
import tqdm
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import polars as pl
import json
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Tuple, Callable
import psutil as ps

from 不同长度的TimeXer.TimeXer import Model
from TimeXer使用.掩码机制 import data_metrix_create, create_padding_mask_from_padded_matrix
from data.get_data import get_data, store_data
from utils.GetPath import *
from utils.COLOR import *
from 模态分解.CEEFD import CEEFD


rd: bool = False  # 是否使用随机划分

def show_memory(is_show: bool = True) -> Tuple[float, float]:
    process = ps.Process(os.getpid())
    mem = process.memory_info()

    RSS = mem.rss / 1024 ** 2
    VMS = mem.vms / 1024 ** 2

    if is_show:
        printc(f"RSS: {RSS} MB, VMS: {VMS} MB", color="blue")

    if RSS >= 5000 or VMS >= 7000:
        _str = []

        if RSS >= 5000:
            _str.append("警告: RSS >= 5000 MB!")

        if VMS >= 7000:
            _str.append("警告: VMS >= 7000 MB!")

        for s in _str:
            printc(s, color="red")

    return RSS, VMS


# -------------------- 配置参数 --------------------
class Config:
    # 是否使用 CEEFD
    use_ceefd: bool = False

    # 数据参数
    seq_len = 48
    pred_len = 24
    patch_len = 16
    features = 'MS'
    enc_in = 1

    # 模型参数（适当减小复杂度）
    d_model = 128          # 原 256 → 128
    n_heads = 4            # 原 8 → 4
    e_layers = 2           # 保持 2  r: 这个参数会显著增加训练时长，但是得到效果可以从现在的第一个epoch的16变为5，效果显著
    d_ff = 128            # 原 512 → 256
    dropout = 0.2
    activation = 'gelu'
    factor = 5
    use_norm = True        # 仍启用，但实际使用全局归一化

    # 训练参数
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 200        # 可适当增加，早停会提前终止
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 滑动窗口验证参数
    train_window_size = 800  # 训练区间长度（时间步数，非样本数）
    val_window_size = 200  # 验证区间长度
    window_stride = 300  # 滑动步长（每次向前移动的时间步数）

    # 路径
    suffix = "_d_ff_256"
    data_path = PATH.get_Data_path()
    model_save_path = f"./Pure_TimeXer_model_weight{suffix}.pth" if not use_ceefd else f"CEEFD_TimeXer_model_weight{suffix}.pth"

    MIN_SEQ_LEN_FOR_DECOMP = 500  # 可根据实际情况调整
    top_k = 2  # 模态分解取前 k 个

ceefd = CEEFD().ceefd

# -------------------- 辅助函数 --------------------
# 定义能量筛选函数
def select_top_imfs(imf_list, top_k: Union[int, float]=-1) -> list:
    """
    从 imf_list（每个元素为一维数组）中选出能量最高的 top_k 个。
    返回筛选后的列表。

    Args:
        top_k: 默认为 -1,表示取所有 IMF | 若取 int 类型，表示取前 top_k 个 IMF | 若为 float(0-1) 则表示前百分之 top_k

    Return:
        返回一个包含筛选后的列表
    """
    if len(imf_list) <= top_k:
        return imf_list

    energies = [np.sum(imf**2) for imf in imf_list]
    sorted_idx = np.argsort(energies)[::-1]  # 降序

    if top_k > 0:
        if isinstance(top_k, int):
            selected = [imf_list[i] for i in sorted_idx[:top_k]]

        elif isinstance(top_k, float):
            if top_k > 1:
                top_k = 0.3

            k = int(len(sorted_idx) * top_k)
            selected = [imf_list[i] for i in sorted_idx[:k]]

        else:
            selected = [imf_list[i] for i in sorted_idx[:3]]

    else:
        selected = imf_list

    return selected


def fill_nan_1d(arr):
    """一维数组的 NaN 填充：前向填充 + 后向填充，最后确保无 NaN"""
    arr = arr.copy()
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    # 前向填充
    idx = np.where(~mask, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    arr[mask] = arr[idx[mask]]
    # 后向填充开头
    if np.isnan(arr[0]):
        first_valid = np.where(~np.isnan(arr))[0]
        if len(first_valid) > 0:
            arr[:first_valid[0]] = arr[first_valid[0]]
        else:
            arr[:] = 0.0
    # 最终安全检查
    if np.any(np.isnan(arr)):
        arr = np.nan_to_num(arr, nan=0.0)
    return arr

def create_samples(en_series, ex_series_list, ex_valids, config):
    """
    根据内生序列、外生序列列表及其有效性掩码，生成训练/验证样本。
    修复了原函数中重复添加外生变量数据的问题，并明确了处理逻辑。

    Args:
        en_series (np.ndarray): 内生变量序列，形状为 [total_len]。
        ex_series_list (List[np.ndarray]): 外生变量序列列表，每个元素形状为 [total_len]。
        ex_valids (List[np.ndarray]): 对应外生变量的有效性布尔掩码列表，True表示原始数据有效(非NaN)。
        config (Config): 配置对象，需包含 seq_len, pred_len 属性。

    Return:
        x_en_list (List[np.ndarray]): 内生变量输入窗口列表，每个元素形状为 [seq_len]。
        x_ex_list (List[List[np.ndarray]]): 外生变量输入窗口列表的列表。
              外层列表对应每个样本，内层列表对应每个外生变量，形状为 [seq_len]。
        y_list (List[np.ndarray]): 目标值窗口列表，每个元素形状为 [pred_len]。
        y_mask_list (List[np.ndarray]): 目标值窗口的有效性掩码，True表示对应位置原始数据有效。
        ex_mask_list (List[List[np.ndarray]]): 外生变量输入窗口的有效性掩码列表。
              结构与 x_ex_list 一致，True表示对应位置原始数据有效。
    """
    total_len = len(en_series)
    seq_len = config.seq_len
    pred_len = config.pred_len
    stride = 1  # 滑动步长，可配置

    x_en_list, x_ex_list, y_list, y_mask_list, ex_mask_list = [], [], [], [], []
    num_exog = len(ex_series_list)  # 外生变量的总数

    for start in range(0, total_len - seq_len - pred_len + 1, stride):
        end_en = start + seq_len
        end_pred = end_en + pred_len

        # --- 1. 处理内生变量 (输入与标签) ---
        x_en_raw = en_series[start:end_en]
        # 如果整个输入窗口都是NaN，跳过该样本（无有效信息）
        if np.all(np.isnan(x_en_raw)):
            continue
        x_en_filled = fill_nan_1d(x_en_raw)  # 前向+后向填充NaN

        y_raw = en_series[end_en:end_pred]
        y_filled = fill_nan_1d(y_raw)       # 填充标签中的NaN
        y_mask = ~np.isnan(y_raw)           # True 表示标签该位置原始有效

        # --- 2. 处理外生变量 ---
        x_ex_sample = []   # 当前样本的所有外生变量序列
        ex_mask_sample = [] # 当前样本的所有外生变量掩码

        # 遍历每一个外生变量
        for ex_seq, valid_mask in zip(ex_series_list, ex_valids):
            ex_window_raw = ex_seq[start:end_en]
            valid_window = valid_mask[start:end_en]  # 当前窗口的原始有效性

            # 填充当前窗口的NaN值
            ex_window_filled = fill_nan_1d(ex_window_raw)

            # **关键修复**：每个外生变量只添加一次
            x_ex_sample.append(ex_window_filled)

            # 生成掩码：如果窗口内全为原始NaN，则整个掩码为False（全部视为填充/无效）
            if not np.any(valid_window):
                ex_mask_sample.append(np.zeros_like(valid_window, dtype=bool))
            else:
                ex_mask_sample.append(valid_window)

        # --- 3. 保存当前样本 ---
        # 安全性检查（理论上每个外生变量都已处理，长度应一致）
        if len(x_ex_sample) != num_exog or len(ex_mask_sample) != num_exog:
            # 此情况不应发生，若发生则打印警告并跳过，确保数据一致性
            print(f"Warning: Sample at start={start} has inconsistent number of exogenous variables. Skipped.")
            continue

        x_en_list.append(x_en_filled)
        x_ex_list.append(x_ex_sample)
        y_list.append(y_filled)
        y_mask_list.append(y_mask)
        ex_mask_list.append(ex_mask_sample)

    # 最终检查（可选，用于调试）
    for idx, ex_list in enumerate(x_ex_list):
        if len(ex_list) != num_exog:
            raise ValueError(
                f"程序逻辑错误：样本 {idx} 的外生变量数量为 {len(ex_list)}，与预期 {num_exog} 不符。"
            )

    return x_en_list, x_ex_list, y_list, y_mask_list, ex_mask_list

class TimeXerDataset(Dataset):
    def __init__(self, en_series, ex_series_list, ex_valids, config, start_idx=0, end_idx=None):
        self.en_series = en_series
        self.ex_series_list = ex_series_list
        self.ex_valids = ex_valids
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.num_exog = len(ex_series_list)

        total_len = len(en_series)
        all_starts = list(range(0, total_len - self.seq_len - self.pred_len + 1))

        # 关键修改：过滤坏样本
        valid_starts = []
        for start in all_starts:
            end_en = start + self.seq_len
            x_en_raw = self.en_series[start:end_en]
            # 跳过输入窗口全为NaN的样本
            if not np.all(np.isnan(x_en_raw)):
                valid_starts.append(start)

        # 根据划分范围切片
        if end_idx is not None:
            self.starts = valid_starts[start_idx:end_idx]
        else:
            self.starts = valid_starts[start_idx:]

        print(f"数据集初始化完成，有效样本数: {len(self.starts)}")


    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end_en = start + self.seq_len
        end_pred = end_en + self.pred_len

        # 内生输入窗口
        x_en_raw = self.en_series[start:end_en]
        # 如果整个窗口为 NaN，理论上应在构建 starts 时过滤掉，这里加个保护
        if np.all(np.isnan(x_en_raw)):
            # 返回一个全零窗口并标记掩码全 False？但最好避免这种情况
            # 此处简单返回一个零窗口，掩码全 False，但训练时需注意
            x_en_filled = np.zeros(self.seq_len, dtype=np.float32)
        else:
            x_en_filled = fill_nan_1d(x_en_raw)

        # 目标窗口
        y_raw = self.en_series[end_en:end_pred]
        y_filled = fill_nan_1d(y_raw)
        y_mask = ~np.isnan(y_raw)   # True=有效

        # 外生变量窗口
        x_ex_sample = []
        ex_mask_sample = []
        for ex_seq, valid in zip(self.ex_series_list, self.ex_valids):
            ex_window_raw = ex_seq[start:end_en]
            valid_window = valid[start:end_en]
            ex_window_filled = fill_nan_1d(ex_window_raw)
            x_ex_sample.append(ex_window_filled)
            # 掩码：如果窗口全无效，则全 False
            if not np.any(valid_window):
                ex_mask_sample.append(np.zeros_like(valid_window, dtype=bool))
            else:
                ex_mask_sample.append(valid_window)

        # 返回 numpy 数组，让 collate_fn 转换为 tensor
        return x_en_filled, x_ex_sample, ex_mask_sample, y_filled, y_mask


def collate_fn(batch):
    # 将 batch 中每个样本的 numpy 数组转为 tensor
    # item: (x_en, x_ex, ex_mask, y, y_mask)
    x_en_batch = torch.stack([torch.tensor(item[0], dtype=torch.float32) for item in batch], dim=0)  # [B, seq_len]
    x_en_batch = x_en_batch.unsqueeze(-1)  # [B, seq_len, 1]
    y_batch = torch.stack([torch.tensor(item[3], dtype=torch.float32) for item in batch], dim=0)  # [B, pred_len]
    y_mask_batch = torch.stack([torch.tensor(item[4], dtype=torch.bool) for item in batch], dim=0)  # [B, pred_len]

    num_ex = len(batch[0][1])
    x_ex_stacked = []
    ex_masks_stacked = []
    for i in range(num_ex):
        ex_i = torch.stack([torch.tensor(item[1][i], dtype=torch.float32) for item in batch], dim=0)  # [B, seq_len]
        ex_i = ex_i.unsqueeze(-1)  # [B, seq_len, 1]
        mask_i = torch.stack([torch.tensor(item[2][i], dtype=torch.bool) for item in batch], dim=0)  # [B, seq_len]
        # 注意力掩码要求形状 [B, 1, 1, seq_len]，True 表示需要屏蔽
        mask_i = mask_i.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]
        # 原始 mask_i 中 True=有效，故取反得到 True=填充（应屏蔽）
        mask_i = ~mask_i
        x_ex_stacked.append(ex_i)
        ex_masks_stacked.append(mask_i)

    return x_en_batch, x_ex_stacked, ex_masks_stacked, y_batch, y_mask_batch


# -------------------- 掩码损失函数 --------------------

class MaskedMSELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target, mask):
        mask = mask.bool()
        if mask.sum() == 0:
            # 返回一个极小的非零损失值，确保梯度可以传播
            return torch.tensor(self.epsilon, device=pred.device, requires_grad=True)
        pred = pred[mask]
        target = target[mask]
        return F.mse_loss(pred, target)

class MaskedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target, mask):
        mask = mask.bool()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        pred = pred[mask]
        target = target[mask]
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.delta, 0.5 * diff**2, self.delta * (diff - 0.5 * self.delta))
        return loss.mean()


def main(use_ceefd: bool = None):
    printc(f"Start: {datetime.now()}", color="red")

    config = Config()
    if use_ceefd is None:
        config = config

    else:
        config.use_ceefd = use_ceefd
        config.model_save_path = './Pure_TimeXer_model_weight.pth' if not use_ceefd else "CEEFD_TimeXer_model_weight.pth"

    print("Loading and aligning data...")
    # 数据加载和对齐部分（保持原样）
    def parse_date(d):
        if isinstance(d, str):
            try:
                return datetime.strptime(d, "%Y/%m/%d").date()
            except ValueError:
                return datetime.strptime(d, "%Y-%m-%d").date()
        elif isinstance(d, datetime):
            return d.date()
        elif isinstance(d, pd.Timestamp):
            return d.date()
        else:
            return d

    all_files = []
    data_iter = get_Data(isIterator=True, isPath=False)
    if data_iter is None:
        raise RuntimeError("无法获取 Data 路径")
    for entry in data_iter:
        if entry.is_file():
            all_files.append(entry)
        else:
            all_files.extend(get_path_list(entry, is_all=True))

    all_dates = set()
    total_rows = 0
    file_info = []
    for fpath in all_files:
        df = pl.read_csv(fpath, infer_schema_length=10000)
        time_col = df[df.columns[0]]
        for d in time_col:
            all_dates.add(parse_date(d))
        is_endogenous = "GDEA" in str(fpath)
        if is_endogenous:
            if len(df.columns) >= 3:
                exog_cols = df.columns[2:]
            else:
                exog_cols = None
            total_rows += 1
        else:
            exog_cols = df.columns[1:]
        if exog_cols is not None:
            for col in exog_cols:
                if config.use_ceefd:
                    series = df[col].to_numpy()
                    other_IMFs, IMF_, Res, Res_ = ceefd(series)
                    if other_IMFs is not None:
                        num_comps = other_IMFs.shape[0] + len(IMF_) + 2
                    else:
                        num_comps = len(IMF_) + 2
                else:
                    num_comps = 1
                total_rows += num_comps
        file_info.append((fpath, exog_cols, is_endogenous))

    sorted_dates = sorted(all_dates)
    T = len(sorted_dates)
    date_to_idx = {date: i for i, date in enumerate(sorted_dates)}
    all_rows = []
    rss, vms = show_memory()
    for fpath, exog_cols, is_endogenous in file_info:
        if is_endogenous:
            df = pl.read_csv(fpath, infer_schema_length=10000)
            time_col = df[df.columns[0]]
            time_indices = [date_to_idx[parse_date(d)] for d in time_col]
            en_series = df[df.columns[1]].to_numpy()
            row = np.full(T, np.nan, dtype=np.float32)
            row[time_indices] = en_series
            all_rows.append(row)
            break
    new_rss, new_vms = show_memory()
    printc(f"delta rss: {new_rss - rss} MB, delta vms: {new_vms - vms} MB", color="magenta")
    _l = len(file_info)
    for idx, (fpath, exog_cols, is_endogenous) in enumerate(file_info):
        rss, vms = show_memory(False)
        df = pl.read_csv(fpath, infer_schema_length=10000)
        time_col = df[df.columns[0]]
        time_indices = [date_to_idx[parse_date(d)] for d in time_col]
        for col in exog_cols:
            series = df[col].to_numpy()
            if len(series) < config.MIN_SEQ_LEN_FOR_DECOMP:
                comps = [series]
            else:
                if config.use_ceefd:
                    other_IMFs, IMF_, Res, Res_ = ceefd(series)
                    if other_IMFs is not None:
                        all_imfs = list(other_IMFs) + list(IMF_)
                    else:
                        all_imfs = list(IMF_)
                    selected_imfs = select_top_imfs(all_imfs, top_k=config.top_k)
                    comps = selected_imfs
                else:
                    comps = [series]
            for comp in comps:
                row = np.full(T, np.nan, dtype=np.float32)
                row[time_indices] = comp
                all_rows.append(row)
        new_rss, new_vms = show_memory()
        printc(f"{idx + 1} / {_l}:\n\tdelta rss: {new_rss - rss}, delta vms: {new_vms - vms}", color="magenta")
    aligned_matrix = np.stack(all_rows)
    actual_rows = aligned_matrix.shape[0]
    printc(f"实际总行数: {actual_rows}, 预期总行数: {total_rows}", color="red")
    if actual_rows != total_rows:
        printc(f"警告: 实际行数 {actual_rows} 与第一遍统计的 {total_rows} 不一致，已使用实际行数", color="yellow")
    all_masks = create_padding_mask_from_padded_matrix(aligned_matrix, pad_value=np.nan)
    en_series = aligned_matrix[0, :]
    ex_series_list = [aligned_matrix[i, :] for i in range(1, aligned_matrix.shape[0])]
    ex_valids = [(~m.squeeze()).cpu().numpy() for m in all_masks[1:]]
    en_valid = en_series[~np.isnan(en_series)]
    if len(en_valid) == 0:
        print("Warning: No valid endogenous values. Using mean=0, std=1.")
        global_mean_en = 0.0
        global_std_en = 1.0
    else:
        global_mean_en = en_valid.mean()
        global_std_en = en_valid.std()
    global_means_ex = []
    global_stds_ex = []
    for i, ex_seq in enumerate(ex_series_list):
        ex_valid = ex_seq[~np.isnan(ex_seq)]
        if len(ex_valid) == 0:
            print(f"Warning: No valid values for exogenous variable {i}. Using mean=0, std=1.")
            global_means_ex.append(0.0)
            global_stds_ex.append(1.0)
        else:
            global_means_ex.append(ex_valid.mean())
            global_stds_ex.append(ex_valid.std())

    global_means_ex = torch.tensor(global_means_ex, dtype=torch.float32)
    global_stds_ex = torch.tensor(global_stds_ex, dtype=torch.float32)

    class SubsetTimeXerDataset(Dataset):
        """基于给定的有效样本起始点列表，包装 TimeXerDataset 的功能。"""

        def __init__(self, en_series, ex_series_list, ex_valids, config, starts_list):
            self.en_series = en_series
            self.ex_series_list = ex_series_list
            self.ex_valids = ex_valids
            self.seq_len = config.seq_len
            self.pred_len = config.pred_len
            self.num_exog = len(ex_series_list)
            self.starts = starts_list  # 直接使用传入的列表
            # 可选：如果希望再次过滤，可以在这里重新过滤，但需确保与传入的列表一致，否则应打印警告
            print(f"数据集初始化完成，有效样本数: {len(self.starts)}")

        def __len__(self):
            return len(self.starts)

        def __getitem__(self, idx):
            # 复制原 TimeXerDataset.__getitem__ 的逻辑，但使用 self.starts[idx]
            start = self.starts[idx]
            end_en = start + self.seq_len
            end_pred = end_en + self.pred_len

            x_en_raw = self.en_series[start:end_en]
            if np.all(np.isnan(x_en_raw)):
                x_en_filled = np.zeros(self.seq_len, dtype=np.float32)
            else:
                x_en_filled = fill_nan_1d(x_en_raw)

            y_raw = self.en_series[end_en:end_pred]
            y_filled = fill_nan_1d(y_raw)
            y_mask = ~np.isnan(y_raw)

            x_ex_sample = []
            ex_mask_sample = []
            for ex_seq, valid in zip(self.ex_series_list, self.ex_valids):
                ex_window_raw = ex_seq[start:end_en]
                valid_window = valid[start:end_en]
                ex_window_filled = fill_nan_1d(ex_window_raw)
                x_ex_sample.append(ex_window_filled)
                if not np.any(valid_window):
                    ex_mask_sample.append(np.zeros_like(valid_window, dtype=bool))
                else:
                    ex_mask_sample.append(valid_window)

            return x_en_filled, x_ex_sample, ex_mask_sample, y_filled, y_mask

    full_dataset = TimeXerDataset(en_series, ex_series_list, ex_valids, config, start_idx=0, end_idx=None)
    all_starts = full_dataset.starts  # List[int], 按时间升序

    # 假设 all_starts 已排序（TimeXerDataset 返回的 starts 是升序的）
    n_samples = len(all_starts)
    train_samples_per_window = 500  # 每个窗口训练样本数（可根据样本总数调整）
    val_samples_per_window = 100  # 每个窗口验证样本数
    step = 200  # 滑动步长（样本数）

    windows = []
    start_idx = 0
    while start_idx + train_samples_per_window + val_samples_per_window <= n_samples:
        train_starts = all_starts[start_idx: start_idx + train_samples_per_window]
        val_starts = all_starts[start_idx + train_samples_per_window:
                                start_idx + train_samples_per_window + val_samples_per_window]
        windows.append((train_starts, val_starts))
        start_idx += step

    print(f"共生成 {len(windows)} 个验证窗口")

    # 存储每个窗口的验证损失
    window_val_losses = []

    history = {"train_loss": [], "val_loss": []}
    for w_idx, (train_starts, val_starts) in enumerate(windows):
        print(f"\n========== 处理窗口 {w_idx + 1}/{len(windows)} ==========")
        print(f"训练集样本数: {len(train_starts)}, 验证集样本数: {len(val_starts)}")
        printc(f"Start: {datetime.now()}", color="red")

        # 重要：在每个窗口内重新计算归一化统计量（仅基于训练集）
        # 收集训练集所有样本的目标有效值（内生变量）
        train_targets = []
        for s in train_starts:
            end_en = s + config.seq_len
            end_pred = end_en + config.pred_len
            y_raw = en_series[end_en:end_pred]
            valid_mask = ~np.isnan(y_raw)
            if valid_mask.any():
                train_targets.extend(y_raw[valid_mask])
        window_mean_en = np.mean(train_targets) if train_targets else 0.0
        window_std_en = np.std(train_targets) if train_targets else 1.0

        # 收集训练集所有样本的外生变量有效值（每个外生变量单独收集）
        window_means_ex = []
        window_stds_ex = []
        num_exog = len(ex_series_list)
        for ex_idx in range(num_exog):
            ex_vals = []
            for s in train_starts:
                end_en = s + config.seq_len
                ex_window_raw = ex_series_list[ex_idx][s:end_en]
                valid_mask = ~np.isnan(ex_window_raw)
                if valid_mask.any():
                    ex_vals.extend(ex_window_raw[valid_mask])
            if len(ex_vals) == 0:
                # 若无有效值，用全局统计量或默认值（此处用全局统计量，但最好避免）
                ex_mean = global_means_ex[ex_idx].item()
                ex_std = global_stds_ex[ex_idx].item()
            else:
                ex_mean = np.mean(ex_vals)
                ex_std = np.std(ex_vals)
            window_means_ex.append(ex_mean)
            window_stds_ex.append(ex_std)

        # 转换为 tensor
        window_means_ex = torch.tensor(window_means_ex, dtype=torch.float32)
        window_stds_ex = torch.tensor(window_stds_ex, dtype=torch.float32)

        if len(train_targets) == 0:
            print("警告：训练集无有效目标值，跳过该窗口")
            continue

        # window_mean_en = np.mean(train_targets)
        # window_std_en = np.std(train_targets)


        # 创建数据集
        train_dataset = SubsetTimeXerDataset(en_series, ex_series_list, ex_valids, config, train_starts)
        val_dataset = SubsetTimeXerDataset(en_series, ex_series_list, ex_valids, config, val_starts)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

        # 重新初始化模型（传入窗口统计量）
        model = Model(
            config,
            global_mean_en=torch.tensor(window_mean_en),
            global_std_en=torch.tensor(window_std_en),
            global_means_ex=window_means_ex,
            global_stds_ex=window_stds_ex
        ).to(config.device)

        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        # criterion = MaskedMSELoss()
        criterion = MaskedHuberLoss()

        # --- 新增：余弦退火调度器 ---
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
        scaler = torch.cuda.amp.GradScaler()  # 初始化

        best_val_loss = float('inf')

        for epoch in range(config.num_epochs):
            model.train()
            train_loss_sum = 0.0
            train_valid_total = 0
            for x_en, x_ex, masks, y, y_mask in tqdm.tqdm(train_loader, desc=f"Window {w_idx + 1} Epoch {epoch + 1}",position=0):
                x_en = x_en.to(config.device)
                x_ex = [ex.to(config.device) for ex in x_ex]
                masks = [m.to(config.device) for m in masks]
                y = y.to(config.device)
                y_mask = y_mask.to(config.device)

                x_dec_dummy = x_en
                x_mark_dec_dummy = masks[0] if masks else None

                output = model(x_en, x_ex, masks, x_dec_dummy, x_mark_dec_dummy)
                loss = criterion(output.squeeze(-1), y, y_mask)

                train_loss_sum += loss.item() * y_mask.sum().item()
                train_valid_total += y_mask.sum().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # train_loss += loss.item()

            # 验证
            model.eval()
            val_loss_sum = 0.0
            val_valid_total = 0
            with torch.no_grad():
                for x_en, x_ex, masks, y, y_mask in val_loader:
                    x_en = x_en.to(config.device)
                    x_ex = [ex.to(config.device) for ex in x_ex]
                    masks = [m.to(config.device) for m in masks]
                    y = y.to(config.device)
                    y_mask = y_mask.to(config.device)

                    x_dec_dummy = x_en
                    x_mark_dec_dummy = masks[0] if masks else None

                    output = model(x_en, x_ex, masks, x_dec_dummy, x_mark_dec_dummy)
                    loss = criterion(output.squeeze(-1), y, y_mask)
                    val_loss_sum += loss.item() * y_mask.sum().item()
                    val_valid_total += y_mask.sum().item()

            train_loss = train_loss_sum / train_valid_total
            val_loss = val_loss_sum / val_valid_total  # 每个有效点的平均损失

            # --- 新增：更新学习率 ---
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # 保存最佳模型（每个窗口单独保存）
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = config.model_save_path.replace('.pth', f'_window{w_idx + 1}.pth')
                torch.save(model.state_dict(), model_path)
                printc(
                    f"Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}\n  -> Model saved to {model_path}",
                    color="yellow")
            else:
                print(f"Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        window_val_losses.append(best_val_loss)

    avg_val_loss = np.mean(window_val_losses)
    print(f"\n所有窗口平均验证损失: {avg_val_loss:.6f}")

    if not config.use_ceefd:
        with open(f"TimeXer_loss{config.suffix}.json", "w") as f:
            json.dump(history, f, indent=4)
    else:
        with open(f"CEEFD_TimeXer_loss{config.suffix}.json", "w") as f:
            json.dump(history, f, indent=4)

    print("Training completed. Best val loss: {:.6f}".format(best_val_loss))

if __name__ == "__main__":
    main(False)
    main(True)
