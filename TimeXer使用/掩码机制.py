"""
接收不同长度的序列，要求采样频率一致且带时间戳。
修改：缺失值用 np.nan 填充，掩码基于 np.isnan 生成。
"""

import numpy as np
import polars as pl
import pandas as pd
import torch
from datetime import datetime
from typing import List, Union, Tuple, Any
from numpy import ndarray


def data_metrix_create(ex_sequence: List[Union[pl.DataFrame, np.ndarray, list]]) -> ndarray:
    """
    将所有单变量序列对齐到全局时间轴，缺失位置用 np.nan 填充。
    返回 shape [num_vars, total_len] 的 numpy 数组。
    """
    # 提取所有序列的时间列和数值列
    all_dates = []
    series_values = []   # 每个元素为 (数值列表, 对应日期列表)

    for seq in ex_sequence:
        if isinstance(seq, pl.DataFrame):
            dates = seq["时间"].to_list()
            # 假设数值列是第二列（索引1）
            values = seq.select(seq.columns[1]).to_series().to_list()
        elif isinstance(seq, np.ndarray):
            # 假设第一列为时间，第二列为数值
            dates = seq[:, 0].tolist()
            values = seq[:, 1].tolist()
        elif isinstance(seq, list):
            # 假设第一个子列表为时间，第二个为数值
            dates = seq[0]
            values = seq[1]
        else:
            raise TypeError(f"Unsupported type: {type(seq)}")

        # 统一日期格式为 datetime 对象
        parsed_dates = []
        for d in dates:
            if isinstance(d, str):
                if '/' in d:
                    dt = datetime.strptime(d, "%Y/%m/%d")
                else:
                    dt = datetime.strptime(d, "%Y-%m-%d")
            else:
                dt = d   # 假设已经是 datetime
            parsed_dates.append(dt)
        all_dates.extend(parsed_dates)
        series_values.append((parsed_dates, values))

    # 全局时间范围
    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range: pd.DatetimeIndex = pd.date_range(start=min_date, end=max_date, freq='D')
    total_len = len(date_range)
    date_to_idx = {d.date(): i for i, d in enumerate(date_range)}  # 使用 date 对象

    # 构建矩阵
    num_vars = len(series_values)
    matrix = np.full((num_vars, total_len), np.nan, dtype=np.float32)

    for var_idx, (dates, values) in enumerate(series_values):
        for d, v in zip(dates, values):
            # 将 datetime 转为 date 用于索引
            idx = date_to_idx.get(d.date())
            if idx is not None and v is not None:
                matrix[var_idx, idx] = float(v)

    return matrix



def create_padding_mask_from_padded_matrix(padded_matrix: np.ndarray, pad_value=np.nan) -> List[torch.Tensor]:
    """
    从填充矩阵生成掩码，True 表示该位置为填充（应屏蔽）。
    基于 np.isnan 逐元素判断，不再依赖连续有效区间。

    Args:
        padded_matrix (np.ndarray): 形状为 [num_vars, total_len] 的矩阵，缺失值用 np.nan 填充。
        pad_value: 保留参数，未使用（仅用于兼容）。

    Returns:
        List[torch.Tensor]: 每个元素为形状 [1, 1, 1, total_len] 的布尔张量，
                             True 表示该位置是填充值（应被模型屏蔽）。
    """
    masks = []
    for row in padded_matrix:
        # 逐元素判断是否为 NaN，True 表示填充
        mask_np = np.isnan(row)                     # 布尔数组，shape [total_len]
        mask_tensor = torch.as_tensor(mask_np, dtype=torch.bool)
        # 扩展维度至 [1, 1, 1, total_len] 以匹配注意力掩码的格式
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        masks.append(mask_tensor)
    return masks