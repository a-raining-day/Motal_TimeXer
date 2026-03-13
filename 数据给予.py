"""
这里直接截断吧
不到一千的直接丢弃
取大于4000长度的数据
将数据保存在文件做准备，用 numpy 直接保存为 csv，后续用 polars 读取
"""

import pandas as pd
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit
from data.get_uniform_time import *
from datetime import datetime
import numpy as np
import polars as pl
from typing import List, AnyStr, Union, Tuple, Any, Set, Literal, Callable
from utils.GetPath import get_Data_path_list
from matplotlib import pyplot as plt
import seaborn as sns

def Metrix_Create():
    """"""

    path_ls = get_Data_path_list(ls_is_str=True)
    Total_Tiem = get_time(path_ls)  # 将时间轴定义为 col

    names = []  # 数据的对应文件名作为 columns
    file_num = 0
    for pth in path_ls:
        file = pl.read_csv(pth, infer_schema_length=10000)
        shape = file.shape

        file_num += shape[1] - 1

    Metrix = np.zeros((file_num, len(Total_Tiem)))
    Col_to_idx = {Total_Tiem[i]: i for i in range(len(Total_Tiem))}
    row = 0
    each_column_start_and_end = []  # 记录各序列开始、结束的索引
    for pth in path_ls:
        file = pl.read_csv(pth, infer_schema_length=10000)
        columns = file.columns

        name = pth.split('\\')
        name = name[-1].split('.')[0]

        for i in range(len(columns)):
            names.append(f"{name}_{i}")

        dates = file[:, 0].to_numpy().tolist()
        dates_ls = []
        for date in dates:
            if '/' in date:
                temp = date.split('/')

            else:
                temp = date.split('-')

            temp = datetime(int(temp[0]), int(temp[1]), int(temp[2]))  # 转为datetime对象
            dates_ls.append(temp)

        idx_to_idx = {idx: Col_to_idx[dates_ls[idx]] for idx in range(len(dates_ls))}

        for col in columns[1:]:
            temp = []
            series = file[col].to_numpy().tolist()
            each_column_start_and_end.append([(row, idx_to_idx[0]), (row, idx_to_idx[len(series) - 1])])
            for idx, j in enumerate(series):
                pos = idx_to_idx[idx]
                Metrix[row, pos] = j
            row += 1

    # ... [原有的代码，直到构建完 Metrix] ...
    # 新增：数据清洗 - 向前填充NaN
    df = pd.DataFrame(Metrix.T)  # 将矩阵转置为 (时间点, 序列数) 的DataFrame便于处理
    df.fillna(method='ffill', inplace=True)  # 用前一个有效值填充NaN
    # 如果第一行还有NaN（开头就是缺失），再用后一个值填充
    df.fillna(method='bfill', inplace=True)
    Metrix = df.values.T  # 转置回原来的 (序列数, 时间点) 形状

    Metrix[Metrix == 0] = np.nan

    return Metrix, Total_Tiem, names, each_column_start_and_end


def Give_Data_Metrix():
    Metrix, _, _, _ = Metrix_Create()

    return Metrix[:, -4000:]


if __name__ == '__main__':
    # metrix, time, names, start_end_range = Metrix_Create()
    # shape = metrix.shape
    #
    # print(names)

    # for i in range(shape[0]):
    #     temp = start_end_range[i]
    #     plt.plot((temp[0][1], temp[1][1]), (metrix[temp[0][0], temp[0][1]], metrix[temp[1][0], temp[1][1]]), label=names[i])
    #
    # plt.show()


    # 获取原始数据矩阵和时间轴
    data_matrix, time_axis, _, _ = Metrix_Create()  # data_matrix: (num_series, total_len)
    # 截取后4000个点（与 Give_Data_Metrix 一致）
    data_matrix = data_matrix[:, -4000:]
    time_axis = time_axis[-4000:]

    # 将时间轴转换为 pandas DatetimeIndex
    datetime_index = pd.DatetimeIndex(time_axis)

    # 为每个特征创建一个 Series
    series_list = []
    for i in range(data_matrix.shape[0]):
        s = pd.Series(data_matrix[i, :], index=datetime_index)
        # 重采样为每日，使用线性插值填充缺失值
        s_resampled = s.resample('D').asfreq().interpolate(method='linear')
        series_list.append(s_resampled)

    # 合并为新的数据矩阵 (num_series, new_len)
    resampled_matrix = np.array([s.values for s in series_list])
    resampled_time = series_list[0].index  # 新的时间轴

    plt.figure(figsize=(15, 5))
    # plt.plot(time_axis, data_matrix[0, :], alpha=0.5, label='Original (with gaps)')
    plt.plot(resampled_time, resampled_matrix[0, :], linewidth=1, label='Resampled (daily)')
    plt.legend()
    plt.title('GDEA Price: Original vs Resampled')
    plt.show()