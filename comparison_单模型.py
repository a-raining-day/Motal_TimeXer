"""
LSTM, Random Forest, TimeXer, XGBoost, Autoformer, Crossformer

改一下代码，将每次运行的结果都保存下来，然后多运行几次，把最好的各个模型的MSE等结果、预测曲线点都保存下来，后面再把所有最好的结果结合起来进行一次画图
"""

import sys
import os
from typing import List
from tqdm import tqdm
import torch
import pickle
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

from utils.COLOR import printc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 导入数据
from 数据给予 import Metrix_Create
from data.get_uniform_time import get_time

# 导入各模型
from models import LSTM, Random_Forest, XGBoost, Crossformer, Autoformer
from models.TimeXer import Model as TimeXerModel
from models import Prophet as ProphetModel

NAMEs = \
    [
        "Autoformer",
        "LSTM",
        "TimeXer",
        "XGBoost",
        "RandomForest",
        "Crossformer",
        "Prophet"
    ]


# 定义一个简单的配置类
class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def quantile_loss(y_true, y_pred, quantile):
    errors = y_true - y_pred
    loss = torch.mean(torch.max((quantile - 1) * errors, quantile * errors))
    return loss

# ==================== 优化后的 TimeXer 训练函数（含验证集与早停）====================
def train_timexer(X_train, y_train, X_val, y_val, X_test, y_test,
                  seq_len, num_series,
                  time_features_train=None, time_features_val=None, time_features_test=None,
                  epochs=200, lr=2e-3, batch_size=32, patience=30):
    """
    X_train: (N_train, seq_len, num_series) 标准化后的特征
    y_train: (N_train,) 标准化后的目标值
    time_features_train: (N_train, seq_len, num_time_features) 或 None
    """

    bias_init = -torch.tensor([torch.mean(torch.Tensor(y_train))])

    # 构建配置（超参数经过调优）
    config = Config(
        task_name='short_term_forecast',
        features='MS',                # 多变量输入，单变量输出（预测最后一个特征）
        seq_len=seq_len,
        pred_len=1,                    # 预测下一步
        use_norm=False,                 # 外部已标准化
        patch_len=12,  # r: 暂时不用调 | 96可整除
        d_model=256,  # r: 暂时不用调
        dropout=0.1,                      # 增加 dropout 正则化
        embed='fixed',
        freq='d',                          # 时间特征频率，此处未使用
        factor=1,  # r: 暂时不用调
        n_heads=8,  # r: 暂时不用调 | d_model 可整除
        e_layers=1,  # r: 暂时不用调
        d_ff=256,  # r: 暂时不用调
        activation='gelu',
        enc_in=num_series,
        bias_init=bias_init
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TimeXerModel(config).to(device)

    # 转换为 tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device).unsqueeze(-1)  # (N, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device).unsqueeze(-1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # 时间特征处理
    if time_features_train is not None:
        train_mark = torch.tensor(time_features_train, dtype=torch.float32).to(device)
        val_mark = torch.tensor(time_features_val, dtype=torch.float32).to(device)
        test_mark = torch.tensor(time_features_test, dtype=torch.float32).to(device)
    else:
        train_mark = val_mark = test_mark = None

    # 构造训练集 DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t) if train_mark is None else TensorDataset(X_train_t, train_mark, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # criterion = nn.HuberLoss(delta=1.0)
    # criterion = nn.MSELoss()  # r: effect: No.2
    # criterion = nn.SmoothL1Loss()
    criterion = nn.L1Loss()  # r: effect: No.1
    # criterion = quantile_loss

    # 获取所有参数，排除 learnable_bias
    # params_except_bias = [p for n, p in model.named_parameters() if 'learnable_bias' not in n]

    # optimizer = torch.optim.AdamW \
    # (
    #     [
    #         {'params': params_except_bias, 'lr': lr, 'weight_decay': 1e-3},
    #         {'params': model.learnable_bias, 'lr': 0.001, 'weight_decay': 1e-5},  # 例如 5e-5
    #         # {'params': model.factor, 'lr': 0.01, 'weight_decay': 0.001}
    #     ]
    # )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)  # 余弦退火
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 早停变量
    # best_val_loss = float('inf')
    # best_model_state = None
    # patience_counter = 0
    #
    # criterion_mse = nn.MSELoss()
    # criterion_huber = nn.HuberLoss(delta=1.0)
    # criterion = criterion_mse
    # switch_patience = 10
    # no_improve = 0
    # 早停相关
    early_stop_best_loss = float('inf')
    early_stop_counter = 0

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if train_mark is not None:
                X_batch, mark_batch, y_batch = batch
            else:
                X_batch, y_batch = batch
                mark_batch = None

            optimizer.zero_grad()
            # x_mark_enc 传入时间特征，x_dec 和 x_mark_dec 传 None
            output = model(X_batch, mark_batch, None, None)   # 返回 (batch, pred_len, enc_in)
            pred = output[:, -1, -1]                           # 取最后一步的内生变量预测值
            loss = criterion(pred, y_batch.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 梯度裁剪
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        # y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        # bias = np.mean(y_pred - y_test)
        # print(f"Mean prediction error (bias): {bias:.4f}")

        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_output = model(X_val_t, val_mark, None, None)
            val_pred = val_output[:, -1, -1]
            val_loss = criterion(val_pred, y_val_t.squeeze()).item()

        scheduler.step(val_loss)

        # 早停检查（基于 early_stop_best_loss）
        if val_loss < early_stop_best_loss:
            early_stop_best_loss = val_loss
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Epoch {epoch}: learnable_bias = {model.learnable_bias.item():.4f}")

    # 加载最佳模型
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("learnable_bias:", model.learnable_bias.item())
        # print("factor:", model.factor.item())

    # --- 计算训练集上的平均偏差（标准化空间） ---
    model.eval()
    with torch.no_grad():
        train_output = model(X_train_t, train_mark, None, None)
        train_pred_scaled = train_output[:, -1, -1].cpu().numpy()  # 训练集标准化预测
    train_bias_scaled = np.mean(train_pred_scaled - y_train)  # 标准化空间的偏差
    print(f"Train bias (scaled): {train_bias_scaled:.6f}")

    # 测试
    with torch.no_grad():
        test_output = model(X_test_t, test_mark, None, None)
        y_pred_scaled = test_output[:, -1, -1].cpu().numpy()
        # 修正：减去训练集偏差
        y_pred_scaled_corrected = y_pred_scaled - train_bias_scaled

    return y_pred_scaled_corrected  # 返回修正后的标准化预测值

# ==================== 指标计算函数 ====================
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mse, rmse, mae, mape

def main(is_show: bool = True, **kwargs):
    iterations = kwargs.get("iter", 1)

    # ========== 参数设置 ==========
    SEQ_LEN = 96
    TRAIN_RATIO = 0.7          # 训练集比例
    VAL_RATIO = 0.1             # 验证集比例（从原训练集中分出）
    TEST_RATIO = 0.2            # 测试集比例
    EPOCHS = 200                 # 最大训练轮数（TimeXer 专用）
    BATCH_SIZE = 32  # r: 暂时不用调
    LR = 2e-3

    print("Loading data...\n")
    data_matrix, time_axis, _, _ = Metrix_Create()
    data_matrix = data_matrix[:, -3000:]   # 取后3000个点
    time_axis = time_axis[-3000:]

    # ========== 重采样为等间隔（每日） ==========
    datetime_index = pd.DatetimeIndex(time_axis)
    series_list = []
    for i in range(data_matrix.shape[0]):
        s = pd.Series(data_matrix[i, :], index=datetime_index)
        s_resampled = s.resample('D').asfreq().interpolate(method='linear', limit_direction='both')
        series_list.append(s_resampled)
    resampled_matrix = np.array([s.values for s in series_list])
    resampled_time = series_list[0].index

    # 确保无 NaN
    if np.any(np.isnan(resampled_matrix)):
        print("警告：重采样后数据中仍存在 NaN，将进行前后向填充。")
        df_resampled = pd.DataFrame(resampled_matrix.T).fillna(method='ffill').fillna(method='bfill')
        resampled_matrix = df_resampled.values.T
        # 剔除全为 NaN 的特征行（如果有）
        nan_rows = np.isnan(resampled_matrix).any(axis=1)
        if nan_rows.any():
            print(f"剔除以下全为 NaN 的特征索引: {np.where(nan_rows)[0]}")
            resampled_matrix = resampled_matrix[~nan_rows, :]

    endogenous = resampled_matrix[1, :]  # 目标变量
    # 外生变量：所有行（包括第一行和其他行）
    exogenous = np.vstack((resampled_matrix[0, :], resampled_matrix[2:, :]))

    # 重新排列特征：外生变量在前，内生变量在最后
    resampled_matrix_reordered = np.vstack([exogenous, endogenous])  # (num_series, time_len)
    num_series = resampled_matrix_reordered.shape[0]
    endog = resampled_matrix_reordered[-1, :]  # 获取目标变量行
    # ---------------------------------------------------------

    # ========== 构建滑动窗口样本 ==========
    X_seq = []
    y = []
    for t in range(SEQ_LEN, resampled_matrix_reordered.shape[1]):
        X_seq.append(resampled_matrix_reordered[:, t - SEQ_LEN:t].T)  # (seq_len, num_series)
        y.append(endogenous[t])  # 注意目标变量仍用原始的 endogenous（未添加未来信息的那一行）
    X_seq = np.array(X_seq)
    y = np.array(y)

    # 最终检查
    assert not np.any(np.isnan(X_seq)), "X_seq 中仍有 NaN！"
    assert not np.any(np.isnan(y)), "y 中仍有 NaN！"

    # ========== 划分训练/验证/测试集（按时间顺序） ==========
    n = len(X_seq)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    X_train, X_val, X_test = X_seq[:train_end], X_seq[train_end:val_end], X_seq[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    print(f"样本总数: {n}, 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # ========== 数据标准化 ==========
    # 对特征 X 进行标准化（基于训练集）
    ns, sl, nf = X_train.shape
    X_train_reshaped = X_train.reshape(-1, nf)          # (样本数*seq_len, 特征数)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(ns, sl, nf)

    ns_val, sl_val, nf_val = X_val.shape
    X_val_reshaped = X_val.reshape(-1, nf_val)
    X_val_scaled = scaler_X.transform(X_val_reshaped).reshape(ns_val, sl_val, nf_val)

    ns_test, sl_test, nf_test = X_test.shape
    X_test_reshaped = X_test.reshape(-1, nf_test)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(ns_test, sl_test, nf_test)

    # 对目标值 y 进行标准化
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # ========== 构建时间特征（可选，有助于 TimeXer） ==========
    # 生成全局时间特征（与重采样后的时间轴对齐）
    time_df = \
    {
        'dayofweek': resampled_time.dayofweek,   # 0-6
        'month': resampled_time.month,           # 1-12
        'day': resampled_time.day,               # 1-31
    }
    time_df['sin_dayofweek'] = np.sin(2 * np.pi * time_df['dayofweek'] / 7)
    time_df['cos_dayofweek'] = np.cos(2 * np.pi * time_df['dayofweek'] / 7)
    time_df['sin_month'] = np.sin(2 * np.pi * (time_df['month'] - 1) / 12)
    time_df['cos_month'] = np.cos(2 * np.pi * (time_df['month'] - 1) / 12)
    # 保留原始数值或仅使用周期特征，注意标准化策略

    time_df = pd.DataFrame(time_df)

    # 标准化时间特征（避免数值范围影响）
    time_features_raw = (time_df - time_df.mean()) / time_df.std()
    time_features_raw = time_features_raw.values   # (total_len, num_time_features)

    # 构建滑动窗口的时间特征窗口
    T_seq = []
    for t in range(SEQ_LEN, len(time_features_raw)):
        T_seq.append(time_features_raw[t - SEQ_LEN:t])   # (seq_len, num_time_features)
    T_seq = np.array(T_seq)  # (样本数, seq_len, num_time_features)

    # 划分时间特征（与 X 保持一致）
    T_train, T_val, T_test = T_seq[:train_end], T_seq[train_end:val_end], T_seq[val_end:]

    # ========== 树模型使用展平特征（基于标准化后的 X） ==========
    X_train_flat_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
    X_val_flat_scaled = X_val_scaled.reshape(X_val_scaled.shape[0], -1)
    X_test_flat_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], -1)

    # ========== 构建每个时间点的外生变量（用于 Prophet） ==========
    # 外生特征：除了目标变量之外的所有行（即 resampled_matrix_reordered 除最后一行）
    exog_features = resampled_matrix_reordered[:-1, :].T  # (total_len, num_series-1)
    # 时间特征（已在前面生成）
    time_features_all = time_features_raw  # (total_len, num_time_features)
    # 合并外生变量
    exog_all = np.concatenate([exog_features, time_features_all], axis=1)  # (total_len, num_exog)

    # 外生变量也要对齐到滑动窗口样本的时间点（从 SEQ_LEN 开始）
    exog_seq = exog_all[SEQ_LEN:]  # (n, num_exog)

    # 划分训练/验证/测试（与 y 的划分一致）
    exog_train, exog_val, exog_test = exog_seq[:train_end], exog_seq[train_end:val_end], exog_seq[val_end:]

    # 对外生变量进行标准化（基于训练集）
    scaler_exog = StandardScaler()
    exog_train_scaled = scaler_exog.fit_transform(exog_train)
    exog_val_scaled = scaler_exog.transform(exog_val)
    exog_test_scaled = scaler_exog.transform(exog_test)

    # 准备对应的日期
    dates_all = resampled_time[SEQ_LEN:]  # 与 y 对应的日期
    dates_train = dates_all[:train_end].tolist()
    dates_val = dates_all[train_end:val_end].tolist()
    dates_test = dates_all[val_end:].tolist()

    # 查看是否泄露未来信息
    # idx = 100  # 任意索引
    # print("exog_train[1]:", exog_train[idx, 1])
    # print("y_train[t]:", y_train[idx])
    # print("y_train[t-1]:", y_train[idx - 1] if idx > 0 else "N/A")
    # corr_lag1 = np.corrcoef(y_train[1:], y_train[:-1])[0, 1]
    # print(corr_lag1)
    # raise

    # ========== 定义模型及其需要的输入格式 ==========
    models = {
        # 'LSTM': (LSTM, 'seq_with_val'),  # r: 得炼
        # 'Prophet': (ProphetModel, 'prophet'),  # r: 一次就好，比XGBoost还快，而且效果也还行
        # 'XGBoost': (XGBoost, 'flat'),  # OK
        # 'RandomForest': (Random_Forest, 'flat'),  # OK, 反正就是比不过XGBoost
        'TimeXer': (train_timexer, 'custom'),  # OK
        # 'Crossformer': (Crossformer, 'seq_with_val'),  # OK, 一次就好了
        # 'Autoformer': (Autoformer, 'seq')  # r: 得炼

        # 'PatchTST': (PatchTST, 'seq'),
        # 'SCINet': (SCINet, 'seq'),  # 莫名发散
        # 'DLinear': (DLinear, 'seq')  # 莫名发散
        # 'iTransformer': (iTransformer, 'seq'),  # 已知产生 NaN，暂时剔除
    }

    printc(f"模型共有：{len(models.keys())}", color="cyan")
    printc(f"分别为：{list(models.keys())}", color="cyan")

    for it in range(iterations):
        results = {}
        predictions = {}
        for name, (module, input_type) in models.items():
            print(f"\nTraining {name}...")
            try:
                if name == 'TimeXer':
                    # TimeXer 使用标准化后的数据 + 时间特征
                    y_pred_scaled = module(
                        X_train_scaled, y_train_scaled,
                        X_val_scaled, y_val_scaled,
                        X_test_scaled, y_test_scaled,
                        SEQ_LEN, num_series,
                        time_features_train=T_train,
                        time_features_val=T_val,
                        time_features_test=T_test,
                        lr=LR,
                        batch_size=BATCH_SIZE,
                        patience=20
                    )

                elif name == "Autoformer":

                    # 为 Autoformer 准备 3 维时间特征 (month, day, dayofweek)

                    time_feat_cols = ['month', 'day', 'dayofweek']

                    time_feat_raw = time_df[time_feat_cols].values  # 原始未标准化

                    # 标准化：基于训练集拟合 scaler

                    train_len = len(T_train)  # T_train 原本基于 7 维特征，这里需要单独计算索引

                    # 注意：time_feat_raw 的长度与 resampled_time 相同（即总长度）

                    # 需要按同样的滑动窗口索引提取训练集部分用于拟合

                    # 简便做法：先对 time_feat_raw 整体滑动窗口，然后取前 train_end 个样本作为训练集拟合

                    # 但要注意不要使用验证/测试集的数据计算均值和标准差

                    # 这里我们重新构造窗口，但只使用训练集部分计算 scaler

                    time_feat_windows = []

                    for t in range(SEQ_LEN, len(time_feat_raw)):
                        time_feat_windows.append(time_feat_raw[t - SEQ_LEN:t])

                    time_feat_windows = np.array(time_feat_windows)  # (总样本数, seq_len, 3)

                    # 划分训练/验证/测试（与 X 保持一致）

                    T_auto_train = time_feat_windows[:train_end]  # 训练集部分

                    T_auto_val = time_feat_windows[train_end:val_end]

                    T_auto_test = time_feat_windows[val_end:]

                    # 对训练集进行拟合，然后转换所有集

                    # 将 (N, seq_len, 3) 重塑为 (N*seq_len, 3) 进行标准化

                    ns_train = T_auto_train.shape[0]

                    T_auto_train_reshaped = T_auto_train.reshape(-1, 3)

                    scaler_T_auto = StandardScaler()

                    T_auto_train_scaled = scaler_T_auto.fit_transform(T_auto_train_reshaped).reshape(ns_train, SEQ_LEN,
                                                                                                     3)

                    ns_val = T_auto_val.shape[0]

                    T_auto_val_reshaped = T_auto_val.reshape(-1, 3)

                    T_auto_val_scaled = scaler_T_auto.transform(T_auto_val_reshaped).reshape(ns_val, SEQ_LEN, 3)

                    ns_test = T_auto_test.shape[0]

                    T_auto_test_reshaped = T_auto_test.reshape(-1, 3)

                    T_auto_test_scaled = scaler_T_auto.transform(T_auto_test_reshaped).reshape(ns_test, SEQ_LEN, 3)

                    y_pred_scaled = Autoformer.train_and_predict(
                        X_train_scaled, y_train_scaled,
                        X_val_scaled, y_val_scaled,
                        X_test_scaled, y_test_scaled,
                        seq_len=SEQ_LEN,
                        num_series=num_series,
                        time_features_train=T_auto_train_scaled,
                        time_features_val=T_auto_val_scaled,
                        time_features_test=T_auto_test_scaled,
                        epochs=200,
                        lr=LR,
                        batch_size=BATCH_SIZE,
                        patience=20
                    )

                elif input_type == 'seq_with_val':
                    y_pred_scaled = module.train_and_predict(
                        X_train_scaled, y_train_scaled,
                        X_val_scaled, y_val_scaled,
                        X_test_scaled, y_test_scaled,
                        seq_len=SEQ_LEN,
                        num_series=num_series,
                        time_features_train=T_train,
                        time_features_val=T_val,
                        time_features_test=T_test,
                        epochs=200,
                        lr=LR,
                        batch_size=BATCH_SIZE,
                        patience=20
                    )

                elif input_type == 'prophet':
                    y_pred_scaled = module.train_and_predict(
                        y_train_scaled, y_val_scaled, y_test_scaled,
                        exog_train=exog_train_scaled,  # 仍可传入标准化后的外生变量
                        exog_val=exog_val_scaled,
                        exog_test=exog_test_scaled,
                        dates_train=dates_train,
                        dates_val=dates_val,
                        dates_test=dates_test,
                        prophet_params={
                            'seasonality_mode': 'additive',  # 改为加法
                            'changepoint_prior_scale': 0.1,  # 增大以允许更多变点
                            'yearly_seasonality': False,
                            'weekly_seasonality': True,
                            'daily_seasonality': False,
                            # 'growth': 'linear',
                            'changepoint_range': 0.8,
                            'seasonality_prior_scale': 5
                        },
                        scaler_y=scaler_y  # 传入 scaler
                    )

                elif input_type == 'seq':
                    # 深度学习模型：LSTM, Crossformer
                    y_pred_scaled = module.train_and_predict(
                        X_train_scaled, y_train_scaled,
                        X_test_scaled, y_test_scaled,
                        seq_len=SEQ_LEN,
                        num_series=num_series,
                        epochs=200,
                        lr=LR,
                        batch_size=BATCH_SIZE
                    )
                else:
                    # 树模型：XGBoost, RandomForest
                    y_pred_scaled = module.train_and_predict(
                        X_train_flat_scaled, y_train_scaled,
                        X_test_flat_scaled, y_test_scaled
                    )

                # 检查预测是否有效
                if y_pred_scaled is None or np.any(~np.isfinite(y_pred_scaled)):
                    print(f"  Warning: {name} 预测结果包含无效值，跳过评估。")
                    results[name] = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
                    predictions[name] = y_pred_scaled
                    continue

                # 反标准化得到原始尺度的预测值
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

                bias = np.mean(y_pred - y_test)
                print(f"Mean prediction error (bias): {bias:.4f}")

                # 计算指标（使用原始尺度的 y_test 和 y_pred）
                mse, rmse, mae, mape = compute_metrics(y_test, y_pred)
                results[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
                predictions[name] = y_pred
                print(f"{name} done. MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

            except Exception as e:
                print(f"Error training {name}: {e}")
                results[name] = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
                predictions[name] = None

        # ========== 输出结果表格 ==========
        print("\n" + "="*60)
        print("Model Performance Comparison")
        print("="*60)
        df_results = pd.DataFrame(results).T
        print(df_results.round(4))

        names = list(models.keys())
        better: List[bool] = [False for _ in range(len(names))]
        if not os.path.exists(temp_path := "model_comparison_results.csv"):  # 第一次创建
            df_results.to_csv(temp_path)  # 这里先做判定，如果效果更好再广播，允许覆盖
            better = [True for _ in better]
        else:
            old_result = pd.read_csv(temp_path, index_col=0)
            # print(old_result)
            for idx, i in enumerate(names):
                if i not in old_result.index:  # 需要创建新行
                    better[idx] = True  # 直接默认True

                elif ((old := old_result.at[i, "MSE"]) > (new := df_results.at[i, "MSE"]) or pd.isna(old)) and pd.notna(new):
                    better[idx] = True

            for idx, i in enumerate(better):
                if i:
                    old_result.loc[names[idx]] = df_results.loc[names[idx]]

            old_result.to_csv(temp_path)

        # 将各个模型单独列出来，进行保存
        for idx, i in enumerate(names):
            if better[idx] or not os.path.exists(f"{i}_predictions_plots.pkl"):  # 如果不存在直接保存一次
                with open(f"{i}_predictions_plots.pkl", "wb") as f:
                    pickle.dump(predictions[i], f)

        if not os.path.exists("y_true.pkl"):
            with open("y_true.pkl", "wb") as f:
                pickle.dump(y_test.tolist(), f)

    if is_show:
        # ========== 绘制预测对比图（取测试集最后200个点） ==========
        plt.figure(figsize=(12, 6))
        plt.plot(y_test[-200:], label='True', color='black', linewidth=2)
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        for i, (name, y_pred) in enumerate(predictions.items()):
            if y_pred is not None and not np.all(np.isnan(y_pred)):
                plt.plot(y_pred[-200:], label=name, color=colors[i], linestyle='--', alpha=0.7, linewidth=0.8)
        plt.legend()
        plt.title('Test Set Predictions (last 200 points)')
        plt.xlabel('Time Step')
        plt.ylabel('GDEA Price')
        plt.tight_layout()
        plt.savefig('prediction_comparison.png', dpi=150)
        plt.show()

def print_comparison() -> None:
    if not os.path.exists(temp_path := "model_comparison_results.csv"):
        printc("No file!")
    else:
        file = pd.read_csv(temp_path)

        print(file)

def plot_comparison(**kwargs) -> None:
    names = kwargs.get("names", NAMEs)
    suffix = kwargs.get("suffix", "predictions_plots.pkl")

    if not os.path.exists("y_true.pkl"):
        raise ValueError("不存在可用的真实值")

    with open("y_true.pkl", "rb") as f:
        y_true = pickle.load(f)

    path_ls = [f"{name}_{suffix}" for name in names]
    seq_ls = {}
    for idx, name in enumerate(names):
        with open(path_ls[idx], "rb") as f:
            seq_ls[name] = pickle.load(f)  # 直接存着序列数据点，后面直接用就行

    plt.figure(figsize=(12, 6))
    plt.plot(y_true[-200:], label='True', color='black', linewidth=2)

    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for i, (name, y_pred) in enumerate(seq_ls.items()):
        if y_pred is not None and not np.all(np.isnan(y_pred)):
            plt.plot(y_pred[-200:], label=name, color=colors[i], linestyle='--', alpha=0.7, linewidth=1)

    plt.legend()
    plt.title('Test Set Predictions (last 200 points)')
    plt.xlabel('Time Step')
    plt.ylabel('GDEA Price')
    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    # print_comparison()
    # main(True)
    print_comparison()
    plot_comparison(names=NAMEs)