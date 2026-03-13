import sys
import os
import json
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from utils.COLOR import printc

# 导入数据
from 数据给予 import Metrix_Create
from data.get_uniform_time import get_time

# 导入各模型
from model.TimeXer import Model as TimeXerModel

# 定义一个配置类
class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def train_timexer(X_train, y_train, X_val, y_val, X_test, y_test,
                  seq_len, patch_len, num_series,
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
        patch_len=patch_len,  # r: 暂时不用调 | 96可整除
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

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

        # print(f"Epoch {epoch}: learnable_bias = {model.learnable_bias.item():.4f}")

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

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mse, rmse, mae, mape

def main(**kwargs):
    SEQ_LEN = kwargs.get("SEQ_LEN", [48, 96, 168, 336, 720])
    PATCH_LEN = kwargs.get("PATCH_LEN", [8, 12, 16, 24, 32])
    EPOCH = kwargs.get("EPOCH", 200)
    iterations = kwargs.get("iterations", 1)

    if not isinstance(SEQ_LEN, list):
        SEQ_LEN = [SEQ_LEN]

    if not isinstance(PATCH_LEN, list):
        PATCH_LEN = [PATCH_LEN]

    # ========== 参数设置 ==========
    TRAIN_RATIO = 0.7          # 训练集比例
    VAL_RATIO = 0.1             # 验证集比例（从原训练集中分出）
    TEST_RATIO = 0.2            # 测试集比例
    BATCH_SIZE = 32
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

    results = {}  # (seq_len, patch_len): MSE | RMSE | MAE | MAPE
    predictions = {}

    print(f"Training TimeXer...\n")
    count = 0
    for it in range(iterations):
        for i, seq_len in enumerate(SEQ_LEN):
            # ========== 构建滑动窗口样本 ==========
            X_seq = []
            y = []
            for t in range(seq_len, resampled_matrix_reordered.shape[1]):  # 注意这里用的是 seq_len 循环
                X_seq.append(resampled_matrix_reordered[:, t - seq_len:t].T)  # (seq_len, num_series)
                y.append(endogenous[t])  # 注意目标变量仍用原始的 endogenous
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

            print(f"样本总数: {n}, 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}\n")

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

            time_df = pd.DataFrame(time_df)

            # 标准化时间特征（避免数值范围影响）
            time_features_raw = (time_df - time_df.mean()) / time_df.std()
            time_features_raw = time_features_raw.values   # (total_len, num_time_features)

            # 构建滑动窗口的时间特征窗口
            T_seq = []
            for t in range(seq_len, len(time_features_raw)):
                T_seq.append(time_features_raw[t - seq_len:t])   # (seq_len, num_time_features)
            T_seq = np.array(T_seq)  # (样本数, seq_len, num_time_features)

            # 划分时间特征（与 X 保持一致）
            T_train, T_val, T_test = T_seq[:train_end], T_seq[train_end:val_end], T_seq[val_end:]

            for j, patch_len in enumerate(PATCH_LEN):
                count += 1
                printc(f"{count} / {len(SEQ_LEN) * len(PATCH_LEN)} -> ({seq_len}, {patch_len})", color='red')

                y_pred_scaled = train_timexer(
                    X_train_scaled, y_train_scaled,
                    X_val_scaled, y_val_scaled,
                    X_test_scaled, y_test_scaled,
                    seq_len, patch_len, num_series,
                    time_features_train=T_train,
                    time_features_val=T_val,
                    time_features_test=T_test,
                    lr=LR,
                    epochs=EPOCH,
                    batch_size=BATCH_SIZE,
                    patience=20
                )

                # 检查预测是否有效
                if y_pred_scaled is None or np.any(~np.isfinite(y_pred_scaled)):
                    print(f"  Warning: {(seq_len, patch_len)} 预测结果包含无效值，跳过评估。")
                    results[(seq_len, patch_len)] = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
                    predictions[(seq_len, patch_len)] = y_pred_scaled
                    continue

                # 反标准化得到原始尺度的预测值
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

                # 计算指标（使用原始尺度的 y_test 和 y_pred）
                mse, rmse, mae, mape = compute_metrics(y_test, y_pred)
                results[(seq_len, patch_len)] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
                predictions[(seq_len, patch_len)] = y_pred
                print(f"(seq_len, patch_len)->{(seq_len, patch_len)} done. MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%\n")

        # 将这部分逻辑放进iterations中，这样后面突然不想继续练了，可以等当前iteration结束后直接退出
        if not os.path.exists("result.pkl"):  # 初次创建
            with open("result.pkl", 'wb') as f:
                pickle.dump(results, f)

        else:
            with open("result.pkl", "rb") as f:
                old_result = pickle.load(f)

        for key, item in results.items():
            if key not in old_result:
                old_result[key] = item

            else:
                if old_result[key]["MSE"] > item["MSE"] or np.isnan(old_result[key]["MSE"]):
                    old_result[key] = item

        with open("result.pkl", "wb") as f:
            pickle.dump(old_result, f)

if __name__ == '__main__':
    main(SEQ_LEN=96, PATCH_LEN=12, iterations=5)