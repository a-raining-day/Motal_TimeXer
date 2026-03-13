import os
import psutil as ps
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

from 模态分解.CEEFD import *
from 模态分解.EFD import *

from utils.COLOR import printc

from give_data import *  # Metrix, Total_Tiem, names, each_column_start_and_end
from TimeXer import Model as TimeXerModel

from typing import Literal, Union

def memory(prefix: str = None, is_print: bool = True) -> Tuple[float, float]:
    process = ps.Process(os.getpid())
    mem = process.memory_info()

    RSS = mem.rss / 1024 ** 2
    VMS = mem.vms / 1024 ** 2

    prefix = "" if prefix is None else f"{prefix} | "

    if is_print:
        printc(f"{prefix}RSS: {RSS}, VMS: {VMS}", color="cyan")

    return RSS, VMS

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def train_timexer(X_train, y_train, X_val, y_val, X_test, y_test,
                  seq_len, num_series,
                  time_features_train=None, time_features_val=None, time_features_test=None,
                  epochs=200, lr=2e-3, batch_size=32, patience=30):
    """
    优化版 TimeXer 训练：带可学习偏置、训练偏差修正、ReduceLROnPlateau 调度、L1Loss
    """
    # 构建配置（使用文档二调优后的超参数）
    config = Config(
        task_name='short_term_forecast',
        features='MS',                # 多变量输入，单变量输出
        seq_len=seq_len,
        pred_len=1,
        use_norm=False,
        patch_len=12,                  # 与文档二一致
        d_model=256,
        dropout=0.1,
        embed='fixed',
        freq='d',
        factor=1,
        n_heads=8,
        e_layers=1,
        d_ff=256,
        activation='gelu',
        enc_in=num_series
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TimeXerModel(config).to(device)

    # 转换为 tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device).unsqueeze(-1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device).unsqueeze(-1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    if time_features_train is not None:
        train_mark = torch.tensor(time_features_train, dtype=torch.float32).to(device)
        val_mark = torch.tensor(time_features_val, dtype=torch.float32).to(device)
        test_mark = torch.tensor(time_features_test, dtype=torch.float32).to(device)
    else:
        train_mark = val_mark = test_mark = None

    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t) if train_mark is None else TensorDataset(X_train_t, train_mark, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.L1Loss()  # 文档二验证效果较好的损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 早停相关
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    Memory = []  # 内存监控（来自文档一）

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if train_mark is not None:
                X_batch, mark_batch, y_batch = batch
            else:
                X_batch, y_batch = batch
                mark_batch = None

            optimizer.zero_grad()
            output = model(X_batch, mark_batch, None, None)
            pred = output[:, -1, -1]
            loss = criterion(pred, y_batch.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 梯度裁剪（文档二使用5.0）
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t, val_mark, None, None)
            val_pred = val_output[:, -1, -1]
            val_loss = criterion(val_pred, y_val_t.squeeze()).item()

        scheduler.step(val_loss)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # 记录内存（文档一保留）
        Memory.append(memory(is_print=False))

        # 打印可学习偏置（文档二调试信息）
        print(f"Epoch {epoch}: learnable_bias = {model.learnable_bias.item():.4f}")

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 计算训练集上的平均偏差（标准化空间），用于修正测试集预测
    model.eval()
    with torch.no_grad():
        train_output = model(X_train_t, train_mark, None, None)
        train_pred_scaled = train_output[:, -1, -1].cpu().numpy()
    train_bias_scaled = np.mean(train_pred_scaled - y_train)  # y_train 是标准化后的值
    print(f"Train bias (scaled): {train_bias_scaled:.6f}")

    # 测试集预测并修正偏差
    with torch.no_grad():
        test_output = model(X_test_t, test_mark, None, None)
        y_pred_scaled = test_output[:, -1, -1].cpu().numpy()
        y_pred_scaled_corrected = y_pred_scaled - train_bias_scaled

    # 输出平均内存占用（文档一）
    printc(f"平均 RSS:{np.average([i[0] for i in Memory])} | VMS: {np.average([i[1] for i in Memory])}", color="cyan")

    return y_pred_scaled_corrected

module = train_timexer

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mse, rmse, mae, mape

ceefd = CEEFD()

ceemdan = ceefd.ceemdan
ceefd = ceefd.ceefd
efd = EFD

def Modal(mode: Literal["ceemdan", "ceefd", "efd"], top_k: Union[int, float]) -> Callable:
    def f(metrix: np.ndarray) -> np.ndarray:
        if np.ndim(metrix) == 1 or metrix.shape[0] == 1:
            data = (eval(mode)(metrix))

            if mode == "ceefd":
                col1, col2, col3, _ = data
                col1 = np.stack(col1)
                col2 = np.stack(col2)
                col3 = np.array(col3).reshape(1, -1)
                total = np.vstack((col1, col2, col3))

            else:
                col1, col2 = data
                col1 = np.stack(col1)
                col2 = np.array(col2).reshape(1, -1)
                total = np.vstack((col1, col2))

            col = total.shape[0]

            if top_k == -1:
                return total

            if isinstance(top_k, int):
                if top_k <= 0:
                    raise ValueError("top_k > 0!")
                if top_k >= total.shape[0]:
                    return total

                return total[:top_k, :]

            elif isinstance(top_k, float):
                if top_k <= 0 or top_k > 1:
                    raise ValueError("top_k in (0, 1]!")
                return total[:int(col * top_k), :]

        else:
            shape = metrix.shape

            Total = np.array([0 for _ in range(shape[1])]).reshape(1, -1)

            for i in tqdm(range(shape[0])):
                data = (eval(mode)(metrix[i, :]))
                if mode == "ceefd":
                    col1, col2, col3, _ = data
                    col1 = np.stack(col1)
                    col2 = np.stack(col2)
                    col3 = np.array(col3).reshape(1, -1)
                    total = np.vstack((col1, col2, col3))

                else:
                    col1, col2 = data
                    col1 = np.stack(col1)
                    col2 = np.array(col2).reshape(1, -1)
                    total = np.vstack((col1, col2))

                col = total.shape[0]

                if top_k == -1:
                    Total = np.vstack((Total, total))

                if isinstance(top_k, int):
                    if top_k <= 0:
                        raise ValueError("top_k > 0!")
                    Total = np.vstack((Total, total[:top_k, :]))

                elif isinstance(top_k, float):
                    if top_k <= 0 or top_k > 1:
                        raise ValueError("top_k in (0, 1]!")
                    Total = np.vstack((Total, total[:int(col * top_k), :]))

            return Total[1:, :]

    return f


def main(modal_decomp: Union[None, Callable], name: str):
    """
    接收give_data给出数据矩阵，以第二行为内生变量，接下来用CEFFD、EFD、CEEMDAN进行分解，然后分别训练得到结果
    :return:
    """

    this_time = datetime.now()

    name = f"{name}-TimeXer"
    printc(f"正在训练: {name}", color="green")
    memory("开始")

    # ========== 参数设置（与单模型对齐） ==========
    SEQ_LEN = 96                      # 从 48 调整为 96
    TRAIN_RATIO = 0.7                  # 训练集比例
    VAL_RATIO = 0.1                     # 验证集比例（从原训练集中分出）
    TEST_RATIO = 0.2                    # 测试集比例
    EPOCHS = 200                        # 最大训练轮数
    BATCH_SIZE = 32                     # 从 16 调整为 32
    LR = 2e-3                           # 从 0.001 调整为 2e-3

    print("Loading data...\n")
    data_matrix, time_axis, _, _ = Metrix_Create()

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

    # ========== 根据要求：内生变量使用第二行（索引 1） ==========
    endogenous = resampled_matrix[1, :]  # 第二行作为目标变量
    # 外生变量：除第二行外的所有行（包括第一行和其他行）
    exogenous = np.vstack([resampled_matrix[0, :], resampled_matrix[2:, :]]) \
        if resampled_matrix.shape[0] > 2 else resampled_matrix[0:1, :]

    # 模态分解（仅对外生变量进行）
    if modal_decomp is not None:
        exogenous = modal_decomp(exogenous)

    memory("已构建内外生变量完毕")

    # 重新排列特征：将所有外生变量放在前面，内生变量放在最后一列
    resampled_matrix_reordered = np.vstack([exogenous, endogenous])  # (num_series, time_len)
    num_series = resampled_matrix_reordered.shape[0]

    # ========== 构建滑动窗口样本 ==========
    X_seq = []
    y = []
    for t in range(SEQ_LEN, resampled_matrix_reordered.shape[1]):
        X_seq.append(resampled_matrix_reordered[:, t - SEQ_LEN:t].T)  # (seq_len, num_series)
        y.append(endogenous[t])  # 目标值仍是内生变量的原始值
    X_seq = np.array(X_seq)  # (样本数, seq_len, num_series)
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
    X_train_reshaped = X_train.reshape(-1, nf)  # (样本数*seq_len, 特征数)
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

    # ========== 构建时间特征（与单模型一致，加入正弦/余弦编码） ==========
    # 生成全局时间特征（与重采样后的时间轴对齐）
    time_df = {
        'dayofweek': resampled_time.dayofweek,
        'month': resampled_time.month,
        'day': resampled_time.day,
    }
    time_df['sin_dayofweek'] = np.sin(2 * np.pi * time_df['dayofweek'] / 7)
    time_df['cos_dayofweek'] = np.cos(2 * np.pi * time_df['dayofweek'] / 7)
    time_df['sin_month'] = np.sin(2 * np.pi * (time_df['month'] - 1) / 12)
    time_df['cos_month'] = np.cos(2 * np.pi * (time_df['month'] - 1) / 12)

    time_df = pd.DataFrame(time_df)

    # 标准化时间特征（避免数值范围影响）
    time_features_raw = (time_df - time_df.mean()) / time_df.std()
    time_features_raw = time_features_raw.values   # (total_len, num_time_features) 此时为 7 维

    # 构建滑动窗口的时间特征窗口
    T_seq = []
    for t in range(SEQ_LEN, len(time_features_raw)):
        T_seq.append(time_features_raw[t - SEQ_LEN:t])   # (seq_len, num_time_features)
    T_seq = np.array(T_seq)  # (样本数, seq_len, num_time_features)

    # 划分时间特征（与 X 保持一致）
    T_train, T_val, T_test = T_seq[:train_end], T_seq[train_end:val_end], T_seq[val_end:]

    results = {}
    predictions = {}

    print(f"\nTraining {name}...")
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

    # 检查预测是否有效
    if y_pred_scaled is None or np.any(~np.isfinite(y_pred_scaled)):
        print(f"  Warning: {name} 预测结果包含无效值，跳过评估。")
        results[name] = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
        predictions[name] = y_pred_scaled

    # 反标准化得到原始尺度的预测值
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # 计算指标（使用原始尺度的 y_test 和 y_pred）
    mse, rmse, mae, mape = compute_metrics(y_test, y_pred)
    results[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    predictions[name] = y_pred
    print(f"{name} done. MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

    now_time = datetime.now()

    return results, predictions, y_test, now_time - this_time

def run(top_k: Union[int, float]):
    names = ["ceemdan", "ceefd", "efd", "pure"]

    ceemdan_results, ceemdan_prediction, y_test, ceemdan_delta_time = main(Modal("ceemdan", top_k), "ceemdan")
    ceefd_results, ceefd_prediction, _, ceefd_delta_time = main(Modal("ceefd", top_k), "ceefd")
    efd_results, efd_prediction, _, efd_delta_time = main(Modal("efd", top_k), "efd")
    pure_results, pure_prediction, _, pure_delta_time = main(None, "pure")

    delta_times = [ceemdan_delta_time, ceefd_delta_time, efd_delta_time, pure_delta_time]
    for idx, n in enumerate(names):
        temp_name = f"{n}_deltatime.pkl"
        with open(temp_name, "wb") as f:
            pickle.dump(delta_times[idx], f)

    colors = ["#60966D", "#FFC839", "#63ADEE"]
    all_results = [ceemdan_results, ceefd_results, efd_results]
    all_prediction = [ceemdan_prediction, ceefd_prediction, efd_prediction]

    plt.figure(figsize=(12, 6))
    plt.plot(y_test[-200:], label='True', color='black', linewidth=2)

    # colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for prediction in all_prediction:
        for i, (name, y_pred) in enumerate(prediction.items()):
            if y_pred is not None and not np.all(np.isnan(y_pred)):
                plt.plot(y_pred[-200:], label=name, color=colors[i], linestyle='--', alpha=0.7, linewidth=0.8)

    plt.legend()
    plt.title('Test Set Predictions')
    plt.xlabel('Time Step')
    plt.ylabel('GDEA Price')
    plt.tight_layout()
    plt.savefig("各模态对比.svg", dpi=300)
    plt.show()

    print(f"CEEMDAN | {ceemdan_results}")
    print(f"CEEFD | {ceefd_results}")
    print(f"EFD | {efd_results}")
    print(f"Pure | {pure_results}")


if __name__ == '__main__':
    run(10)