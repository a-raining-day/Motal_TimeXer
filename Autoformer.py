import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import copy
from tqdm import tqdm

# 正确导入官方 Autoformer 模型
from 单模型预测比较.models.Autoformer_main.models import Autoformer
Model = Autoformer.Model

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def train_and_predict(X_train, y_train, X_val, y_val, X_test, y_test,
                      seq_len, num_series,
                      time_features_train=None, time_features_val=None, time_features_test=None,
                      epochs=200, lr=0.001, batch_size=32, patience=20,
                      device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    参数：
        X_train, y_train : 训练集特征（标准化后）和目标值（标准化后）
        X_val, y_val     : 验证集（用于早停）
        X_test, y_test   : 测试集
        seq_len          : 输入窗口长度
        num_series       : 特征总数（包括外生和内生）
        time_features_*  : 时间特征，形状 (N, seq_len, num_time_features) 或 None
        其余参数为训练超参数
    返回：
        y_pred_scaled : 测试集预测值（标准化空间，一维数组）
    """
    # -------------------- 超参数设置 --------------------
    pred_len = 1
    label_len = seq_len // 2          # 解码器已知部分长度
    d_model = 128
    n_heads = 8
    e_layers = 1
    d_layers = 2
    d_ff = 256
    moving_avg = 7
    factor = 1
    dropout = 0.1
    activation = 'gelu'
    embed_type = 'timeF'  # 使用 TimeFeatureEmbedding（线性层）
    freq = 'd'                         # 频率，用于内部时间嵌入，实际我们传入自定义时间特征时此参数影响不大

    # 时间特征维度（如果没有提供时间特征，则使用 4 维零张量）
    if time_features_train is not None:
        mark_dim = time_features_train.shape[-1]
    else:
        mark_dim = 4
        print("Warning: No time features provided, using zeros.")

    configs = Config(
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        output_attention=False,
        moving_avg=moving_avg,
        enc_in=num_series,
        dec_in=num_series,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_layers=d_layers,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        factor=factor,
        embed=embed_type,
        freq=freq,
        c_out=num_series
    )

    # -------------------- 数据准备 --------------------
    # 转换为 tensor 并移至设备
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)  # 仅用于可选的偏差计算，不用于训练

    # 处理时间特征
    if time_features_train is not None:
        T_train_t = torch.tensor(time_features_train, dtype=torch.float32).to(device)
        T_val_t = torch.tensor(time_features_val, dtype=torch.float32).to(device)
        T_test_t = torch.tensor(time_features_test, dtype=torch.float32).to(device)
        has_time_feat = True
    else:
        # 创建全零时间特征
        T_train_t = torch.zeros(X_train_t.size(0), seq_len, mark_dim, device=device)
        T_val_t = torch.zeros(X_val_t.size(0), seq_len, mark_dim, device=device)
        T_test_t = torch.zeros(X_test_t.size(0), seq_len, mark_dim, device=device)
        has_time_feat = False

    # 构造训练数据集：包含特征、时间特征、目标
    train_dataset = TensorDataset(X_train_t, T_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # -------------------- 实例化模型 --------------------
    model = Model(configs).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # -------------------- 训练（带早停） --------------------
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in tqdm(range(epochs), desc="Autoformer Training"):
        model.train()
        train_loss = 0.0
        for batch_x, batch_t, batch_y in train_loader:
            batch_size_curr = batch_x.size(0)

            # 构造解码器输入 x_dec 和对应的时间特征 x_mark_dec
            # x_dec 的前 label_len 个时间步用 batch_x 的最后 label_len 个值，后面 pred_len 个时间步填充 0
            x_dec = torch.zeros(batch_size_curr, label_len + pred_len, num_series, device=device)
            x_dec[:, :label_len, :] = batch_x[:, -label_len:, :]  # 取最后 label_len 个时间步

            # x_mark_dec 的时间特征：前 label_len 个时间步用 batch_t 的最后 label_len 个值，后面 pred_len 个时间步用最后一个时间特征重复
            x_mark_dec = torch.zeros(batch_size_curr, label_len + pred_len, mark_dim, device=device)
            x_mark_dec[:, :label_len, :] = batch_t[:, -label_len:, :]
            # 用最后一个时间特征填充未来部分
            last_time_feat = batch_t[:, -1:, :]  # (batch, 1, mark_dim)
            x_mark_dec[:, label_len:, :] = last_time_feat.repeat(1, pred_len, 1)

            optimizer.zero_grad()
            # Autoformer 前向：输入 (batch_x, x_mark_enc, x_dec, x_mark_dec)
            output = model(batch_x, batch_t, x_dec, x_mark_dec)   # output 形状 (batch, pred_len, num_series)
            pred = output[:, -1, -1]                              # 取最后一步的内生变量预测值
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        with torch.no_grad():
            # 构造验证集的解码器输入
            x_dec_val = torch.zeros(X_val_t.size(0), label_len + pred_len, num_series, device=device)
            x_dec_val[:, :label_len, :] = X_val_t[:, -label_len:, :]
            x_mark_dec_val = torch.zeros(X_val_t.size(0), label_len + pred_len, mark_dim, device=device)
            x_mark_dec_val[:, :label_len, :] = T_val_t[:, -label_len:, :]
            last_time_feat_val = T_val_t[:, -1:, :]
            x_mark_dec_val[:, label_len:, :] = last_time_feat_val.repeat(1, pred_len, 1)

            val_output = model(X_val_t, T_val_t, x_dec_val, x_mark_dec_val)
            val_pred = val_output[:, -1, -1]
            val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step(val_loss)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Autoformer early stopping at epoch {epoch+1}")
                break

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # -------------------- 偏差修正（参考 TimeXer） --------------------
    model.eval()
    with torch.no_grad():
        # 计算训练集上的平均偏差（标准化空间）
        x_dec_train = torch.zeros(X_train_t.size(0), label_len + pred_len, num_series, device=device)
        x_dec_train[:, :label_len, :] = X_train_t[:, -label_len:, :]
        x_mark_dec_train = torch.zeros(X_train_t.size(0), label_len + pred_len, mark_dim, device=device)
        x_mark_dec_train[:, :label_len, :] = T_train_t[:, -label_len:, :]
        last_time_feat_train = T_train_t[:, -1:, :]
        x_mark_dec_train[:, label_len:, :] = last_time_feat_train.repeat(1, pred_len, 1)

        train_output = model(X_train_t, T_train_t, x_dec_train, x_mark_dec_train)
        train_pred_scaled = train_output[:, -1, -1].cpu().numpy()
        train_bias_scaled = np.mean(train_pred_scaled - y_train)

        # 测试集预测
        x_dec_test = torch.zeros(X_test_t.size(0), label_len + pred_len, num_series, device=device)
        x_dec_test[:, :label_len, :] = X_test_t[:, -label_len:, :]
        x_mark_dec_test = torch.zeros(X_test_t.size(0), label_len + pred_len, mark_dim, device=device)
        x_mark_dec_test[:, :label_len, :] = T_test_t[:, -label_len:, :]
        last_time_feat_test = T_test_t[:, -1:, :]
        x_mark_dec_test[:, label_len:, :] = last_time_feat_test.repeat(1, pred_len, 1)

        test_output = model(X_test_t, T_test_t, x_dec_test, x_mark_dec_test)
        y_pred_scaled = test_output[:, -1, -1].cpu().numpy()

    # 应用偏差修正
    y_pred_scaled_corrected = y_pred_scaled - train_bias_scaled
    print(f"Autoformer train bias (scaled): {train_bias_scaled:.6f}")

    return y_pred_scaled_corrected