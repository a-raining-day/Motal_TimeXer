import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import copy
from tqdm import tqdm

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out) + self.bias
        return out.squeeze()

def train_and_predict(X_train, y_train, X_val, y_val, X_test, y_test, seq_len, num_series,
                      time_features_train=None, time_features_val=None, time_features_test=None,
                      epochs=200, lr=2e-3, batch_size=32, patience=30,
                      device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    改进后的 LSTM 训练函数，支持验证集和时间特征，并包含偏差校正。
    """
    input_size = num_series
    # 如果提供了时间特征，则拼接
    if time_features_train is not None:
        # 假设时间特征形状：(N, seq_len, n_time_feat)
        X_train = np.concatenate([X_train, time_features_train], axis=-1)
        X_val = np.concatenate([X_val, time_features_val], axis=-1)
        X_test = np.concatenate([X_test, time_features_test], axis=-1)
        input_size += time_features_train.shape[-1]

    # 转换为 tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # 构建训练 DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = LSTMModel(input_size=input_size).to(device)

    # 损失函数、优化器、调度器
    # criterion = nn.HuberLoss(delta=1.0)
    criterion = nn.MSELoss()  # r: effect: No.2
    # criterion = nn.SmoothL1Loss()
    # criterion = nn.L1Loss()  # r: effect: No.1

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in tqdm(range(epochs)):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
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
                print(f"LSTM early stopping at epoch {epoch+1}")
                break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 计算训练集偏差（标准化空间）
    model.eval()
    with torch.no_grad():
        train_pred_scaled = model(X_train_t).cpu().numpy()
    train_bias_scaled = np.mean(train_pred_scaled - y_train)

    # 测试预测并修正偏差
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).cpu().numpy()
    y_pred_scaled_corrected = y_pred_scaled - train_bias_scaled

    return y_pred_scaled_corrected