import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import copy
from tqdm import tqdm

from 单模型预测比较.models.Crossformer_master.cross_models.cross_former import Crossformer

def train_and_predict(X_train, y_train, X_val, y_val, X_test, y_test, seq_len, num_series,
                      time_features_train=None, time_features_val=None, time_features_test=None,
                      epochs=200, lr=2e-3, batch_size=32, patience=30,
                      device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 转换为 tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # 构建训练 DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 实例化模型
    model = Crossformer(
        data_dim=num_series,
        in_len=seq_len,
        out_len=1,
        seg_len=8,
        win_size=2,
        factor=1,
        d_model=64,
        d_ff=64,
        n_heads=4,
        e_layers=1,
        dropout=0.2,
        baseline=False,
        device=device
    ).to(device)

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
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)               # [batch, 1, num_series]
            pred = output[:, -1, -1]               # 取最后一个特征
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t)
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
                print(f"Crossformer early stopping at epoch {epoch+1}")
                break

        # 可选打印
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # --- 计算训练集上的平均偏差（标准化空间） ---
    model.eval()
    with torch.no_grad():
        train_output = model(X_train_t)
        train_pred_scaled = train_output[:, -1, -1].cpu().numpy()
    train_bias_scaled = np.mean(train_pred_scaled - y_train)
    print(f"Train bias (scaled): {train_bias_scaled:.6f}")

    # 测试预测
    with torch.no_grad():
        test_output = model(X_test_t)
        y_pred_scaled = test_output[:, -1, -1].cpu().numpy()
        # 修正：减去训练集偏差
        y_pred_scaled_corrected = y_pred_scaled - train_bias_scaled

    return y_pred_scaled_corrected