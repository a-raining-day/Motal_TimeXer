import pandas as pd
import numpy as np
from prophet import Prophet

def train_and_predict(y_train, y_val, y_test,
                                exog_train=None, exog_val=None, exog_test=None,
                                dates_train=None, dates_val=None, dates_test=None,
                                prophet_params=None,
                                scaler_y=None):  # 传入 scaler 用于反标准化
    if prophet_params is None:
        prophet_params = {}

    # 如果提供了 scaler，将 y 还原为原始尺度
    if scaler_y is not None:
        y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
        y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()
    else:
        y_train_orig = y_train
        y_val_orig = y_val

    y_train_full = np.concatenate([y_train_orig, y_val_orig])
    dates_train_full = dates_train + dates_val

    df_train = pd.DataFrame({'ds': dates_train_full, 'y': y_train_full})

    # 选择少量外生变量（例如只选前5个相关性最高的，此处简化）
    if exog_train is not None and exog_val is not None:
        exog_train_full = np.concatenate([exog_train, exog_val], axis=0)
        # 简单起见，只取前5个
        n_exog_use = min(5, exog_train_full.shape[1])
        for i in range(n_exog_use):
            df_train[f'regressor_{i}'] = exog_train_full[:, i]

    model = Prophet(**prophet_params)
    for col in df_train.columns:
        if col.startswith('regressor_'):
            model.add_regressor(col)
    model.fit(df_train)

    df_test = pd.DataFrame({'ds': dates_test})
    if exog_test is not None:
        for i in range(n_exog_use):
            df_test[f'regressor_{i}'] = exog_test[:, i]

    forecast = model.predict(df_test)
    y_pred_orig = forecast['yhat'].values

    # 将预测值标准化回原尺度（用于与其他模型比较）
    if scaler_y is not None:
        y_pred_scaled = scaler_y.transform(y_pred_orig.reshape(-1, 1)).ravel()
    else:
        y_pred_scaled = y_pred_orig

    return y_pred_scaled