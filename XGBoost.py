import xgboost as xgb
import numpy as np

def train_and_predict(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=13,
        learning_rate=0.05,
        random_state=42,
        verbosity=0,
        min_child_weight=5,
        alpha=3,
        gamma=0.8,
        subsample=0.7
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred