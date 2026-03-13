from sklearn.ensemble import RandomForestRegressor
import numpy as np

def train_and_predict(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        max_features=0.7
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred