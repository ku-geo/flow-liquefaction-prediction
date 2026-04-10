import numpy as np


def train_xgb(X_tr, y_tr, cfg, seed):
    from xgboost import XGBRegressor
    xc = cfg["xgboost"]
    model = XGBRegressor(
        n_estimators=xc["n_estimators"], max_depth=xc["max_depth"],
        learning_rate=xc["learning_rate"], subsample=xc["subsample"],
        random_state=seed, verbosity=0, n_jobs=1)
    model.fit(X_tr.reshape(len(X_tr), -1), y_tr)
    return model


def predict_xgb(model, X, thresholds):
    mu = np.maximum(model.predict(X.reshape(len(X), -1)), 0.0).astype(np.float32)
    probs = [(mu <= thr).astype(np.float32) for thr in thresholds]
    return mu, probs
