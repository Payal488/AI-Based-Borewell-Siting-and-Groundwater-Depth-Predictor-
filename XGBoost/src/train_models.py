import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier

from src.data_preprocessing import load_and_preprocess_data, FEATURE_COLS

def train_and_save_models(data_path="data/groundwater_data.csv", models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    X, y_depth, y_success, _ = load_and_preprocess_data(data_path)
    X_tr, X_te, yd_tr, yd_te, ys_tr, ys_te = train_test_split(X, y_depth, y_success, test_size=0.2, random_state=42)

    reg = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
    reg.fit(X_tr, yd_tr)
    reg_preds = reg.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(yd_te, reg_preds)))
    mae = float(mean_absolute_error(yd_te, reg_preds))
    r2 = float(r2_score(yd_te, reg_preds))

    clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.08, random_state=42, eval_metric="logloss")
    clf.fit(X_tr, ys_tr)
    clf_preds = clf.predict(X_te)
    clf_probs = clf.predict_proba(X_te)[:, 1]
    acc = float(accuracy_score(ys_te, clf_preds))
    prec = float(precision_score(ys_te, clf_preds, zero_division=0))
    rec = float(recall_score(ys_te, clf_preds, zero_division=0))
    f1 = float(f1_score(ys_te, clf_preds, zero_division=0))
    try:
        roc_auc = float(roc_auc_score(ys_te, clf_probs))
    except Exception:
        roc_auc = 0.5

    joblib.dump(reg, os.path.join(models_dir, "depth_regressor.joblib"))
    joblib.dump(clf, os.path.join(models_dir, "success_classifier.joblib"))
    
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump({
            "rmse": round(rmse, 3), "mae": round(mae, 3), "r2": round(r2, 3),
            "accuracy": round(acc, 3), "f1_score": round(f1, 3), "roc_auc": round(roc_auc, 3)
        }, f, indent=2)

    print(f"Models trained successfully. RMSE: {rmse:.2f}m, Acc: {acc*100:.1f}%")

if __name__ == "__main__":
    train_and_save_models()
