# src/model_training.py
from pyexpat import model
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


FEATURE_COLS = [
    "Mom_12_1",
    "Vol_4w",
    "Vol_12w",
    "Px_SMA4",
    "Px_SMA12",
    "RSI_14",
    "ATR_14",
    # 可以之後再加其他 features
]


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.dropna(subset=FEATURE_COLS + ["Label"]).copy()
    X = df[FEATURE_COLS]
    y = df["Label"].astype(int)
    return X, y


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[LGBMClassifier, Dict]:
    params = {
    }
    model = LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc"
    )
    val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    metrics = {"val_auc": val_auc}
    return model, metrics