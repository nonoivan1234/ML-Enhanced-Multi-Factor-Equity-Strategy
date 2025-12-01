# src/walk_forward.py
import os
from typing import List, Dict
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split

from .model_training import prepare_xy, train_lgbm, FEATURE_COLS
from .backtest import run_backtest
from .performance_metrics import summarize_performance

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PROC_DIR = os.path.join(DATA_DIR, "processed")


def load_labeled(
    filename: str = "features_labels_weekly.parquet",
) -> pd.DataFrame:
    path = os.path.join(PROC_DIR, filename)
    return pd.read_parquet(path)


def walk_forward_backtest(
    df: pd.DataFrame,
    start_year: int = 2018,
    end_year: int = 2024,
) -> pd.DataFrame:
    """
    Simple walk-forward:
    - For each year Y in [start_year, end_year]:
      - train on data up to Y-1
      - predict + backtest on year Y
    """
    all_results = []

    for year in range(start_year, end_year + 1):
        train_mask = df.index.get_level_values("Date").year < year
        test_mask = df.index.get_level_values("Date").year == year

        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()

        if df_train.empty or df_test.empty:
            continue

        # Prepare X, y
        X_train, y_train = prepare_xy(df_train)
        X_test, y_test = prepare_xy(df_test)

        # To avoid data leakage, use last part of train as validation
        # (e.g. 20% of train period) – here we just random split, but
        # you can replace with time-based split later.
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=False
        )

        model, metrics = train_lgbm(X_tr, y_tr, X_val, y_val)
        print(f"Year {year} val AUC: {metrics['val_auc']:.4f}")

        # Predict probabilities on test year (for ranking)
        df_test = df_test.dropna(subset=FEATURE_COLS + ["Label"]).copy()
        X_test = df_test[FEATURE_COLS]
        df_test["PredProb"] = model.predict_proba(X_test)[:, 1]

        # Run backtest for this year
        year_bt = run_backtest(df_test)
        year_bt["Year"] = year
        all_results.append(year_bt)

    results = pd.concat(all_results, axis=0).sort_index()
    return results


if __name__ == "__main__":
    df = load_labeled()
    results = walk_forward_backtest(df, start_year=2018, end_year=2024)
    summarize_performance(results)