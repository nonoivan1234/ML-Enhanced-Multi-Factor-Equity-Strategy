# src/label_construction.py
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PROC_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(PROC_DIR, exist_ok=True)


def load_features(filename: str = "features_weekly.parquet") -> pd.DataFrame:
    path = os.path.join(PROC_DIR, filename)
    return pd.read_parquet(path)


def compute_forward_returns(
    feat: pd.DataFrame,
    horizon_weeks: int = 4,
) -> pd.DataFrame:
    """
    Compute forward returns over next horizon_weeks for each ticker.
    """
    feat = feat.copy()
    feat["FwdRet_4w"] = (
        feat.groupby(level="Ticker")["Close"]
        .shift(-horizon_weeks) / feat["Close"] - 1.0
    )
    return feat


def construct_labels_binary(
    feat: pd.DataFrame,
    benchmark_ticker: str = "SPY",
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Label = 1 if stock outperforms SPY by > threshold over next 4 weeks.
    """
    feat = feat.copy()
    # extract benchmark forward returns
    bench = (
        feat.xs(benchmark_ticker, level="Ticker")[["FwdRet_4w"]]
        .rename(columns={"FwdRet_4w": "Bench_FwdRet_4w"})
    )
    # align benchmark to all tickers by date
    feat = feat.join(bench, on="Date")
    feat["ExcessFwdRet_4w"] = feat["FwdRet_4w"] - feat["Bench_FwdRet_4w"]
    feat["Label"] = (feat["ExcessFwdRet_4w"] > threshold).astype(int)
    return feat


def save_labeled(
    df: pd.DataFrame,
    filename: str = "features_labels_weekly.parquet",
) -> str:
    path = os.path.join(PROC_DIR, filename)
    df.to_parquet(path)
    print(f"Saved features + labels to {path}")
    return path


if __name__ == "__main__":
    feat = load_features()
    feat = compute_forward_returns(feat, horizon_weeks=4)
    labeled = construct_labels_binary(feat, benchmark_ticker="SPY", threshold=0.0)
    save_labeled(labeled)