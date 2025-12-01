# src/feature_engineering.py
import os
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(PROC_DIR, exist_ok=True)


def load_raw_prices(filename: str = "prices_2015_2024.parquet") -> pd.DataFrame:
    path = os.path.join(RAW_DIR, filename)
    return pd.read_parquet(path)


def resample_to_weekly(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily OHLCV to weekly frequency (e.g., Friday close).
    prices index: (Date, Ticker) with columns [Open, High, Low, Close, Volume]
    """
    dfs = []
    for ticker, df_t in prices.groupby(level="Ticker"):
        df_t = df_t.reset_index(level="Ticker", drop=True)
        o = df_t["Open"].resample("W-FRI").first()
        h = df_t["High"].resample("W-FRI").max()
        l = df_t["Low"].resample("W-FRI").min()
        c = df_t["Close"].resample("W-FRI").last()
        v = df_t["Volume"].resample("W-FRI").sum()
        out = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v})
        out["Ticker"] = ticker
        dfs.append(out.reset_index())

    weekly = pd.concat(dfs, axis=0, ignore_index=True)
    weekly = weekly.set_index(["Date", "Ticker"]).sort_index()
    return weekly


def compute_returns(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.copy()
    weekly["Return_1w"] = weekly.groupby("Ticker")["Close"].pct_change()
    return weekly


def add_technical_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators per ticker:
    - 12-1w momentum
    - 4w and 12w volatility
    - Price vs 4w/12w SMA
    - RSI(14)
    - ATR(14)
    """
    dfs = []
    for ticker, df_t in weekly.groupby(level="Ticker"):
        df_t = df_t.reset_index(level="Ticker", drop=True).copy()

        df_t["Ret_1w"] = df_t["Close"].pct_change()

        # Momentum: last price / price 12 weeks ago - 1
        df_t["Mom_12_1"] = df_t["Close"] / df_t["Close"].shift(12) - 1

        # Volatility: rolling std of weekly returns
        df_t["Vol_4w"] = df_t["Ret_1w"].rolling(4).std()
        df_t["Vol_12w"] = df_t["Ret_1w"].rolling(12).std()

        # Moving averages and ratios
        df_t["SMA_4w"] = df_t["Close"].rolling(4).mean()
        df_t["SMA_12w"] = df_t["Close"].rolling(12).mean()
        df_t["Px_SMA4"] = df_t["Close"] / df_t["SMA_4w"]
        df_t["Px_SMA12"] = df_t["Close"] / df_t["SMA_12w"]

        # RSI & ATR (need High / Low / Close)
        rsi = RSIIndicator(close=df_t["Close"], window=14)
        df_t["RSI_14"] = rsi.rsi()

        atr = AverageTrueRange(
            high=df_t["High"],
            low=df_t["Low"],
            close=df_t["Close"],
            window=14,
        )
        df_t["ATR_14"] = atr.average_true_range()

        df_t["Ticker"] = ticker
        dfs.append(df_t.reset_index())

    feat = pd.concat(dfs, axis=0, ignore_index=True)
    feat = feat.set_index(["Date", "Ticker"]).sort_index()
    return feat


def save_features(df: pd.DataFrame, filename: str = "features_weekly.parquet") -> str:
    path = os.path.join(PROC_DIR, filename)
    df.to_parquet(path)
    print(f"Saved features to {path}")
    return path


if __name__ == "__main__":
    raw = load_raw_prices()
    weekly = resample_to_weekly(raw)
    weekly = compute_returns(weekly)
    features = add_technical_features(weekly)
    save_features(features)