# src/data_loader.py
import os
from typing import List
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

os.makedirs(RAW_DIR, exist_ok=True)


def get_universe() -> List[str]:
    """
    Return a list of tickers for the equity universe.
    For now, this can be a manually curated list of large-cap US stocks.
    Later: load from CSV or web-scraped S&P 500 list.
    """
    # TODO: replace with your actual universe
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "JPM", "XOM", "UNH", "V", "HD", "PG", "MA", "JNJ", "PFE",
        "DIS", "BAC", "VZ", "ADBE", "CMCSA", "NFLX"
        # ... extend to ~100–200 names
    ]
    # Include SPY as benchmark
    if "SPY" not in tickers:
        tickers.append("SPY")
    return tickers


def download_prices(
    tickers: List[str],
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download adjusted OHLCV data for the given tickers using yfinance.
    Returns a multi-index DataFrame with (date, ticker).
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )

    # Normalize to long format: index = date, columns: ticker, [Open, High, Low, Close, Volume]
    records = []
    for ticker in tickers:
        if ticker not in data:
            continue
        df_t = data[ticker].copy()
        df_t["Ticker"] = ticker
        df_t.index.name = "Date"
        records.append(df_t.reset_index())

    full_df = pd.concat(records, axis=0, ignore_index=True)
    full_df = full_df.set_index(["Date", "Ticker"]).sort_index()
    return full_df


def save_raw_prices(df: pd.DataFrame, filename: str = "prices_2015_2024.parquet") -> str:
    path = os.path.join(RAW_DIR, filename)
    df.to_parquet(path)
    print(f"Saved raw price data to {path}")
    return path


if __name__ == "__main__":
    universe = get_universe()
    df = download_prices(universe)
    save_raw_prices(df)