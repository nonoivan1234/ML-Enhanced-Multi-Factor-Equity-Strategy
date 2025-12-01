# src/backtest.py
import pandas as pd
import numpy as np


def run_backtest(
    df: pd.DataFrame,
    top_quantile: float = 0.2,
    transaction_cost_bps: float = 20.0,
    benchmark_ticker: str = "SPY",
) -> pd.DataFrame:
    """
    df: multi-index (Date, Ticker) with columns:
        - PredProb
        - Return_1w
    """
    df = df.copy()

    # Rank by prediction within each week
    df["Rank"] = df.groupby("Date")["PredProb"].rank(ascending=False, method="first")
    df["N"] = df.groupby("Date")["PredProb"].transform("count")
    df["Quantile"] = df["Rank"] / df["N"]

    # Select top quantile
    df["InPortfolio"] = (df["Quantile"] <= top_quantile).astype(int)

    # Equal-weight among selected names
    df["Weight"] = df["InPortfolio"] / df.groupby("Date")["InPortfolio"].transform("sum").replace(0, np.nan)

    # Prevent division by zero
    df["Weight"] = df["Weight"].fillna(0.0)

    # Gross portfolio return per week
    weekly_port_ret = (df["Weight"] * df["Return_1w"]).groupby("Date").sum()

    # Transaction cost approximation via turnover:
    # turnover_t = 0.5 * sum(|w_t - w_{t-1}|) over names
    df["PrevWeight"] = df.groupby("Ticker")["Weight"].shift(1).fillna(0.0)
    df["Turnover"] = (df["Weight"] - df["PrevWeight"]).abs()
    weekly_turnover = 0.5 * df["Turnover"].groupby("Date").sum()

    tc = weekly_turnover * (transaction_cost_bps / 10000.0)
    net_port_ret = weekly_port_ret - tc

    # Benchmark (SPY)
    bench = (
        df.xs(benchmark_ticker, level="Ticker")["Return_1w"]
        .rename("BenchRet_1w")
        .to_frame()
    )

    out = pd.DataFrame({"PortRet_1w": net_port_ret}).join(bench, how="left")
    out["ExcessRet_1w"] = out["PortRet_1w"] - out["BenchRet_1w"]
    return out