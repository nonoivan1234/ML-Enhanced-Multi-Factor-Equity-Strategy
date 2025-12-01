# src/performance_metrics.py
import numpy as np
import pandas as pd


def annualized_return(weekly_returns: pd.Series) -> float:
    mean_weekly = weekly_returns.mean()
    return (1 + mean_weekly) ** 52 - 1


def annualized_vol(weekly_returns: pd.Series) -> float:
    return weekly_returns.std() * np.sqrt(52)


def sharpe_ratio(weekly_returns: pd.Series, rf: float = 0.0) -> float:
    # rf = 0 assumed weekly
    excess = weekly_returns - rf / 52
    vol = annualized_vol(excess)
    if vol == 0:
        return np.nan
    return annualized_return(excess) / vol


def max_drawdown(equity_curve: pd.Series) -> float:
    cum_max = equity_curve.cummax()
    dd = equity_curve / cum_max - 1
    return dd.min()


def summarize_performance(results: pd.DataFrame):
    """
    results: index=Date, columns: PortRet_1w, BenchRet_1w, ExcessRet_1w
    """
    port_ret = results["PortRet_1w"].dropna()
    bench_ret = results["BenchRet_1w"].dropna()

    eq_port = (1 + port_ret).cumprod()
    eq_bench = (1 + bench_ret).cumprod()

    stats = {
        "Port_AnnRet": annualized_return(port_ret),
        "Port_AnnVol": annualized_vol(port_ret),
        "Port_Sharpe": sharpe_ratio(port_ret),
        "Port_MaxDD": max_drawdown(eq_port),
        "Bench_AnnRet": annualized_return(bench_ret),
        "Bench_AnnVol": annualized_vol(bench_ret),
        "Bench_Sharpe": sharpe_ratio(bench_ret),
        "Bench_MaxDD": max_drawdown(eq_bench),
    }
    print("=== Performance Summary ===")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")
    return stats