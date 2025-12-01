# ML-Enhanced Multi-Factor Equity Strategy  
2015–2024 · Weekly Data · Python

---

## 1. Summary

A reproducible Python implementation of a multi-factor equity strategy where a LightGBM classifier ranks large-cap US stocks by probability of outperforming SPY over the next 4 weeks. The goal is to evaluate whether ML-based factor weighting yields incremental value versus simple baselines, using walk‑forward evaluation and conservative anti-overfitting choices.

---

## 2. Strategy (high level)

- Download daily OHLCV (2015–2024) with `yfinance` and resample to weekly.
- Engineer technical / market features on weekly data.
- Label each week/ticker as 1 if its 4-week forward return > SPY’s 4-week forward return, else 0.
- Train a LightGBM classifier to estimate outperformance probabilities.
- Backtest a weekly-rebalanced, long-only portfolio that buys the top q% by predicted probability (equal weight by default), with realistic transaction-cost adjustments.
- Evaluate versus baselines (SPY, momentum, linear multi-factor).

---

## 3. Data & universe

- Period: 2015-01-01 → 2024-12-31 (weekly)
- Universe: static list of large-cap US tickers (see `get_universe()` in `src/data_loader.py`)
- Benchmark: `SPY`
- Raw data saved to `data/raw/`; processed feature/label tables saved to `data/processed/`.

---

## 4. Features & labels

Features (computed weekly per ticker):

- Momentum: 12–1 week (`Mom_12_1`)
- Volatility: `Vol_4w`, `Vol_12w`
- Moving averages: `SMA_4w`, `SMA_12w` and price/SMA ratios
- RSI(14), ATR(14), weekly returns

Label:

- Label = 1 if stock 4-week forward return − SPY 4-week forward return > threshold (default 0), else 0.
- Label logic: `src/label_construction.py`

---

## 5. Model & training

- Algorithm: LightGBM (binary classification)
- Regularization: shallow trees (max_depth ≈ 3–5), row/column subsampling, small learning rate, early stopping
- Validation metric: AUC (per-year validation)
- Code: `src/model_training.py`

---

## 6. Walk-forward evaluation

- For each test year Y (e.g., 2018–2024):
    - Train on data with year < Y
    - Tune on a held validation slice from the train period
    - Predict and backtest on year Y
- Predictions and yearly backtests are concatenated into a full equity curve.
- Implementation: `src/walk_forward.py`

---

## 7. Portfolio & backtest mechanics

- Weekly ranking by predicted probability; select top q fraction (e.g., 20%).
- Equal-weight positions (v1). Optionally volatility-scale later.
- Weekly portfolio return = weighted sum of constituent weekly returns.
- Turnover computed as 0.5 × Σ|w_t − w_{t−1}|; transaction cost = turnover × cost_bps/10,000.
- Performance metrics: annualized return, volatility, Sharpe, max drawdown, turnover (`src/backtest.py`, `src/performance_metrics.py`).

---

## 8. Project layout
```
quant-ml-factor-model/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── label_construction.py
│   ├── model_training.py
│   ├── walk_forward.py
│   ├── backtest.py
│   ├── performance_metrics.py
│   └── utils.py
├── notebooks/          # EDA & plotting only
├── README.md
├── requirements.txt
└── .gitignore
```
Core logic lives in `src/`. Notebooks are for exploration and visualization only.

---

## 9. Quick start

1) Install:
pip install -r requirements.txt

2) Download raw prices:
python -m src.data_loader

3) Build weekly features:
python -m src.feature_engineering

4) Construct labels:
python -m src.label_construction

5) Run walk‑forward backtest:
python -m src.walk_forward

Outputs: processed tables in `data/processed/` and backtest summaries printed/saved.

---

## 10. Baselines & comparisons

- Buy & hold SPY
- Naive momentum (rank by 12–1 week momentum)
- Linear multi-factor model (logistic/regression on same features)

Aim: measure incremental value of non-linear ML after transaction costs and realistic evaluation.

---

## 11. Extensions (planned)

- Add fundamentals (value, quality)
- Regime-aware models (high/low vol)
- Improved position sizing (volatility scaling, risk parity)
- IC analysis and sensitivity tests (transaction costs, thresholds, holding horizons)
- Broader universes / non-US markets

---

## 12. Disclaimer

Educational / research purposes only — not investment advice. Use at your own risk.

