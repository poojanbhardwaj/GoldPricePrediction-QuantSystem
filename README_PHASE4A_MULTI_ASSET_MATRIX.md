# Phase 4A — Multi-Asset Research Validation Matrix

This patch upgrades Phase 4 from single-asset validation to an all-asset validation matrix.

## Why this exists

A serious financial intelligence platform cannot validate only Gold. Every configured asset must be checked separately because each market has different behavior, volatility, liquidity, and predictability.

This patch adds:

- `src/multiasset_validation.py`
- Streamlit page: `🌐 Multi-Asset Matrix`
- test: `tests/test_phase4A_multiasset_matrix.py`

## What the matrix reports

For every selected asset:

- Best model
- Conservative trust score
- Verdict
- RMSE vs naive baseline
- Directional accuracy
- Long-only backtest Sharpe
- Max drawdown
- Strategy vs buy-and-hold
- Leakage/alignment audit
- Optional walk-forward summary

## Model depth options

- `fast`: Linear Regression + Decision Tree. Good for checking the pipeline.
- `core`: Linear Regression + Random Forest + XGBoost + LightGBM + CatBoost. Recommended for serious runs.
- `full`: all ML models including SVR. Slowest.

## Usage

1. Backup `app.py`.
2. Copy `src/multiasset_validation.py` and patched `app.py`.
3. Run syntax check.
4. Run test.
5. Open Streamlit and use `🌐 Multi-Asset Matrix`.

This patch does not claim all assets are predictable. It tells the truth asset by asset.
