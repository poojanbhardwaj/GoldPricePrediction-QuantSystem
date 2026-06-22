# Phase 5 — Better Feature Intelligence

This phase adds market-aware, multi-asset feature engineering to the project.
It does **not** fake better results. It gives the models more useful market context, then Phase 4A validates honestly whether the new features actually help.

## What is added

### New module

- `src/feature_intelligence.py`

### New Streamlit page

- `🧠 Feature Intelligence`

### Updated files

- `app.py`
- `src/multiasset_validation.py`

### New test

- `tests/test_phase5_feature_intelligence.py`

## New feature families

All new features start with `FI_`.

- `FI_Target_*` — target momentum, volatility, drawdown, breakout/breakdown pressure
- `FI_Asset_*` — returns and volatility for each asset
- `FI_Cross_*` — rolling correlation, beta, relative strength across assets
- `FI_Macro_*` — DXY, VIX, TNX, S&P 500 risk pressure
- `FI_Regime_*` — high-volatility, trend, above-SMA, risk-off regimes

## Leakage rule

These features use only current and past information at timestamp `t`.
They do not use `shift(-1)` or any future return. They are valid for the project's next-day target:

```text
log(price[t+1] / price[t])
```

## Expected result

This phase may not instantly turn models into High Trust.
The correct process is:

1. Add better features.
2. Run Phase 4A Multi-Asset Matrix again.
3. Compare trust score, RMSE-vs-naive, directional accuracy, and overfit gap.
4. Keep only what improves honestly.

## How to run after applying

```powershell
python -m py_compile app.py src\feature_intelligence.py src\multiasset_validation.py
$env:PYTHONPATH = (Get-Location).Path
python tests\test_phase5_feature_intelligence.py
streamlit cache clear
streamlit run app.py
```

Then open:

```text
🧠 Feature Intelligence
```

Run the audit.

Then open:

```text
🌐 Multi-Asset Matrix
```

Use:

```text
Assets: all selected
Model depth: fast first
Use Phase 5 feature intelligence: ON
Walk-forward: OFF first
```

Then run:

```text
Assets: all selected
Model depth: core
Use Phase 5 feature intelligence: ON
Walk-forward: ON for serious validation
```
