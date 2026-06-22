# Phase 4 — Research-Grade Validation Engine

This patch moves the project from dashboard-style evaluation toward a serious research platform.

## What is included

1. **Train-only scaler fix** in `src/preprocessing.py`
   - Feature scaler is fitted only on the training split.
   - Target scaler is fitted only on the training split.
   - Validation/test are transformed using train-fitted scalers.
   - This avoids future-distribution leakage from scaling.

2. **Research validation module** in `src/research_validation.py`
   - Model trust score
   - Model vs naive baseline comparison
   - Long-only risk-aware model scoring
   - Regime-wise performance: bull/bear/sideways and volatility regimes
   - Leakage/alignment audit
   - Walk-forward validation for selected models

3. **New Streamlit page**
   - `🧪 Research Validation`
   - Trust Score tab
   - Regime Performance tab
   - Walk-Forward tab
   - Leakage Audit tab

## Important expectation

After train-only scaling, model results can change again. This is good. The goal is not fake high accuracy; the goal is a serious system that survives honest validation.

## Apply safely

Backup first:

```powershell
Copy-Item app.py app_before_phase4_research_validation.py -Force
Copy-Item src\preprocessing.py src\preprocessing_before_phase4_research_validation.py -Force
```

Copy files from the patch ZIP, then run:

```powershell
python -m py_compile app.py src\preprocessing.py src\research_validation.py
$env:PYTHONPATH = (Get-Location).Path
python tests\test_phase4_research_validation.py
```

Then restart Streamlit and retrain models.
