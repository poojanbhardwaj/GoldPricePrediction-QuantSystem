# GoldPricePrediction Quant Upgrade Pack

This ZIP is a safe drop-in upgrade pack for your existing `GoldPricePrediction` project.

It does **not** replace your working project.  
It adds modular files that you can copy into your project step by step.

## What this ZIP adds

### Core modules

Copy these into your existing `src/` folder:

- `src/asset_config.py`
- `src/prediction_ranges.py`
- `src/signals.py`
- `src/backtesting.py`
- `src/risk.py`
- `src/data_quality.py`
- `src/benchmarks.py`
- `src/walk_forward.py`
- `src/direct_forecast.py`
- `src/portfolio.py`
- `src/explainability.py`

### Streamlit integration

Copy the folder:

- `streamlit_integration/`

or copy the functions from it into your existing `app.py`.

## Most important first step

Start with these 3 files:

```bash
src/prediction_ranges.py
src/signals.py
src/backtesting.py
```

Then test:

```bash
python -m src.prediction_ranges
python -m src.signals
python -m src.backtesting
```

## Full test

After copying all files into your project root:

```bash
python tests/test_phase1_modules.py
```

## How to integrate prediction range

In `src/predict.py` add:

```python
from src.prediction_ranges import calculate_prediction_range
from src.signals import generate_trading_signal
```

After your current prediction code calculates:

```python
last_price
predicted_price
rmse
model_used
```

add:

```python
prediction_range = calculate_prediction_range(
    last_price=last_price,
    predicted_price=predicted_price,
    rmse=rmse,
    confidence_level=0.68,
    model_used=model_used,
)

trading_signal = generate_trading_signal(
    predicted_return_pct=prediction_range.predicted_return_pct,
    lower_return_pct=prediction_range.lower_return_pct,
    upper_return_pct=prediction_range.upper_return_pct,
)
```

Return this result dictionary:

```python
result = {
    "last_known_price": prediction_range.last_price,
    "predicted_next_day_price": prediction_range.predicted_price,
    "predicted_return_pct": prediction_range.predicted_return_pct,
    "expected_lower_bound": prediction_range.lower_bound,
    "expected_upper_bound": prediction_range.upper_bound,
    "confidence_level": prediction_range.confidence_level,
    "error_used": prediction_range.error_used,
    "error_source": prediction_range.error_source,
    "model_used": prediction_range.model_used,
    "signal": trading_signal.signal,
    "signal_confidence": trading_signal.confidence_label,
    "confidence_score": trading_signal.confidence_score,
    "risk_label": trading_signal.risk_label,
    "signal_explanation": trading_signal.explanation,
}
```

## How to integrate backtesting

Your backtesting page expects:

```python
st.session_state["backtest_df"]
```

with columns like:

```text
Gold_Close
Predicted_Price
```

or:

```text
Gold_Close
Predicted_Return
```

Example:

```python
backtest_df = pd.DataFrame(
    {
        target_col: y_test,
        "Predicted_Price": y_pred,
    },
    index=pd.to_datetime(test_dates),
)

st.session_state["backtest_df"] = backtest_df
```

## Important safety rule

The app should always show this disclaimer:

> This system is for educational and research purposes only. Predictions are estimates and should not be considered financial advice.

## Recommended development order

1. Add prediction range.
2. Add trading signals.
3. Add backtesting.
4. Add multi-asset selector.
5. Add benchmark comparison.
6. Add walk-forward validation.
7. Add direct 1-day, 7-day, 30-day forecast targets.
8. Add risk report.
9. Add portfolio allocation.
10. Add explainability and data quality pages.

## Notes

- This pack avoids data leakage by keeping target-generation utilities separate.
- Do not include future target columns as features.
- Use time-series-safe train/test split only.
- Do not shuffle financial time-series data.
- Use test RMSE or validation RMSE for prediction range, not training RMSE.
