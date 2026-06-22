# src/benchmarks.py

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        raise ValueError("No valid values for metric calculation.")

    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(errors) / denom) * 100.0)

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot

    directional_accuracy = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100.0) if len(y_true) > 2 else 0.0

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE_pct": round(mape, 4),
        "R2": round(float(r2), 4),
        "Directional_Accuracy_pct": round(directional_accuracy, 4),
    }


def add_baseline_predictions(
    df: pd.DataFrame,
    *,
    price_col: str,
    moving_average_window: int = 20,
) -> pd.DataFrame:
    if price_col not in df.columns:
        raise ValueError(f"Missing price column: {price_col}")

    out = df.copy()
    price = pd.to_numeric(out[price_col], errors="coerce")

    # For next-day prediction, today's close is the naive prediction for tomorrow.
    out["Naive_Prediction"] = price.shift(1)

    # Moving average baseline.
    out["Moving_Average_Prediction"] = price.shift(1).rolling(moving_average_window).mean()

    # Random walk baseline is same as naive close-to-close expectation.
    out["Random_Walk_Prediction"] = price.shift(1)

    return out


def compare_against_baselines(
    df: pd.DataFrame,
    *,
    actual_col: str,
    model_prediction_col: str,
) -> pd.DataFrame:
    required = [
        actual_col,
        model_prediction_col,
        "Naive_Prediction",
        "Moving_Average_Prediction",
        "Random_Walk_Prediction",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rows = []
    for name, col in [
        ("Model", model_prediction_col),
        ("Naive", "Naive_Prediction"),
        ("Moving Average", "Moving_Average_Prediction"),
        ("Random Walk", "Random_Walk_Prediction"),
    ]:
        temp = df[[actual_col, col]].dropna()
        if temp.empty:
            continue
        metrics = regression_metrics(temp[actual_col], temp[col])
        metrics["Model"] = name
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("Model")


if __name__ == "__main__":
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    prices = pd.Series(np.linspace(100, 130, len(dates)) + np.random.default_rng(1).normal(0, 1, len(dates)))
    demo = pd.DataFrame({"Gold_Close": prices.values}, index=dates)
    demo = add_baseline_predictions(demo, price_col="Gold_Close")
    demo["CatBoost_Prediction"] = demo["Gold_Close"].shift(1) * 1.001
    print(compare_against_baselines(demo, actual_col="Gold_Close", model_prediction_col="CatBoost_Prediction"))
