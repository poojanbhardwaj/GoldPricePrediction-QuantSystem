# src/baselines.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class BaselineResult:
    name: str
    predictions: np.ndarray
    metrics: Dict[str, float]


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _price_metrics(actual_prices: np.ndarray, predicted_prices: np.ndarray, anchors: np.ndarray) -> Dict[str, float]:
    actual = np.asarray(actual_prices, dtype=float).flatten()
    pred = np.asarray(predicted_prices, dtype=float).flatten()
    anchor = np.asarray(anchors, dtype=float).flatten()

    mask = np.isfinite(actual) & np.isfinite(pred) & np.isfinite(anchor) & (actual != 0) & (anchor > 0)
    if mask.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan, "DirectionalAccuracy": np.nan}

    actual = actual[mask]
    pred = pred[mask]
    anchor = anchor[mask]

    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100.0)
    r2 = float(_safe_r2(actual, pred))

    actual_dir = actual > anchor
    pred_dir = pred > anchor
    directional_accuracy = float(np.mean(actual_dir == pred_dir) * 100.0)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "R2": round(r2, 4),
        "DirectionalAccuracy": round(directional_accuracy, 2),
    }


def _anchors_from_preprocessed(data) -> np.ndarray:
    actual_prices = np.asarray(data.prices_test, dtype=float).flatten()
    if len(actual_prices) == 0:
        return np.array([])
    first_anchor = float(data.last_price_before_test)
    return np.concatenate([[first_anchor], actual_prices[:-1]])


def _rolling_mean_prediction(anchors: np.ndarray, window: int) -> np.ndarray:
    anchors = np.asarray(anchors, dtype=float).flatten()
    preds = np.empty(len(anchors), dtype=float)
    for i in range(len(anchors)):
        start = max(0, i - window + 1)
        preds[i] = float(np.mean(anchors[start : i + 1]))
    return preds


def _momentum_prediction(anchors: np.ndarray) -> np.ndarray:
    anchors = np.asarray(anchors, dtype=float).flatten()
    if len(anchors) == 0:
        return anchors
    preds = anchors.copy()
    for i in range(1, len(anchors)):
        prev = anchors[i - 1]
        if np.isfinite(prev) and prev > 0:
            last_return = anchors[i] / prev - 1.0
        else:
            last_return = 0.0
        preds[i] = anchors[i] * (1.0 + last_return)
    preds[0] = anchors[0]
    return preds


def price_baseline_results(data) -> List[BaselineResult]:
    """
    Build realistic one-step-ahead baseline predictions on the held-out test set.

    actual_prices are true next-day prices P[t+1].
    anchors are known prices P[t] available when the prediction is made.
    """
    actual_prices = np.asarray(data.prices_test, dtype=float).flatten()
    anchors = _anchors_from_preprocessed(data)

    if len(actual_prices) != len(anchors):
        raise ValueError("prices_test and anchor arrays have different lengths.")

    baseline_specs = [
        ("Naive: tomorrow = today", anchors),
        ("MA-5 price baseline", _rolling_mean_prediction(anchors, 5)),
        ("MA-20 price baseline", _rolling_mean_prediction(anchors, 20)),
        ("1-day momentum baseline", _momentum_prediction(anchors)),
    ]

    out: List[BaselineResult] = []
    for name, pred in baseline_specs:
        out.append(
            BaselineResult(
                name=name,
                predictions=np.asarray(pred, dtype=float),
                metrics=_price_metrics(actual_prices, pred, anchors),
            )
        )
    return out


def price_baseline_leaderboard(data) -> pd.DataFrame:
    rows = []
    for result in price_baseline_results(data):
        row = {"Model": result.name}
        row.update(result.metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty and "RMSE" in df.columns:
        df = df.sort_values("RMSE", ascending=True).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def model_vs_naive_summary(model_board: pd.DataFrame, baseline_board: pd.DataFrame) -> Dict[str, float | str]:
    """Return a compact comparison of best trained model against naive baseline."""
    if model_board is None or model_board.empty or baseline_board is None or baseline_board.empty:
        return {}

    best_model_row = model_board.sort_values("RMSE", ascending=True).iloc[0]
    naive_rows = baseline_board[baseline_board["Model"].str.contains("Naive", case=False, na=False)]
    if naive_rows.empty:
        return {}
    naive_row = naive_rows.iloc[0]

    best_rmse = float(best_model_row["RMSE"])
    naive_rmse = float(naive_row["RMSE"])
    improvement = (naive_rmse - best_rmse) / naive_rmse * 100.0 if naive_rmse > 0 else np.nan

    return {
        "best_model": str(best_model_row["Model"]),
        "best_model_rmse": round(best_rmse, 4),
        "naive_rmse": round(naive_rmse, 4),
        "rmse_improvement_pct": round(float(improvement), 2),
    }
