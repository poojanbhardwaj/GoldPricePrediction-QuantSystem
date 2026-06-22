# src/walk_forward.py

from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


def expanding_window_splits(
    n_samples: int,
    *,
    initial_train_size: int,
    test_size: int,
    step_size: int | None = None,
    max_folds: int | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Time-series-safe expanding-window split.

    Fold 1: train older data, test next period.
    Fold 2: expand train data, test next period.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if initial_train_size <= 0 or test_size <= 0:
        raise ValueError("initial_train_size and test_size must be positive.")

    if initial_train_size + test_size > n_samples:
        raise ValueError("Not enough samples for requested walk-forward split.")

    if step_size is None:
        step_size = test_size

    splits = []
    train_end = initial_train_size

    while train_end + test_size <= n_samples:
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, train_end + test_size)
        splits.append((train_idx, test_idx))

        if max_folds is not None and len(splits) >= max_folds:
            break

        train_end += step_size

    return splits


def fold_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        raise ValueError("No valid fold values.")

    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(err) / denom) * 100.0)

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot

    if len(y_true) > 2:
        directional_accuracy = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100.0)
    else:
        directional_accuracy = 0.0

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE_pct": round(mape, 4),
        "R2": round(float(r2), 4),
        "Directional_Accuracy_pct": round(directional_accuracy, 4),
    }


if __name__ == "__main__":
    splits = expanding_window_splits(1000, initial_train_size=500, test_size=100, max_folds=5)
    print(f"Generated folds: {len(splits)}")
    for i, (tr, te) in enumerate(splits, start=1):
        print(f"Fold {i}: train={tr[0]}..{tr[-1]}, test={te[0]}..{te[-1]}")
