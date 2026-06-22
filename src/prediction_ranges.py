# src/prediction_ranges.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import NormalDist
from typing import Iterable, Optional, Dict, Any

import numpy as np


@dataclass
class PredictionRangeResult:
    last_price: float
    predicted_price: float
    predicted_return_pct: float
    lower_bound: float
    upper_bound: float
    lower_return_pct: float
    upper_return_pct: float
    confidence_level: float
    error_used: float
    error_source: str
    model_used: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_price(value: float, name: str) -> float:
    try:
        value = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be numeric.") from exc

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if value <= 0:
        raise ValueError(f"{name} must be positive.")

    return value


def calculate_rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)

    if len(y_true_arr) == 0 or len(y_pred_arr) == 0:
        raise ValueError("y_true and y_pred cannot be empty.")

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have the same length.")

    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if mask.sum() == 0:
        raise ValueError("No valid values available for RMSE calculation.")

    return float(np.sqrt(np.mean((y_true_arr[mask] - y_pred_arr[mask]) ** 2)))


def residual_volatility_error(
    residuals: Iterable[float],
    fallback_error: Optional[float] = None,
) -> float:
    residual_arr = np.asarray(list(residuals), dtype=float)
    residual_arr = residual_arr[np.isfinite(residual_arr)]

    if len(residual_arr) >= 5:
        return float(np.std(residual_arr, ddof=1))

    if fallback_error is not None and fallback_error > 0:
        return float(fallback_error)

    raise ValueError("Not enough residuals to estimate volatility error.")


def calculate_prediction_range(
    *,
    last_price: float,
    predicted_price: float,
    model_used: str,
    rmse: Optional[float] = None,
    residuals: Optional[Iterable[float]] = None,
    confidence_level: float = 0.68,
) -> PredictionRangeResult:
    """
    Builds an approximate prediction uncertainty range.

    confidence_level:
        0.68 gives roughly +/- 1 standard error.
        0.95 gives roughly +/- 1.96 standard errors.

    Important:
        This is not true market certainty. It is only an approximate band
        based on historical model error.
    """

    last_price = _validate_price(last_price, "last_price")
    predicted_price = _validate_price(predicted_price, "predicted_price")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    error_source = "rmse"

    if rmse is not None:
        error_used = float(rmse)
        if not np.isfinite(error_used) or error_used <= 0:
            raise ValueError("rmse must be a positive finite number.")
    elif residuals is not None:
        error_used = residual_volatility_error(residuals)
        error_source = "recent_residual_volatility"
    else:
        raise ValueError("Either rmse or residuals must be provided.")

    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    range_error = error_used * z_score

    lower_bound = max(0.0, predicted_price - range_error)
    upper_bound = predicted_price + range_error

    predicted_return_pct = ((predicted_price - last_price) / last_price) * 100.0
    lower_return_pct = ((lower_bound - last_price) / last_price) * 100.0
    upper_return_pct = ((upper_bound - last_price) / last_price) * 100.0

    return PredictionRangeResult(
        last_price=round(last_price, 4),
        predicted_price=round(predicted_price, 4),
        predicted_return_pct=round(predicted_return_pct, 4),
        lower_bound=round(lower_bound, 4),
        upper_bound=round(upper_bound, 4),
        lower_return_pct=round(lower_return_pct, 4),
        upper_return_pct=round(upper_return_pct, 4),
        confidence_level=round(confidence_level * 100.0, 2),
        error_used=round(error_used, 4),
        error_source=error_source,
        model_used=str(model_used),
    )


if __name__ == "__main__":
    result = calculate_prediction_range(
        last_price=4215.00,
        predicted_price=4296.65,
        rmse=61.39,
        confidence_level=0.68,
        model_used="CatBoost",
    )
    print("Prediction range test:")
    print(result.to_dict())
