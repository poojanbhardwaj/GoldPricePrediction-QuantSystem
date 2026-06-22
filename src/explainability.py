# src/explainability.py

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def get_feature_importance(
    model,
    feature_names: List[str],
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Lightweight explainability helper.

    Works with:
    - CatBoost: get_feature_importance()
    - LightGBM / XGBoost / RandomForest / sklearn trees: feature_importances_
    - Linear models: abs(coef_)

    If SHAP is too heavy, use this first.
    """

    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    if hasattr(model, "get_feature_importance"):
        importance = np.asarray(model.get_feature_importance(), dtype=float)
    elif hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        raise ValueError("Model does not expose feature importances or coefficients.")

    if len(importance) != len(feature_names):
        raise ValueError(
            f"Feature importance length ({len(importance)}) does not match "
            f"feature_names length ({len(feature_names)})."
        )

    out = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )

    total = out["importance"].abs().sum()
    if total > 0:
        out["importance_pct"] = out["importance"].abs() / total * 100.0
    else:
        out["importance_pct"] = 0.0

    return out.sort_values("importance_pct", ascending=False).head(top_n).reset_index(drop=True)


def explain_signal_direction(predicted_return_pct: float, top_features: pd.DataFrame) -> str:
    direction = "bullish" if predicted_return_pct > 0 else "bearish" if predicted_return_pct < 0 else "neutral"

    if top_features is None or top_features.empty:
        return f"The model prediction is {direction}, but feature importance is unavailable."

    features = ", ".join(top_features["feature"].head(3).astype(str).tolist())
    return (
        f"The model prediction is {direction}. The largest contributing feature groups "
        f"by model importance are: {features}. This is a model explanation, not a causal proof."
    )


if __name__ == "__main__":
    class DemoModel:
        feature_importances_ = np.array([0.5, 0.2, 0.3])

    print(get_feature_importance(DemoModel(), ["RSI", "MACD", "Volatility"]))
