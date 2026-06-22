"""
Smoke tests for Phase 5 Better Feature Intelligence.

These tests are intentionally lightweight and synthetic. They verify:
  1. FI_* features are created for a multi-asset dataset.
  2. The features are not all NaN after cleaning.
  3. No future values are used: changing data after a date must not change
     features before that date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_intelligence import (
    add_phase5_feature_intelligence,
    build_feature_intelligence_report,
    phase5_feature_columns,
)


def _make_synthetic_market_data(n: int = 360) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    t = np.arange(n, dtype=float)

    return pd.DataFrame(
        {
            "Gold_Close": 1500 + 0.8 * t + 12 * np.sin(t / 11),
            "Silver_Close": 18 + 0.02 * t + 0.6 * np.sin(t / 9),
            "Oil_Close": 55 + 0.03 * t + 4 * np.sin(t / 19),
            "BTC_Close": 9000 + 40 * t + 400 * np.sin(t / 17),
            "SP500_Close": 3000 + 2.5 * t + 30 * np.sin(t / 15),
            "GLD_Close": 140 + 0.08 * t + 1.5 * np.sin(t / 10),
            "DXY_Close": 100 + 0.04 * np.sin(t / 13),
            "VIX_Close": 20 + 3 * np.sin(t / 5),
            "TNX_Close": 4 + 0.2 * np.sin(t / 23),
        },
        index=idx,
    )


def test_phase5_features_created():
    df = _make_synthetic_market_data()
    out = add_phase5_feature_intelligence(df, target_col="Gold_Close")
    fi_cols = phase5_feature_columns(out)

    assert len(out) > 250
    assert len(fi_cols) >= 80
    assert out[fi_cols].isna().sum().sum() == 0

    report = build_feature_intelligence_report(out, target_col="Gold_Close")
    assert report.phase5_columns == len(fi_cols)
    assert not report.family_counts.empty
    assert "Cross-asset relationship" in set(report.family_counts["FeatureFamily"])
    assert "Macro/risk pressure" in set(report.family_counts["FeatureFamily"])


def test_phase5_features_do_not_use_future_values():
    df1 = _make_synthetic_market_data()
    df2 = df1.copy()

    # Create a huge future shock only AFTER the comparison date.
    comparison_date = df1.index[170]
    future_start = df1.index[220]
    df2.loc[future_start:, "Gold_Close"] *= 3.0
    df2.loc[future_start:, "BTC_Close"] *= 0.3
    df2.loc[future_start:, "VIX_Close"] *= 5.0

    out1 = add_phase5_feature_intelligence(df1, target_col="Gold_Close")
    out2 = add_phase5_feature_intelligence(df2, target_col="Gold_Close")

    common_cols = phase5_feature_columns(out1)
    common_cols = [c for c in common_cols if c in out2.columns]

    # Since all Phase 5 features are current/past rolling calculations,
    # changing future data must not change earlier feature values.
    left = out1.loc[comparison_date, common_cols]
    right = out2.loc[comparison_date, common_cols]
    diff = (left - right).abs().max()
    assert float(diff) < 1e-12


if __name__ == "__main__":
    test_phase5_features_created()
    test_phase5_features_do_not_use_future_values()
    print("Phase 5 feature intelligence tests passed.")
