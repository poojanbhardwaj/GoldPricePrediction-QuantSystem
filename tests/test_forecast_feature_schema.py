import numpy as np
import pandas as pd
import pytest

from src.predict import (
    FEATURE_SCHEMA_MISMATCH_MESSAGE,
    Predictor,
    build_forecast_feature_frame,
    validate_feature_schema,
)


REQUIRED_PHASE5_FEATURES = [
    "FI_Target_ReturnZ_3d",
    "FI_Target_ReturnZ_5d",
    "FI_Target_ReturnZ_10d",
    "FI_Target_ReturnZ_20d",
    "FI_Target_ReturnZ_60d",
    "FI_Target_DistSMA_5",
    "FI_Target_DistSMA_20",
    "FI_Target_DistSMA_50",
    "FI_Target_Trend_5_20",
    "FI_Target_Trend_20_50",
    "FI_Target_Trend_50_100",
    "FI_Target_VolRatio_5_20",
    "FI_Target_VolRatio_20_60",
    "FI_Target_TrendPersistence_5d",
    "FI_Target_TrendPersistence_20d",
    "FI_Target_MomentumExhaustion_20d",
    "FI_Target_MomentumExhaustion_60d",
    "FI_Regime_TrendUp_20_50",
]


class _IdentityIndicators:
    def add_all(self, frame):
        return frame.copy()


class _IdentityFeatureEngineer:
    def build_features(self, frame):
        return frame.copy()


class _IdentityScaler:
    def transform(self, values):
        return np.asarray(values, dtype=float)


class _ZeroReturnModel:
    def predict(self, values, **kwargs):
        return np.zeros(len(values), dtype=float)


class _ReturnPreprocessor:
    target_col = "Gold_Close"
    predict_returns = True
    seq_len = 60
    _feature_scaler = _IdentityScaler()

    @staticmethod
    def reconstruct_prices_from_returns(returns, last_known_price, actual_prices=None):
        return last_known_price * np.exp(np.asarray(returns, dtype=float))


def _historical_prices(rows=340):
    index = pd.bdate_range("2024-01-01", periods=rows)
    x = np.arange(rows, dtype=float)
    prices = 1800.0 + (0.35 * x) + (8.0 * np.sin(x / 9.0))
    return pd.DataFrame({"Gold_Close": prices}, index=index)


def test_schema_validation_detects_missing_phase5_features():
    frame = _historical_prices()
    result = validate_feature_schema(frame, REQUIRED_PHASE5_FEATURES)

    assert result["safe"] is False
    assert "FI_Target_ReturnZ_3d" in result["missing_columns"]
    assert FEATURE_SCHEMA_MISMATCH_MESSAGE in result["message"]


def test_forecast_enrichment_generates_training_phase5_schema():
    enriched = build_forecast_feature_frame(
        _historical_prices(),
        target_col="Gold_Close",
        indicators_engine=_IdentityIndicators(),
        feature_engineer=_IdentityFeatureEngineer(),
    )
    result = validate_feature_schema(enriched, REQUIRED_PHASE5_FEATURES)

    assert result["safe"] is True
    assert not enriched.empty
    assert enriched["FI_Target_ReturnZ_3d"].abs().sum() > 0


def test_forecast_uses_derived_phase5_features_without_zero_fill():
    predictor = Predictor(_ZeroReturnModel(), _ReturnPreprocessor())

    forecast = predictor.forecast(
        _historical_prices(),
        feature_cols=["FI_Target_ReturnZ_3d"],
        n_days=2,
        indicators_engine=_IdentityIndicators(),
        feature_engineer=_IdentityFeatureEngineer(),
    )

    assert len(forecast) == 2
    assert forecast["Predicted_Price"].notna().all()


def test_incompatible_schema_raises_controlled_error_not_pandas_keyerror():
    predictor = Predictor(_ZeroReturnModel(), _ReturnPreprocessor())

    with pytest.raises(ValueError, match="Model feature schema mismatch") as exc_info:
        predictor.forecast(
            _historical_prices(),
            feature_cols=["FI_Target_ReturnZ_3d", "ArtifactOnlyFeature"],
            n_days=1,
            indicators_engine=_IdentityIndicators(),
            feature_engineer=_IdentityFeatureEngineer(),
        )

    assert not isinstance(exc_info.value, KeyError)
    assert "ArtifactOnlyFeature" in str(exc_info.value)


def test_schema_validation_does_not_add_or_zero_fill_unknown_features():
    frame = _historical_prices()
    result = validate_feature_schema(frame, ["FI_Target_ReturnZ_3d"])

    assert result["safe"] is False
    assert "FI_Target_ReturnZ_3d" not in frame.columns
