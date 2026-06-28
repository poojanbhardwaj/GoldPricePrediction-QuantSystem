import re

import numpy as np
import pandas as pd

from src.asset_config import get_asset_names, get_target_column
from src.market_regime_intelligence import (
    ASSET_HORIZON_REGIME_COLUMNS,
    ASSET_REGIME_COLUMNS,
    MARKET_REGIME_PHASE_NAME,
    NEXT_REGIME_ACTION_COLUMNS,
    REGIME_ADJUSTED_SIZING_COLUMNS,
    REGIME_FACTOR_COLUMNS,
    REGIME_HORIZONS,
    REGIME_INPUT_SOURCE_COLUMNS,
    REGIME_RISK_COLUMNS,
    REGIME_SUMMARY_COLUMNS,
    REGIME_TRANSITION_COLUMNS,
    _classify_trend,
    run_market_regime_intelligence,
)


FORBIDDEN_LANGUAGE = re.compile(
    r"\b(Buy|Strong Buy|Invest Now|Production Ready|Guaranteed|Safe Profit)\b",
    flags=re.IGNORECASE,
)


def _synthetic_market_data(rows=320, include_optional=True):
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    x = np.arange(rows, dtype=float)
    data = pd.DataFrame({"Date": dates})
    data["Gold_Close"] = 100 + x * 0.22
    data["Silver_Close"] = 160 - x * 0.28
    data["Oil_Close"] = 92 + np.sin(x * 2 * np.pi / 40.0) * 0.45 + np.cos(x * 2 * np.pi / 23.0) * 0.15

    btc = np.full(rows, 100.0)
    btc[:260] += np.sin(x[:260] / 18.0) * 0.3
    btc[260:] += np.sin(x[260:] / 1.7) * 18.0 + np.linspace(0, 8, rows - 260)
    data["BTC_Close"] = btc

    data["SP500_Close"] = 100 + x * 0.12
    data["GLD_Close"] = 50 + x * 0.11
    if include_optional:
        data["VIX_Close"] = 16 + np.sin(x / 30.0)
        data["DXY_Close"] = 104 - x * 0.005
        data["TNX_Close"] = 4.2 - x * 0.001
    return data


def _phase14_sizing_table():
    rows = []
    for asset in get_asset_names():
        for horizon in REGIME_HORIZONS:
            weight = 3.0
            if asset == "Gold" and horizon == 30:
                weight = 10.0
            if asset == "Silver" and horizon == 5:
                weight = 8.0
            if asset == "Crude Oil" and horizon == 1:
                weight = 0.0
            if asset == "Bitcoin" and horizon == 1:
                weight = 6.0
            rows.append(
                {
                    "Asset": asset,
                    "Horizon": horizon,
                    "Phase12PaperWeightPct": weight,
                    "OptimizedPaperWeightPct": weight,
                    "RealCapitalAllowed": False,
                    "SuggestedRealWeightPct": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _run_report(**kwargs):
    return run_market_regime_intelligence(
        market_data=_synthetic_market_data(),
        use_project_market_data=False,
        use_artifact_store=False,
        assets=get_asset_names(),
        horizons=REGIME_HORIZONS,
        dynamic_position_sizing_table=_phase14_sizing_table(),
        autosave=False,
        **kwargs,
    )


def _all_text(report):
    frames = [
        report.regime_summary_table,
        report.asset_regime_table,
        report.asset_horizon_regime_table,
        report.regime_factor_table,
        report.regime_transition_table,
        report.regime_risk_table,
        report.regime_adjusted_sizing_table,
        report.next_regime_actions_table,
        report.regime_input_sources_table,
    ]
    return "\n".join(frame.astype(str).to_csv(index=False) for frame in frames)


def test_phase15_outputs_required_tables_and_columns():
    report = _run_report()

    expected_columns = {
        "regime_summary_table": REGIME_SUMMARY_COLUMNS,
        "asset_regime_table": ASSET_REGIME_COLUMNS,
        "asset_horizon_regime_table": ASSET_HORIZON_REGIME_COLUMNS,
        "regime_factor_table": REGIME_FACTOR_COLUMNS,
        "regime_transition_table": REGIME_TRANSITION_COLUMNS,
        "regime_risk_table": REGIME_RISK_COLUMNS,
        "regime_adjusted_sizing_table": REGIME_ADJUSTED_SIZING_COLUMNS,
        "next_regime_actions_table": NEXT_REGIME_ACTION_COLUMNS,
        "regime_input_sources_table": REGIME_INPUT_SOURCE_COLUMNS,
    }
    for table_name, columns in expected_columns.items():
        table = getattr(report, table_name)
        assert set(columns).issubset(table.columns), table_name

    assert len(report.asset_regime_table) == len(get_asset_names())
    assert len(report.asset_horizon_regime_table) == len(get_asset_names()) * len(REGIME_HORIZONS)
    assert len(report.regime_adjusted_sizing_table) == len(get_asset_names()) * len(REGIME_HORIZONS)


def test_phase15_classifies_trend_and_volatility_regimes():
    report = _run_report()
    by_asset = report.asset_regime_table.set_index("Asset")

    assert by_asset.loc["Gold", "TrendRegime"] in {"Uptrend", "StrongUptrend"}
    assert by_asset.loc["Silver", "TrendRegime"] in {"Downtrend", "StrongDowntrend"}
    assert by_asset.loc["Crude Oil", "TrendRegime"] == "Sideways"
    assert by_asset.loc["Bitcoin", "VolatilityRegime"] in {"HighVolatility", "ExtremeVolatility"}


def test_phase15_regime_confidence_is_capped_and_not_fake_certainty():
    report = _run_report()

    assert (report.asset_regime_table["RegimeConfidence"] <= 90.0).all()
    assert report.regime_summary_table.iloc[0]["RegimeConfidence"] <= 90.0


def test_phase15_strong_trend_with_mixed_cross_support_is_not_100_confidence():
    market = _synthetic_market_data()
    x = np.arange(len(market), dtype=float)
    market["SP500_Close"] = 100 + np.sin(x * 2 * np.pi / 45.0) * 0.15
    market["VIX_Close"] = 22.0

    report = run_market_regime_intelligence(
        market_data=market,
        use_project_market_data=False,
        use_artifact_store=False,
        assets=get_asset_names(),
        horizons=REGIME_HORIZONS,
        dynamic_position_sizing_table=_phase14_sizing_table(),
    )

    gold = report.asset_regime_table.set_index("Asset").loc["Gold"]
    assert gold["TrendRegime"] in {"Uptrend", "StrongUptrend"}
    assert gold["CrossAssetSupport"] == "Mixed"
    assert gold["RegimeConfidence"] <= 85.0


def test_phase15_summary_explains_stress_source():
    report = _run_report()
    summary = report.regime_summary_table.iloc[0]

    assert "MarketStressLevel" in report.regime_summary_table.columns
    assert "AssetRegimeStressLevel" in report.regime_summary_table.columns
    if "Stress" in str(summary["OverallMarketRegime"]) and summary["MarketStressLevel"] == "Low":
        assert summary["AssetRegimeStressLevel"] in {"Medium", "High"}
        assert "Asset-level" in str(summary["MainDrivers"])


def test_phase15_factor_explanations_align_with_impact():
    report = _run_report()
    factors = report.regime_factor_table

    for _, row in factors.iterrows():
        impact = str(row["RegimeImpact"])
        explanation = str(row["Explanation"]).lower()
        if impact == "Supportive":
            assert "supportive" in explanation
        elif impact == "Pressure":
            assert "pressure" in explanation or "unfavorable" in explanation
        elif impact == "Neutral":
            assert "neutral" in explanation
        elif impact == "Stress":
            assert "stress" in explanation


def test_phase15_trend_classifier_keeps_flat_noisy_series_sideways():
    x = np.arange(180, dtype=float)
    bounded_noise = 100 + np.sin(x * 2 * np.pi / 24.0) * 0.45 + np.cos(x * 2 * np.pi / 11.0) * 0.2

    trend, reason, score = _classify_trend(pd.Series(bounded_noise))

    assert trend == "Sideways"
    assert score == 50.0
    assert "small" in reason.lower() or "mixed" in reason.lower()


def test_phase15_trend_classifier_keeps_small_drift_sideways():
    x = np.arange(180, dtype=float)
    small_drift = 100 + np.linspace(0, 1.2, len(x)) + np.sin(x * 2 * np.pi / 30.0) * 0.25

    trend, _, _ = _classify_trend(pd.Series(small_drift))

    assert trend == "Sideways"


def test_phase15_trend_classifier_detects_clear_positive_and_negative_trends():
    x = np.arange(180, dtype=float)
    positive = 100 + np.linspace(0, 34, len(x)) + np.sin(x * 2 * np.pi / 30.0) * 0.2
    negative = 140 - np.linspace(0, 42, len(x)) + np.sin(x * 2 * np.pi / 30.0) * 0.2

    positive_trend, _, _ = _classify_trend(pd.Series(positive))
    negative_trend, _, _ = _classify_trend(pd.Series(negative))

    assert positive_trend in {"Uptrend", "StrongUptrend"}
    assert negative_trend in {"Downtrend", "StrongDowntrend"}


def test_phase15_missing_optional_cross_data_is_graceful():
    report = run_market_regime_intelligence(
        market_data=_synthetic_market_data(include_optional=False),
        use_project_market_data=False,
        use_artifact_store=False,
        assets=get_asset_names(),
        horizons=REGIME_HORIZONS,
        dynamic_position_sizing_table=_phase14_sizing_table(),
    )

    assert not report.regime_summary_table.empty
    factor_rows = report.regime_factor_table[report.regime_factor_table["Factor"].isin(["DXY trend", "VIX stress", "TNX/yield movement"])]
    assert set(factor_rows["RegimeImpact"]) == {"Unknown"}


def test_phase15_missing_critical_price_does_not_create_fake_regime():
    market = _synthetic_market_data().drop(columns=[get_target_column("Silver")])
    report = run_market_regime_intelligence(
        market_data=market,
        use_project_market_data=False,
        use_artifact_store=False,
        assets=get_asset_names(),
        horizons=REGIME_HORIZONS,
        dynamic_position_sizing_table=_phase14_sizing_table(),
    )

    silver = report.asset_regime_table.set_index("Asset").loc["Silver"]
    assert silver["AssetSpecificRegime"] == "Unknown"
    assert silver["MainRegimeRisk"] == "InsufficientRegimeData"
    source = report.regime_input_sources_table[report.regime_input_sources_table["SourceName"].eq("market_data")].iloc[0]
    assert get_target_column("Silver") in str(source["MissingCriticalColumns"])


def test_phase15_regime_adjustment_keeps_or_reduces_phase14_weights_by_default():
    report = _run_report()
    sizing = report.regime_adjusted_sizing_table

    assert (sizing["RegimeAdjustedPaperWeightPct"] <= sizing["Phase14OptimizedPaperWeightPct"] + 1e-9).all()

    gold_30 = sizing[(sizing["Asset"].eq("Gold")) & (sizing["Horizon"].eq(30))].iloc[0]
    assert gold_30["RegimeAdjustedPaperWeightPct"] == gold_30["Phase14OptimizedPaperWeightPct"]

    silver_5 = sizing[(sizing["Asset"].eq("Silver")) & (sizing["Horizon"].eq(5))].iloc[0]
    assert silver_5["RegimeAdjustedPaperWeightPct"] < silver_5["Phase14OptimizedPaperWeightPct"]

    oil_1 = sizing[(sizing["Asset"].eq("Crude Oil")) & (sizing["Horizon"].eq(1))].iloc[0]
    assert oil_1["Phase14OptimizedPaperWeightPct"] == 0.0
    assert oil_1["RegimeAdjustedPaperWeightPct"] == 0.0


def test_phase15_dangerous_regime_can_zero_simulated_paper_weight():
    report = _run_report()
    silver_5 = report.regime_adjusted_sizing_table[
        (report.regime_adjusted_sizing_table["Asset"].eq("Silver"))
        & (report.regime_adjusted_sizing_table["Horizon"].eq(5))
    ].iloc[0]
    assert silver_5["FinalRegimeSizingDecision"] in {"ZeroDueToDangerousRegime", "ReduceDueToRegime"}
    assert silver_5["RegimeAdjustedPaperWeightPct"] <= silver_5["Phase14OptimizedPaperWeightPct"]


def test_phase15_keeps_failed_and_risky_rows_visible():
    report = _run_report()
    risky_assets = set(report.regime_risk_table["Asset"].astype(str))
    assert "Silver" in risky_assets or "Bitcoin" in risky_assets
    assert set(get_asset_names()).issubset(set(report.asset_regime_table["Asset"]))


def test_phase15_forbidden_live_trading_language_absent():
    report = _run_report()
    assert FORBIDDEN_LANGUAGE.search(_all_text(report)) is None


def test_phase15_autosaves_artifacts(monkeypatch):
    saved = {}

    def fake_save(phase_name, artifacts, inputs=None, config=None, warnings=None, run_id=None):
        saved["phase_name"] = phase_name
        saved["artifacts"] = artifacts
        saved["config"] = config
        return {"RunId": "test_run", "Artifacts": {name: {} for name in artifacts}}

    monkeypatch.setattr("src.market_regime_intelligence.save_phase_artifacts", fake_save)

    report = run_market_regime_intelligence(
        market_data=_synthetic_market_data(),
        use_project_market_data=False,
        use_artifact_store=False,
        assets=get_asset_names(),
        horizons=REGIME_HORIZONS,
        dynamic_position_sizing_table=_phase14_sizing_table(),
        autosave=True,
    )

    assert saved["phase_name"] == MARKET_REGIME_PHASE_NAME
    assert "regime_adjusted_sizing_table" in saved["artifacts"]
    assert report.saved_artifacts["RunId"] == "test_run"
