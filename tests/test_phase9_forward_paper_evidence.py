from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.asset_config import get_asset_names, get_target_column
import src.forward_evidence_tracker as tracker
from src.forward_evidence_tracker import (
    FORWARD_ACCURACY_SUMMARY_COLUMNS,
    FORWARD_COVERAGE_COLUMNS,
    FORWARD_PROBABILITY_CALIBRATION_COLUMNS,
    FORWARD_SIGNAL_LOG_COLUMNS,
    run_forward_paper_evidence_tracker,
)


def _price_data() -> pd.DataFrame:
    index = pd.bdate_range("2026-01-01", periods=80)
    data = {}
    for asset_i, asset in enumerate(get_asset_names()):
        col = get_target_column(asset)
        base = 100.0 + asset_i * 10.0
        data[col] = [base + i * (1.0 + asset_i * 0.05) for i in range(len(index))]
    return pd.DataFrame(data, index=index)


def _prediction(asset="Gold", horizon=5, probability=0.62, signal_date="2026-01-02", model="UnitTestModel") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Asset": asset,
                "Horizon": horizon,
                "SignalDate": signal_date,
                "ModelName": model,
                "ProbabilityUp": probability,
                "PredictedDirection": "Up" if probability >= 0.5 else "Down",
            }
        ]
    )


def _all_asset_horizon_predictions() -> pd.DataFrame:
    rows = []
    for asset in get_asset_names():
        for horizon in [1, 5, 10, 20, 30]:
            rows.append(
                {
                    "Asset": asset,
                    "Horizon": horizon,
                    "SignalDate": "2026-01-02",
                    "ModelName": "UnitTestModel",
                    "ProbabilityUp": 0.60,
                    "PredictedDirection": "Up",
                }
            )
    return pd.DataFrame(rows)


def _matured_existing_log() -> pd.DataFrame:
    return run_forward_paper_evidence_tracker(
        raw_df=_price_data(),
        prediction_table=_prediction(asset="Gold", horizon=5, signal_date="2026-01-02"),
        assets=["Gold"],
        horizons=[5],
        generate_new_signals=True,
        update_matured_outcomes=True,
        as_of_date="2026-01-20",
        min_forward_evidence=1,
    ).forward_signal_log


def test_pending_signal_does_not_get_future_outcome_before_target_date():
    report = run_forward_paper_evidence_tracker(
        raw_df=_price_data(),
        prediction_table=_prediction(horizon=5),
        assets=["Gold"],
        horizons=[5],
        generate_new_signals=True,
        update_matured_outcomes=True,
        as_of_date="2026-01-05",
        min_forward_evidence=2,
    )

    row = report.forward_signal_log.iloc[0]
    assert row["Status"] == "Pending"
    assert pd.isna(row["ActualDirection"])
    assert pd.isna(row["ExitPrice"])
    assert "OutcomeNotMatured" in str(row["Warnings"])


def test_generates_pending_row_per_selected_asset_horizon_from_prediction_source():
    report = run_forward_paper_evidence_tracker(
        raw_df=_price_data(),
        prediction_table=_all_asset_horizon_predictions(),
        assets=get_asset_names(),
        horizons=[1, 5, 10, 20, 30],
        generate_new_signals=True,
        update_matured_outcomes=True,
        as_of_date="2026-01-02",
        min_forward_evidence=1,
    )

    assert len(report.forward_signal_log) == len(get_asset_names()) * 5
    assert report.forward_signal_log["Status"].eq("Pending").all()
    assert report.forward_signal_log["ActualDirection"].isna().all()
    assert report.forward_signal_log["ExitPrice"].isna().all()
    assert report.forward_signal_log["WinLoss"].fillna("").eq("").all()


def test_fresh_generation_prefers_direct_model_source_over_true_raw_replay():
    original_generator = tracker.generate_forward_model_prediction_rows
    try:
        tracker.generate_forward_model_prediction_rows = lambda **kwargs: _all_asset_horizon_predictions()
        true_raw_replay = _matured_existing_log()
        report = tracker.run_forward_paper_evidence_tracker(
            raw_df=_price_data(),
            true_raw_trade_log_table=true_raw_replay,
            assets=get_asset_names(),
            horizons=[1, 5, 10, 20, 30],
            generate_new_signals=True,
            update_matured_outcomes=True,
            as_of_date="2026-01-02",
            min_forward_evidence=1,
        )
    finally:
        tracker.generate_forward_model_prediction_rows = original_generator

    assert len(report.forward_signal_log) == len(get_asset_names()) * 5
    assert report.forward_signal_log["Status"].eq("Pending").all()
    assert report.forward_signal_log["EvidenceMode"].eq("ForwardPaperSignal").all()


def test_unavailable_prediction_source_creates_invalid_rows_with_warnings():
    report = run_forward_paper_evidence_tracker(
        raw_df=None,
        prediction_table=None,
        assets=["Gold", "Bitcoin"],
        horizons=[1, 5],
        generate_new_signals=True,
        update_matured_outcomes=True,
        as_of_date="2026-01-02",
        min_forward_evidence=1,
    )

    assert len(report.forward_signal_log) == 4
    assert report.forward_signal_log["Status"].eq("Invalid").all()
    assert report.forward_signal_log["Warnings"].astype(str).str.contains("FreshPredictionUnavailable").all()
    assert report.forward_signal_log["Warnings"].astype(str).str.contains("MissingProbability").all()


def test_matured_signal_gets_outcome_only_after_target_date():
    report = run_forward_paper_evidence_tracker(
        raw_df=_price_data(),
        prediction_table=_prediction(horizon=5),
        assets=["Gold"],
        horizons=[5],
        generate_new_signals=True,
        update_matured_outcomes=True,
        as_of_date="2026-01-20",
        min_forward_evidence=1,
    )

    row = report.forward_signal_log.iloc[0]
    assert row["Status"] == "Matured"
    assert row["ActualDirection"] == 1
    assert pd.notna(row["ActualOutcomeDate"])
    assert pd.notna(row["RealizedReturn"])


def test_all_configured_assets_and_horizons_are_supported():
    report = run_forward_paper_evidence_tracker(
        raw_df=_price_data(),
        prediction_table=_all_asset_horizon_predictions(),
        assets=get_asset_names(),
        horizons=[1, 5, 10, 20, 30],
        generate_new_signals=True,
        update_matured_outcomes=True,
        as_of_date="2026-03-31",
        min_forward_evidence=1,
    )

    assert set(FORWARD_SIGNAL_LOG_COLUMNS).issubset(report.forward_signal_log.columns)
    assert set(FORWARD_ACCURACY_SUMMARY_COLUMNS).issubset(report.forward_accuracy_summary.columns)
    assert set(FORWARD_PROBABILITY_CALIBRATION_COLUMNS).issubset(report.forward_probability_calibration_summary.columns)
    assert set(FORWARD_COVERAGE_COLUMNS).issubset(report.asset_horizon_forward_coverage.columns)
    assert set(get_asset_names()).issubset(set(report.forward_signal_log["Asset"]))
    assert {1, 5, 10, 20, 30}.issubset(set(report.forward_signal_log["Horizon"].astype(int)))


def test_existing_log_can_be_uploaded_and_appended_without_deleting_history():
    first = run_forward_paper_evidence_tracker(
        raw_df=_price_data(),
        prediction_table=_prediction(asset="Gold", horizon=5, signal_date="2026-01-02"),
        assets=["Gold", "Bitcoin"],
        horizons=[5],
        generate_new_signals=True,
        update_matured_outcomes=True,
        as_of_date="2026-01-20",
        min_forward_evidence=1,
    )
    second = run_forward_paper_evidence_tracker(
        raw_df=_price_data(),
        existing_forward_signal_log=first.forward_signal_log,
        prediction_table=_prediction(asset="Bitcoin", horizon=5, signal_date="2026-01-05"),
        assets=["Gold", "Bitcoin"],
        horizons=[5],
        generate_new_signals=True,
        update_matured_outcomes=True,
        as_of_date="2026-01-22",
        min_forward_evidence=1,
    )

    assert len(second.forward_signal_log) == 2
    assert {"Gold", "Bitcoin"} == set(second.forward_signal_log["Asset"])
    assert second.forward_signal_log[second.forward_signal_log["Asset"].eq("Gold")]["Status"].iloc[0] == "Matured"


def test_rerun_same_asset_horizon_signal_date_model_does_not_duplicate_rows():
    first = run_forward_paper_evidence_tracker(
        raw_df=_price_data(),
        prediction_table=_prediction(asset="Gold", horizon=5, signal_date="2026-01-02"),
        assets=["Gold"],
        horizons=[5],
        generate_new_signals=True,
        update_matured_outcomes=True,
        as_of_date="2026-01-02",
        min_forward_evidence=1,
    )
    second = run_forward_paper_evidence_tracker(
        raw_df=_price_data(),
        existing_forward_signal_log=first.forward_signal_log,
        prediction_table=_prediction(asset="Gold", horizon=5, signal_date="2026-01-02"),
        assets=["Gold"],
        horizons=[5],
        generate_new_signals=True,
        update_matured_outcomes=True,
        as_of_date="2026-01-02",
        min_forward_evidence=1,
    )

    assert len(second.forward_signal_log) == 1


def test_failed_forward_trades_remain_visible():
    report = run_forward_paper_evidence_tracker(
        raw_df=_price_data(),
        prediction_table=_prediction(asset="Gold", horizon=5, probability=0.30, signal_date="2026-01-02"),
        assets=["Gold"],
        horizons=[5],
        generate_new_signals=True,
        update_matured_outcomes=True,
        as_of_date="2026-01-20",
        min_forward_evidence=1,
    )

    row = report.forward_signal_log.iloc[0]
    assert row["Status"] == "Matured"
    assert row["PredictedDirection"] == "Down"
    assert row["WinLoss"] == "Loss"
    assert row["RealizedReturn"] < 0


def test_no_production_ready_or_candidate_promotion_labels_created():
    report = run_forward_paper_evidence_tracker(
        raw_df=_price_data(),
        prediction_table=_prediction(),
        assets=["Gold"],
        horizons=[5],
        generate_new_signals=True,
        update_matured_outcomes=True,
        as_of_date="2026-01-20",
    )

    assert report.settings["production_ready_label_allowed"] == False
    assert report.settings["candidate_promotion_allowed"] == False
    forbidden = {"ProductionReady", "ProductionReadyLabelAllowed", "RecommendedReliabilityGrade", "ShouldPromoteGrade"}
    for table in [
        report.forward_signal_log,
        report.forward_accuracy_summary,
        report.forward_probability_calibration_summary,
        report.asset_horizon_forward_coverage,
    ]:
        assert forbidden.isdisjoint(set(table.columns))


if __name__ == "__main__":
    test_pending_signal_does_not_get_future_outcome_before_target_date()
    test_generates_pending_row_per_selected_asset_horizon_from_prediction_source()
    test_fresh_generation_prefers_direct_model_source_over_true_raw_replay()
    test_unavailable_prediction_source_creates_invalid_rows_with_warnings()
    test_matured_signal_gets_outcome_only_after_target_date()
    test_all_configured_assets_and_horizons_are_supported()
    test_existing_log_can_be_uploaded_and_appended_without_deleting_history()
    test_rerun_same_asset_horizon_signal_date_model_does_not_duplicate_rows()
    test_failed_forward_trades_remain_visible()
    test_no_production_ready_or_candidate_promotion_labels_created()
    print("Phase 9 forward paper evidence tests passed.")
