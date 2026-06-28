from pathlib import Path
import io
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import src.artifact_store as store
from src.action_plan_engine import run_actionable_research_plan
from src.artifact_store import (
    ArtifactNotFoundError,
    build_input_source_table,
    get_artifact_registry,
    list_latest_artifacts,
    load_latest_artifact,
    resolve_artifact,
    save_phase_artifacts,
    validate_required_artifacts,
)


def _with_temp_store(fn):
    old_root = store.ARTIFACT_ROOT
    with tempfile.TemporaryDirectory() as tmp:
        store.ARTIFACT_ROOT = Path(tmp) / "artifacts"
        try:
            return fn()
        finally:
            store.ARTIFACT_ROOT = old_root


def _df(asset="Gold", horizon=5, value=1.0):
    return pd.DataFrame([{"Asset": asset, "Horizon": horizon, "Value": value}])


def _uploaded_csv(text: str, name: str = "uploaded.csv"):
    uploaded = io.BytesIO(text.encode("utf-8"))
    uploaded.name = name
    return uploaded


def test_save_phase_artifacts_creates_run_latest_and_registry():
    def run():
        result = save_phase_artifacts("Phase 10B Test Phase", {"table": _df()}, config={"mode": "unit"})
        run_id = result["RunId"]

        assert (store.ARTIFACT_ROOT / "runs" / run_id).exists()
        assert (store.ARTIFACT_ROOT / "latest" / "phase_10b_test_phase").exists()
        assert (store.ARTIFACT_ROOT / "registry.json").exists()
        assert (store.ARTIFACT_ROOT / "runs" / run_id / "manifest.json").exists()

        registry = get_artifact_registry()
        assert run_id in registry["Runs"]
        assert "phase_10b_test_phase" in registry["Latest"]

    _with_temp_store(run)


def test_repeated_runs_do_not_overwrite_and_latest_updates():
    def run():
        first = save_phase_artifacts("Phase 10B Test Phase", {"table": _df(value=1.0)})
        second = save_phase_artifacts("Phase 10B Test Phase", {"table": _df(value=2.0)})

        assert first["RunId"] != second["RunId"]
        assert (store.ARTIFACT_ROOT / "runs" / first["RunId"]).exists()
        assert (store.ARTIFACT_ROOT / "runs" / second["RunId"]).exists()
        latest = load_latest_artifact("Phase 10B Test Phase", "table", required=True)
        assert latest["Value"].iloc[0] == 2.0

    _with_temp_store(run)


def test_load_latest_artifact_and_metadata_fields():
    def run():
        saved = save_phase_artifacts("Phase 10B Test Phase", {"table": _df(asset="Bitcoin", horizon=10)})
        latest = load_latest_artifact("Phase 10B Test Phase", "table", required=True)
        metadata = saved["Artifacts"]["table"]

        assert latest["Asset"].iloc[0] == "Bitcoin"
        assert metadata["Phase"] == "Phase 10B Test Phase"
        assert metadata["ArtifactName"] == "table"
        assert metadata["RunId"] == saved["RunId"]
        assert metadata["CreatedAt"]
        assert metadata["Rows"] == 1
        assert "Asset" in metadata["Columns"]
        assert metadata["AssetsCovered"] == ["Bitcoin"]
        assert metadata["HorizonsCovered"] == [10]

    _with_temp_store(run)


def test_resolve_artifact_latest_default_and_uploaded_override_only_when_preferred():
    def run():
        save_phase_artifacts("Phase 10B Test Phase", {"table": _df(value=1.0)})
        uploaded = _uploaded_csv("Asset,Horizon,Value\nGold,5,9.0\n")

        default_resolved = resolve_artifact("Phase 10B Test Phase", "table", uploaded_file=uploaded, prefer_uploaded=False)
        assert default_resolved["Source"] == "LatestSavedArtifact"
        assert default_resolved["Data"]["Value"].iloc[0] == 1.0

        uploaded = _uploaded_csv("Asset,Horizon,Value\nGold,5,9.0\n")
        override_resolved = resolve_artifact("Phase 10B Test Phase", "table", uploaded_file=uploaded, prefer_uploaded=True)
        assert override_resolved["Source"] == "UploadedOverride"
        assert override_resolved["Data"]["Value"].iloc[0] == 9.0

    _with_temp_store(run)


def test_missing_required_and_optional_artifacts_are_controlled():
    def run():
        optional = resolve_artifact("Missing Phase", "missing_table", required=False)
        assert optional["Source"] == "Missing"
        assert optional["Status"] == "MissingOptional"

        try:
            resolve_artifact("Missing Phase", "missing_table", required=True)
        except ArtifactNotFoundError as exc:
            assert "Missing required artifact" in str(exc)
        else:
            raise AssertionError("Expected ArtifactNotFoundError")

    _with_temp_store(run)


def test_input_source_table_statuses():
    def run():
        save_phase_artifacts("Phase 10B Test Phase", {"table": _df()})
        latest = resolve_artifact("Phase 10B Test Phase", "table")
        uploaded = resolve_artifact("Phase 10B Test Phase", "other", uploaded_file=_uploaded_csv("A\n1\n"), prefer_uploaded=True)
        missing = resolve_artifact("Phase 10B Test Phase", "missing")
        source_table = build_input_source_table([latest, uploaded, missing])

        assert {"LatestSavedArtifact", "UploadedOverride", "Missing"}.issubset(set(source_table["Source"]))
        assert {"Loaded", "MissingOptional"}.issubset(set(source_table["Status"]))

    _with_temp_store(run)


def test_validate_required_artifacts_returns_clear_diagnostics():
    def run():
        save_phase_artifacts("Phase 10B Test Phase", {"table": _df()})
        diagnostics = validate_required_artifacts(
            [
                {"phase_name": "Phase 10B Test Phase", "artifact_name": "table", "required": True},
                {"phase_name": "Phase 10B Test Phase", "artifact_name": "missing", "required": True},
            ]
        )

        assert "Loaded" in set(diagnostics["Status"])
        assert "MissingRequired" in set(diagnostics["Status"])

    _with_temp_store(run)


def test_phase10_can_resolve_inputs_from_artifact_store_without_uploads():
    def run():
        save_phase_artifacts(
            "Phase 8F Probability Calibration",
            {
                "probability_calibration_summary": pd.DataFrame(
                    [
                        {
                            "Asset": "Bitcoin",
                            "Horizon": 5,
                            "RawProbabilityOutcomesAvailable": True,
                            "TotalTrades": 12,
                            "BrierScore": 0.20,
                            "CalibrationGrade": "UsefulButNoisy",
                            "UsefulProbabilityFilterFound": True,
                        }
                    ]
                ),
                "probability_calibration_warnings": pd.DataFrame(columns=["Asset", "Horizon", "WarningType"]),
            },
        )
        save_phase_artifacts(
            "Phase 8I True Raw Trade Logs",
            {
                "true_raw_trade_log": pd.DataFrame(
                    [
                        {
                            "Asset": "Bitcoin",
                            "Horizon": 5,
                            "ProbabilityUp": 0.66,
                            "ActualDirection": 1,
                            "RealizedReturn": 0.01,
                            "VsBuyHold": 0.004,
                            "MaxDrawdownDuringTrade": -0.04,
                        }
                    ]
                )
            },
        )
        save_phase_artifacts(
            "Phase 9 Forward Paper Evidence",
            {
                "forward_signal_log": pd.DataFrame(
                    [
                        {
                            "Asset": "Bitcoin",
                            "Horizon": 5,
                            "Status": "Pending",
                            "ProbabilityUp": 0.67,
                            "PredictedDirection": "Up",
                            "SignalStrength": "Medium",
                        }
                    ]
                ),
                "forward_accuracy_summary": pd.DataFrame(
                    [{"Asset": "Bitcoin", "Horizon": 5, "PendingSignals": 1, "MaturedSignals": 0, "Warnings": "NotEnoughForwardEvidence"}]
                ),
                "forward_probability_calibration_summary": pd.DataFrame(columns=["Asset", "Horizon", "BrierScore"]),
                "forward_warning_table": pd.DataFrame(columns=["Asset", "Horizon", "WarningType"]),
            },
        )

        resolved = {
            "probability_calibration_summary": resolve_artifact("Phase 8F Probability Calibration", "probability_calibration_summary"),
            "probability_calibration_warnings": resolve_artifact("Phase 8F Probability Calibration", "probability_calibration_warnings"),
            "true_raw_trade_log": resolve_artifact("Phase 8I True Raw Trade Logs", "true_raw_trade_log"),
            "forward_signal_log": resolve_artifact("Phase 9 Forward Paper Evidence", "forward_signal_log"),
            "forward_accuracy_summary": resolve_artifact("Phase 9 Forward Paper Evidence", "forward_accuracy_summary"),
            "forward_probability_calibration_summary": resolve_artifact("Phase 9 Forward Paper Evidence", "forward_probability_calibration_summary"),
            "forward_warning_table": resolve_artifact("Phase 9 Forward Paper Evidence", "forward_warning_table"),
        }
        assert all(item["Source"] == "LatestSavedArtifact" for item in resolved.values())

        report = run_actionable_research_plan(
            probability_calibration_summary=resolved["probability_calibration_summary"]["Data"],
            probability_calibration_warnings=resolved["probability_calibration_warnings"]["Data"],
            true_raw_trade_log=resolved["true_raw_trade_log"]["Data"],
            forward_signal_log=resolved["forward_signal_log"]["Data"],
            forward_accuracy_summary=resolved["forward_accuracy_summary"]["Data"],
            forward_probability_calibration_summary=resolved["forward_probability_calibration_summary"]["Data"],
            forward_warning_table=resolved["forward_warning_table"]["Data"],
            assets=["Bitcoin"],
            horizons=[5],
        )
        assert not report.ranked_asset_horizon_plan.empty
        assert report.ranked_asset_horizon_plan["Asset"].iloc[0] == "Bitcoin"

    _with_temp_store(run)


def test_list_latest_artifacts_returns_registry_table():
    def run():
        save_phase_artifacts("Phase 10B Test Phase", {"table": _df()})
        latest = list_latest_artifacts()

        assert not latest.empty
        assert {"Phase", "ArtifactName", "RunId", "Rows", "CreatedAt", "Path"}.issubset(latest.columns)

    _with_temp_store(run)


if __name__ == "__main__":
    test_save_phase_artifacts_creates_run_latest_and_registry()
    test_repeated_runs_do_not_overwrite_and_latest_updates()
    test_load_latest_artifact_and_metadata_fields()
    test_resolve_artifact_latest_default_and_uploaded_override_only_when_preferred()
    test_missing_required_and_optional_artifacts_are_controlled()
    test_input_source_table_statuses()
    test_validate_required_artifacts_returns_clear_diagnostics()
    test_phase10_can_resolve_inputs_from_artifact_store_without_uploads()
    test_list_latest_artifacts_returns_registry_table()
    print("Phase 10B artifact store tests passed.")
