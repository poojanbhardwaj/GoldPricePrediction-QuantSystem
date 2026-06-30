from __future__ import annotations

import ast
from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _app_function(name: str, namespace: dict):
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    values = dict(namespace)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), values)
    return values[name]


def test_phase26_loader_falls_back_to_direct_csv_when_registry_is_empty(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "latest" / "phase26_product_experience"
    artifact_dir.mkdir(parents=True)
    expected = pd.DataFrame([{"Asset": "Silver", "Horizon": 5, "OpportunityScore": 41.5}])
    expected.to_csv(artifact_dir / "phase26_asset_plans.csv", index=False)

    load_table = _app_function("_load_phase26_table", {
        "Path": Path,
        "pd": pd,
        "__file__": str(tmp_path / "app.py"),
        "PHASE26_PRODUCT_EXPERIENCE": "phase26_product_experience",
        "load_latest_artifact": lambda *args, **kwargs: pd.DataFrame(),
        "_safe_filename_part": lambda value: str(value).strip().lower().replace(" ", "_"),
    })

    result = load_table("phase26_asset_plans")
    pd.testing.assert_frame_equal(result, expected)


def test_phase29_loader_reads_checked_in_csv_and_handles_missing_file(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "latest" / "phase29_final_user_experience"
    artifact_dir.mkdir(parents=True)
    expected = pd.DataFrame([{
        "Asset": "Gold", "PredictedPrice": 4114.5446, "PredictedMovePct": 0.8788,
    }])
    expected.to_csv(artifact_dir / "phase29_all_asset_prediction_snapshot.csv", index=False)

    load_table = _app_function("_load_phase29_table", {
        "Path": Path,
        "pd": pd,
        "__file__": str(tmp_path / "app.py"),
        "PHASE29_FINAL_USER_EXPERIENCE": "phase29_final_user_experience",
    })

    pd.testing.assert_frame_equal(
        load_table("phase29_all_asset_prediction_snapshot.csv"), expected
    )
    assert load_table("missing.csv").empty


def test_deployment_demo_artifacts_are_narrowly_allowlisted():
    expected_paths = {
        "artifacts/latest/phase26_product_experience/phase26_asset_plans.csv",
        "artifacts/latest/phase26_product_experience/phase26_portfolio_plan.csv",
        "artifacts/latest/phase26_product_experience/phase26_research_snapshot.csv",
        "artifacts/latest/phase29_final_user_experience/phase29_all_asset_prediction_snapshot.csv",
        "artifacts/latest/phase29_final_user_experience/phase29_final_user_plans.csv",
        "artifacts/latest/phase29_final_user_experience/phase29_cost_aware_asset_plans.csv",
    }
    ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")

    for relative_path in expected_paths:
        assert (ROOT / relative_path).exists(), relative_path
        assert f"!{relative_path}" in ignore_text


def test_clean_session_hydrates_saved_research_and_phase29_fallback_views():
    phase29_path = (
        ROOT / "artifacts" / "latest" / "phase29_final_user_experience"
        / "phase29_all_asset_prediction_snapshot.csv"
    )
    saved_snapshot = pd.read_csv(phase29_path)
    first_row = saved_snapshot.iloc[0]
    expected_price = f"{float(first_row['PredictedPrice']):,.2f}"
    expected_move = f"{float(first_row['PredictedMovePct']):,.2f}%"

    with patch("src.artifact_store.load_latest_artifact", return_value=pd.DataFrame()):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run(timeout=90)

    assert not app.exception
    report = app.session_state["phase29_user_report"]
    hydrated_snapshot = report["AllAssetPredictionSnapshot"]
    assert len(hydrated_snapshot) == len(saved_snapshot)
    assert hydrated_snapshot["PredictedPrice"].notna().any()
    assert hydrated_snapshot["PredictedMovePct"].notna().any()
    assert not app.session_state["phase26_asset_plans"].empty
    assert not app.session_state["phase26_portfolio_plan"].empty
    assert not app.session_state["phase26_research_snapshot"].empty

    main_content = "\n".join(str(item.value) for item in app.markdown)
    assert "Showing saved research snapshot from the latest checked-in demo run" in main_content
    assert expected_price in main_content
    assert expected_move in main_content
    assert "Run research" not in main_content
    assert "No saved estimate" not in main_content
    assert "ExpectedDelay" not in main_content

    app.session_state["phase26_asset_plans"] = pd.DataFrame()
    app.session_state["phase26_portfolio_plan"] = pd.DataFrame()
    app.sidebar.radio[0].set_value("Asset Plans").run(timeout=90)
    asset_plan_content = "\n".join(str(item.value) for item in app.markdown)
    assert not app.exception
    assert "Predicted price" in asset_plan_content
    assert expected_price in asset_plan_content

    app.sidebar.radio[0].set_value("Portfolio Summary").run(timeout=90)
    portfolio_content = "\n".join(str(item.value) for item in app.markdown)
    assert not app.exception
    assert "Current market snapshot" in portfolio_content
    assert "Average opportunity score" in portfolio_content
    assert "Highest opportunity asset" in portfolio_content
    assert "No portfolio summary" not in portfolio_content

