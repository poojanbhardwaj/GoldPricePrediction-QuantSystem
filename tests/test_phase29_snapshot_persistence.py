from __future__ import annotations

import ast
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _StreamlitStub:
    def __init__(self, state):
        self.session_state = state


def _helpers(state: _SessionState, saved_snapshot: pd.DataFrame):
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        "_has_real_phase29_predictions",
        "_get_phase29_snapshot",
        "_store_phase29_run_report",
        "_phase29_placeholder_snapshot",
    }
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "pd": pd,
        "np": np,
        "st": _StreamlitStub(state),
        "_load_phase29_table": lambda filename: saved_snapshot.copy(),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "app.py", "exec"), namespace)
    return namespace


def _real_snapshot(asset: str, price: float, move: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "Asset": asset,
        "LatestPrice": price - 1,
        "PredictedPrice": price,
        "PredictedMovePct": move,
    }])


def test_get_snapshot_prefers_session_over_saved_artifact():
    session = _real_snapshot("Gold", 4100.0, 1.2)
    saved = _real_snapshot("Silver", 60.0, 0.8)
    state = _SessionState({
        "phase29_user_report": {"AllAssetPredictionSnapshot": session},
        "phase29_last_good_snapshot": None,
    })
    helpers = _helpers(state, saved)

    result = helpers["_get_phase29_snapshot"]()

    pd.testing.assert_frame_equal(result, session)
    pd.testing.assert_frame_equal(state["phase29_last_good_snapshot"], session)
    assert state["phase29_snapshot_source"] == "session"


def test_get_snapshot_falls_back_to_last_good_before_saved_artifact():
    last_good = _real_snapshot("Crude Oil", 72.0, -1.4)
    saved = _real_snapshot("Silver", 60.0, 0.8)
    state = _SessionState({
        "phase29_user_report": {"AllAssetPredictionSnapshot": pd.DataFrame()},
        "phase29_last_good_snapshot": last_good,
    })
    helpers = _helpers(state, saved)

    result = helpers["_get_phase29_snapshot"]()

    pd.testing.assert_frame_equal(result, last_good)
    assert state["phase29_snapshot_source"] == "last_good"


def test_empty_run_report_preserves_last_good_snapshot():
    last_good = _real_snapshot("Bitcoin", 60000.0, 0.6)
    state = _SessionState({
        "phase29_user_report": {},
        "phase29_last_good_snapshot": last_good.copy(),
        "phase29_snapshot_source": "last_good",
    })
    helpers = _helpers(state, pd.DataFrame())

    report = helpers["_store_phase29_run_report"]({
        "AllAssetPredictionSnapshot": pd.DataFrame(),
        "Warnings": [],
    })

    pd.testing.assert_frame_equal(state["phase29_last_good_snapshot"], last_good)
    pd.testing.assert_frame_equal(report["AllAssetPredictionSnapshot"], last_good)
    assert "latest saved research snapshot" in state["phase29_snapshot_notice"]
    assert state["phase29_snapshot_notice"] in report["Warnings"]


def test_placeholder_is_created_only_after_all_real_sources_fail():
    state = _SessionState({
        "phase29_user_report": {"AllAssetPredictionSnapshot": pd.DataFrame()},
        "phase29_last_good_snapshot": pd.DataFrame(),
    })
    helpers = _helpers(state, pd.DataFrame())

    snapshot = helpers["_get_phase29_snapshot"]()
    assert snapshot.empty
    assert state["phase29_snapshot_source"] == "placeholder"

    prices = pd.DataFrame([{"Asset": "Gold", "LatestPrice": 4000.0}])
    placeholder = helpers["_phase29_placeholder_snapshot"](prices)
    assert not placeholder.empty
    assert pd.isna(placeholder.iloc[0]["PredictedPrice"])
    assert helpers["_has_real_phase29_predictions"](placeholder) is False


def test_market_assistant_warns_before_rendering_placeholder_and_has_diagnostics():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    page_block = source.split('if page == "Market Research Assistant":', 1)[1].split(
        'elif page == "Candidate Watchlist":', 1
    )[0]

    assert (
        "Prediction snapshot unavailable. Showing current prices only. Run Full Research or "
        in page_block
    )
    assert 'st.expander("Research snapshot diagnostics"' in page_block
    assert page_block.index("_get_phase29_snapshot()") < page_block.index("_phase29_placeholder_snapshot(")


def test_incomplete_full_research_run_keeps_saved_predictions_visible():
    incomplete_report = {
        "AllAssetPredictionSnapshot": pd.DataFrame(),
        "ResearchSnapshot": pd.DataFrame(),
        "AssetPlans": pd.DataFrame(),
        "Warnings": ["Synthetic incomplete run"],
    }
    with patch(
        "src.final_user_dashboard.run_full_user_research",
        return_value=incomplete_report,
    ):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run(timeout=90)
        next(button for button in app.button if button.label == "Run Full Research").click().run(
            timeout=90
        )

    assert not app.exception
    content = "\n".join(str(item.value) for item in app.markdown)
    assert "Run research" not in content
    assert "No saved estimate" not in content
    assert any(
        "latest saved research snapshot" in str(warning.value).casefold()
        for warning in app.warning
    )
    diagnostics = next(
        table.value for table in app.dataframe if "SourceUsed" in table.value.columns
    ).iloc[0]
    assert diagnostics["SourceUsed"] == "last_good"
    assert diagnostics["NumericPredictedPrices"] > 0
