from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.candidate_watchlist import build_candidate_watchlist
from src.data_provenance import get_snapshot_freshness_label
from src.evidence_of_edge import build_edge_evidence_table
from src.research_record_lookup import (
    find_asset_horizon_record,
    normalize_asset_key,
    normalize_horizon_key,
)
from src.ui_components import humanize_label


ROOT = Path(__file__).resolve().parents[1]


def _visible_text(app: AppTest) -> str:
    output = []
    for collection in (
        app.markdown, app.caption, app.info, app.warning, app.success, app.error, app.metric
    ):
        output.extend(str(item.value) for item in collection)
    return "\n".join(output)


def _record() -> pd.DataFrame:
    return pd.DataFrame([{
        "Asset": "Gold",
        "Horizon": "1D",
        "BestHorizon": 1,
        "OpportunityScore": 72.0,
        "PredictedMovePct": 1.25,
        "PredictedPrice": pd.NA,
        "Status": "Watch",
        "CostVerdict": "CostsManageable",
    }])


def test_asset_and_horizon_aliases_resolve_one_record():
    records = _record()
    for asset in ("Gold", "gold", "Gold_Close", "GC=F"):
        for horizon in ("1D", "1d", "1 day", "1-day", 1):
            match = find_asset_horizon_record(records, asset, horizon)
            assert match is not None
            assert match["OpportunityScore"] == 72.0


def test_multiasset_aliases_cover_configured_sources():
    expected = {
        "SI=F": "Silver",
        "Oil_Close": "Crude Oil",
        "CL=F": "Crude Oil",
        "BTC": "Bitcoin",
        "BTC-USD": "Bitcoin",
        "SP500_Close": "S&P 500",
        "^GSPC": "S&P 500",
        "GLD_Close": "Gold ETF",
    }
    assert {alias: normalize_asset_key(alias) for alias in expected} == expected
    assert [normalize_horizon_key(value) for value in ("5D", "5 day", "5-day", 5)] == [5] * 4


def test_partial_record_preserves_real_score_without_fake_price():
    match = find_asset_horizon_record(_record(), "gold", "1d")
    assert match is not None
    assert match["OpportunityScore"] == 72.0
    assert pd.isna(match["PredictedPrice"])


def test_watchlist_and_edge_use_canonical_record_identity():
    prediction = _record().assign(Asset="GC=F")
    cost = _record().assign(Asset="Gold_Close", NetActiveEstimatePct=0.8, NetPassiveEstimatePct=0.3)
    plan = _record().assign(Asset="gold")
    watchlist = build_candidate_watchlist(prediction, cost, plan)
    assert list(watchlist["Asset"]) == ["Gold"]
    assert float(watchlist.iloc[0]["OpportunityScore"]) == 72.0
    edge = build_edge_evidence_table(watchlist, prediction, cost, plan)
    assert list(edge["Asset"]) == ["Gold"]
    assert float(edge.iloc[0]["OpportunityScore"]) == 72.0


def test_saved_provenance_is_not_mislabeled_as_refreshed():
    label = get_snapshot_freshness_label("2026-06-29", "saved_artifact", today=date(2026, 7, 2))
    assert label["source_label"] == "Saved research snapshot"
    assert label["latest_date_label"] == "Latest source date: 2026-06-29"
    assert label["age_label"] == "Snapshot age: 3 calendar days"
    assert label["freshness_label"] == "Freshness: Delayed"
    assert "refreshed" not in label["source_label"].casefold()


def test_humanize_internal_and_broken_labels():
    assert humanize_label("OpportunityScore") == "Opportunity score"
    assert humanize_label("MissingEstimate") == "Missing estimate"
    assert humanize_label("NotEnoughEvidence") == "Not enough evidence"
    assert humanize_label("CostManageable") == "Costs manageable"
    assert humanize_label("Re view tar get re play estim ate") == "Review target replay estimate"


def test_pending_navigation_is_applied_before_public_navigation_widget():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    apply_position = source.index("apply_pending_product_navigation()")
    widget_position = source.index('key="primary_product_navigation"', apply_position)
    assert apply_position < widget_position
    auth_source = (ROOT / "src" / "auth_manager.py").read_text(encoding="utf-8")
    assert 'session_state["_pending_product_navigation"]' in auth_source
    assert "session_state.primary_product_navigation =" not in auth_source


def test_authenticated_gold_1d_plan_hydrates_saved_values_without_whole_card_fallback():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    app.session_state["user_unlocked"] = True
    app.session_state["current_app_user_id"] = 999999
    app.session_state["auth_provider"] = "local_password"
    app.session_state["auth_user_id"] = "quality-audit-user"
    app.session_state["current_user_label"] = "Quality Audit"
    app.session_state["current_user_email"] = "quality-audit@example.com"
    app.session_state["current_user_is_local_dev"] = True
    app.session_state["primary_product_navigation"] = "Asset Plans"
    app.session_state["selected_asset"] = "Gold"
    app.session_state["selected_horizon"] = 1
    app.run(timeout=90)
    text = _visible_text(app)
    assert not app.exception
    assert "Gold" in text
    assert "Predicted price" in text
    assert "No usable asset-horizon research record was found" not in text
    assert "Opportunity score: 30/100" not in text
