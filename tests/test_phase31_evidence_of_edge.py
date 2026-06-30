from __future__ import annotations

import ast
from pathlib import Path
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evidence_of_edge import (
    EDGE_EVIDENCE_COLUMNS,
    build_edge_evidence_table,
    build_validation_metric_table,
    classify_edge_status,
    format_edge_metric,
    load_latest_edge_metric_sources,
    summarize_edge_evidence,
)


FORBIDDEN = re.compile(
    r"\b(buy|sell|guaranteed|profitable|approved)\b",
    flags=re.IGNORECASE,
)


def _watch_row(
    asset: str = "Gold",
    *,
    category: str = "Watchlist Candidate",
    move: float | None = 1.0,
    score: float = 70.0,
    status: str = "Watch",
    cost: str = "CostsManageable",
) -> dict:
    return {
        "Asset": asset,
        "Category": category,
        "Direction": "Bullish" if move is not None and move >= 0.35 else "Neutral",
        "PredictedMovePct": move,
        "PredictedPrice": 101.0 if move is not None else None,
        "BestHorizon": 5,
        "OpportunityScore": score,
        "Status": status,
        "CostVerdict": cost,
        "UpgradeTrigger": "Require another matching forward validation window.",
    }


def _cost_row(
    asset: str = "Gold",
    *,
    active: float | None = 2.0,
    passive: float | None = 1.0,
    gap: float | None = 1.0,
    drag: float | None = 0.2,
    cost: str = "CostsManageable",
) -> dict:
    return {
        "Asset": asset,
        "BestHorizon": 5,
        "PassiveBenchmarkName": f"{asset} passive reference",
        "NetActiveEstimatePct": active,
        "NetPassiveEstimatePct": passive,
        "ActiveMinusPassiveNetPct": gap,
        "CostDragPct": drag,
        "CostVerdict": cost,
        "MainRisk": "Validation breadth remains limited.",
    }


def test_positive_gap_manageable_cost_and_validation_support_edge():
    watchlist = pd.DataFrame([_watch_row()])
    costs = pd.DataFrame([_cost_row()])
    validation = pd.DataFrame([{"Asset": "Gold", "Horizon": 5, "Sharpe": 1.1}])

    result = build_edge_evidence_table(watchlist, cost_plans=costs, validation_metrics=validation)
    row = result.iloc[0]

    assert row["EdgeStatus"] == "Edge Supported"
    assert row["EvidenceGrade"] == "B"
    assert row["Sharpe"] == 1.1


def test_negative_active_minus_passive_is_benchmark_weak():
    result = build_edge_evidence_table(
        pd.DataFrame([_watch_row()]),
        cost_plans=pd.DataFrame([_cost_row(active=0.5, passive=1.2, gap=-0.7)]),
        validation_metrics=pd.DataFrame([{"Asset": "Gold", "Horizon": 5, "Sharpe": 1.0}]),
    )

    assert result.iloc[0]["EdgeStatus"] == "Benchmark Weak"
    assert result.iloc[0]["EvidenceGrade"] == "D"


def test_excessive_cost_verdict_is_cost_blocked():
    result = build_edge_evidence_table(
        pd.DataFrame([_watch_row(cost="CostsTooHighForSignal")]),
        cost_plans=pd.DataFrame([_cost_row(cost="CostsTooHighForSignal")]),
    )

    assert result.iloc[0]["EdgeStatus"] == "Cost Blocked"
    assert result.iloc[0]["EvidenceGrade"] == "F"


def test_candidate_with_missing_validation_is_watch_only():
    result = build_edge_evidence_table(
        pd.DataFrame([_watch_row()]),
        cost_plans=pd.DataFrame([_cost_row()]),
        validation_metrics=pd.DataFrame(),
    )

    assert result.iloc[0]["EdgeStatus"] == "Watch Only"
    assert "validation evidence is unavailable" in result.iloc[0]["EvidenceSummary"].casefold()


def test_missing_all_evidence_is_insufficient():
    watch = _watch_row(category="Insufficient Evidence", move=None, score=0, status="Not Enough Evidence", cost="MissingEstimate")
    watch["PredictedPrice"] = None
    result = build_edge_evidence_table(pd.DataFrame([watch]))

    assert result.iloc[0]["EdgeStatus"] == "Insufficient Evidence"
    assert result.iloc[0]["EvidenceGrade"] == "F"


def test_missing_metrics_stay_missing_instead_of_becoming_zero():
    result = build_edge_evidence_table(
        pd.DataFrame([_watch_row()]),
        cost_plans=pd.DataFrame([_cost_row()]),
    )
    row = result.iloc[0]

    for field in ("WinRatePct", "Sharpe", "MaxDrawdownPct", "WalkForwardReturnPct", "TradeCount"):
        assert pd.isna(row[field]), field
        assert row[field] != 0


def test_summary_counts_and_top_supported_edge_are_correct():
    table = pd.DataFrame([
        {"Asset": "Gold", "EdgeStatus": "Edge Supported", "EvidenceGrade": "B", "OpportunityScore": 70},
        {"Asset": "Silver", "EdgeStatus": "Edge Supported", "EvidenceGrade": "A", "OpportunityScore": 65},
        {"Asset": "Crude Oil", "EdgeStatus": "Watch Only", "EvidenceGrade": "C", "OpportunityScore": 50},
        {"Asset": "Bitcoin", "EdgeStatus": "Cost Blocked", "EvidenceGrade": "F", "OpportunityScore": 45},
        {"Asset": "Gold ETF", "EdgeStatus": "Insufficient Evidence", "EvidenceGrade": "F", "OpportunityScore": None},
    ])

    assert summarize_edge_evidence(table) == {
        "total_assets": 5,
        "edge_supported_count": 2,
        "watch_only_count": 1,
        "insufficient_evidence_count": 1,
        "cost_blocked_count": 1,
        "top_edge_asset": "Silver",
        "top_edge_grade": "A",
    }


def test_column_aliases_are_resolved_without_metric_invention():
    watchlist = pd.DataFrame([_watch_row(category="Actionable Candidate")])
    costs = pd.DataFrame([{
        "Asset": "Gold",
        "Horizon": 5,
        "BenchmarkName": "Gold benchmark",
        "ActiveEstimatePct": 2.2,
        "PassiveEstimatePct": 1.0,
        "ExcessReturnPct": 1.2,
        "CostDragPct": 0.25,
        "CostVerdict": "CostsLow",
    }])
    research = build_validation_metric_table({
        "phase20_true_ml_performance": pd.DataFrame([{
            "Asset": "Gold",
            "Horizon": 5,
            "HitRatePct": 58.0,
            "SharpeRatio": 1.3,
            "MaxDrawdown": -8.0,
            "StrategyReturnPct": 4.5,
            "Trades": 18,
        }])
    })

    row = build_edge_evidence_table(watchlist, cost_plans=costs, validation_metrics=research).iloc[0]
    assert row["BenchmarkName"] == "Gold benchmark"
    assert row["ActiveEstimatePct"] == 2.2
    assert row["PassiveEstimatePct"] == 1.0
    assert row["ActiveMinusPassivePct"] == 1.2
    assert row["WinRatePct"] == 58.0
    assert row["Sharpe"] == 1.3
    assert row["MaxDrawdownPct"] == -8.0
    assert row["WalkForwardReturnPct"] == 4.5
    assert row["TradeCount"] == 18.0
    assert row["EdgeStatus"] == "Edge Supported"
    assert row["EvidenceGrade"] == "A"


def test_validation_table_maps_sharpe_proxy():
    table = build_validation_metric_table({
        "phase16_asset_horizon_benchmark": pd.DataFrame([{
            "Asset": "Gold", "Horizon": 5, "SharpeProxy": 0.72,
        }])
    })

    assert table.iloc[0]["Sharpe"] == 0.72


def test_validation_table_maps_hit_rate_to_win_rate():
    table = build_validation_metric_table({
        "phase20_true_ml_performance": pd.DataFrame([{
            "Asset": "Silver", "Horizon": 10, "HitRatePct": 57.5,
        }])
    })

    assert table.iloc[0]["WinRatePct"] == 57.5


def test_validation_table_maps_saved_return_aliases():
    table = build_validation_metric_table({
        "phase16_asset_horizon_benchmark": pd.DataFrame([{
            "Asset": "Gold", "Horizon": 5, "ModelStrategyReturnPct": 4.2,
        }]),
        "phase17_historical_replay_performance": pd.DataFrame([{
            "Asset": "Bitcoin", "Horizon": 1, "TotalReturnPct": 3.1,
        }]),
    })

    returns = table.set_index("Asset")["WalkForwardReturnPct"].to_dict()
    assert returns == {"Gold": 4.2, "Bitcoin": 3.1}


def test_edge_table_hydrates_validation_by_asset_and_horizon():
    metrics = build_validation_metric_table({
        "phase20_true_ml_performance": pd.DataFrame([
            {"Asset": "Gold", "Horizon": 1, "SharpeLike": 0.4, "MaturedTrades": 20},
            {"Asset": "Gold", "Horizon": 5, "SharpeLike": 1.4, "MaturedTrades": 8},
        ])
    })

    row = build_edge_evidence_table(
        pd.DataFrame([_watch_row()]),
        cost_plans=pd.DataFrame([_cost_row()]),
        validation_metrics=metrics,
    ).iloc[0]

    assert row["Sharpe"] == 1.4
    assert row["TradeCount"] == 8.0
    assert row["ValidationMetricSource"] == "phase20_true_ml_performance"
    assert "Validation metrics are loaded from phase20_true_ml_performance." in row["EvidenceSummary"]


def test_edge_table_falls_back_to_same_asset_when_horizon_is_unavailable():
    metrics = build_validation_metric_table({
        "phase17_historical_replay_performance": pd.DataFrame([{
            "Asset": "Gold", "Horizon": 20, "SharpeProxy": 0.9, "TradeCount": 31,
        }])
    })

    row = build_edge_evidence_table(
        pd.DataFrame([_watch_row()]), validation_metrics=metrics
    ).iloc[0]

    assert row["Sharpe"] == 0.9
    assert row["TradeCount"] == 31.0


def test_missing_metric_source_files_are_safe(tmp_path):
    sources = load_latest_edge_metric_sources(tmp_path / "does-not-exist")

    assert sources
    assert all(isinstance(frame, pd.DataFrame) and frame.empty for frame in sources.values())
    assert build_validation_metric_table(sources).empty


def test_format_edge_metric_preserves_missing_evidence_label():
    assert format_edge_metric(None) == "Evidence unavailable"
    assert format_edge_metric(float("nan"), "%") == "Evidence unavailable"
    assert format_edge_metric(2.345, "%", 2) == "2.35%"


def test_evidence_page_is_after_candidate_watchlist_and_language_is_restrained():
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    module_source = (ROOT / "src" / "evidence_of_edge.py").read_text(encoding="utf-8")
    ast.parse(app_source)
    navigation = app_source.split("PRIMARY_PRODUCT_PAGES = [", 1)[1].split("]", 1)[0]
    page_block = app_source.split('elif page == "Evidence of Edge":', 1)[1].split(
        'elif page == "Paper Research Journey":', 1
    )[0]

    assert navigation.index('"Candidate Watchlist"') < navigation.index('"Evidence of Edge"')
    assert navigation.index('"Evidence of Edge"') < navigation.index('"Asset Plans"')
    assert "_render_evidence_of_edge_section(phase29_snapshot)" in page_block
    assert FORBIDDEN.search(module_source) is None
    assert FORBIDDEN.search(page_block) is None
    assert EDGE_EVIDENCE_COLUMNS == [
        "Asset", "Category", "Direction", "EdgeStatus", "EvidenceGrade", "OpportunityScore",
        "PredictedMovePct", "CostVerdict", "Status", "BenchmarkName", "ActiveEstimatePct",
        "PassiveEstimatePct", "ActiveMinusPassivePct", "CostDragPct", "WinRatePct", "Sharpe",
        "MaxDrawdownPct", "WalkForwardReturnPct", "TradeCount", "ValidationMetricSource",
        "EvidenceSummary", "MainRisk", "RequiredBeforeAction",
    ]


def test_classify_edge_status_returns_three_readable_fields():
    result = classify_edge_status({
        "Category": "Watchlist Candidate",
        "PredictedMovePct": 1.0,
        "ActiveMinusPassivePct": 0.5,
        "CostVerdict": "Costs manageable",
        "Status": "Watch",
        "Sharpe": 0.8,
    })
    assert result[0] == "Edge Supported"
    assert result[1] in {"A", "B"}
    assert result[2]
