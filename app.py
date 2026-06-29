"""Multi-asset market research and risk intelligence command center."""

import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config_loader import ConfigLoader
from src.logger import get_logger
from src.data_loader import DataLoader
from src.indicators import TechnicalIndicators
from src.feature_engineering import FeatureEngineer
from src.feature_intelligence import (
    add_phase5_feature_intelligence,
    build_feature_intelligence_report,
    phase5_feature_columns,
)
from src.preprocessing import Preprocessor
from src.train import ModelTrainer
from src.predict import Predictor
from src.visualization import Visualizer
from src.prediction_ranges import calculate_prediction_range
from src.signals import generate_trading_signal
from src.backtesting import run_backtest_from_predictions
from src.asset_config import get_asset_names, get_asset_config
from src.app_context import (
    DEFAULT_ASSET,
    DEFAULT_HORIZON,
    build_data_freshness_table,
    get_asset_target,
    get_available_horizons,
    get_supported_assets,
    validate_asset_horizon,
)
from src.explanation_glossary import glossary_entries
from src.workflow_guide import run_multiasset_workflow_audit
from src.research_orchestrator import (
    ADVANCED_DIAGNOSTIC_PAGES,
    PHASE26_PRODUCT_EXPERIENCE,
    PRIMARY_USER_PAGES,
    build_navigation_audit,
    collect_asset_horizon_evidence,
    load_latest_research_snapshot,
    run_research_engine,
    save_product_experience_artifacts,
)
from src.user_plan_generator import (
    build_high_risk_explanations,
    build_monitoring_plan,
    generate_all_asset_plans,
    generate_portfolio_plan,
    rank_asset_plans,
    save_premium_product_artifacts,
)
from src.cost_aware_plan import (
    COST_DISCLAIMER,
    compare_active_vs_passive_after_costs,
    default_cost_assumptions,
    generate_cost_aware_asset_plan,
)
from src.final_user_dashboard import (
    PHASE29_FINAL_USER_EXPERIENCE,
    build_all_asset_prediction_snapshot,
    generate_final_user_plan,
    get_latest_asset_prices,
    resolve_horizon_estimates,
    run_full_user_research,
    set_plan_navigation_state,
)
from src.baselines import price_baseline_leaderboard, model_vs_naive_summary
from src.directional_models import (
    train_directional_models,
    directional_leaderboard,
    directional_baseline_leaderboard,
    run_directional_probability_backtest,
)
from src.research_validation import (
    build_validation_report,
    regime_performance,
    walk_forward_validate_model,
)
from src.multiasset_validation import (
    run_multiasset_validation,
    summarize_asset_status,
)
from src.direct_forecast_models import (
    DIRECT_FORECAST_HORIZONS,
    run_asset_horizon_scan,
    run_direct_forecast_report,
    run_direct_forecast_signal_output,
)
from src.signal_engine import (
    run_candidate_deep_diagnostics,
    run_risk_controlled_candidate_upgrade,
    run_signal_engine,
    run_signal_research_scan,
    run_walk_forward_risk_validation,
    run_validation_locked_signal_engine,
)
from src.meta_signal_engine import (
    run_evidence_expansion,
    run_evidence_quality_diagnostics,
    run_meta_decision_audit,
    run_meta_score_calibration,
    run_probability_calibration,
    run_raw_trade_log_exporter,
    run_regime_aware_meta_signal,
    run_signal_policy_sensitivity,
    run_trade_evidence_ledger,
    run_true_raw_trade_log_generation,
)
from src.forward_evidence_tracker import run_forward_paper_evidence_tracker
from src.action_plan_engine import run_actionable_research_plan
from src.daily_research_center import run_daily_research_control_center
from src.portfolio_capital_simulator import run_portfolio_capital_simulator
from src.risk_warning_intelligence import RISK_INTELLIGENCE_PHASE_NAME, run_risk_warning_intelligence
from src.dynamic_risk_sizing import DYNAMIC_RISK_SIZING_PHASE_NAME, run_dynamic_risk_sizing
from src.market_regime_intelligence import MARKET_REGIME_PHASE_NAME, run_market_regime_intelligence
from src.strategy_benchmark_arena import STRATEGY_BENCHMARK_PHASE_NAME, run_strategy_benchmark_arena
from src.historical_model_replay import HISTORICAL_REPLAY_PHASE_NAME, run_historical_model_replay
from src.replay_benchmark_audit import REPLAY_BENCHMARK_AUDIT_PHASE_NAME, run_replay_benchmark_audit
from src.signal_policy_edge_lab import POLICY_LAB_HORIZONS, run_signal_policy_edge_lab
from src.true_historical_ml_replay import (
    MODEL_CHOICES as TRUE_ML_MODEL_CHOICES,
    TRUE_ML_HORIZONS,
    run_true_historical_ml_replay,
)
from src.unified_risk_command_center import (
    UNIFIED_RISK_COMMAND_CENTER_PHASE_NAME,
    run_unified_risk_command_center,
)
from src.prediction_edge_improvement import (
    DEFAULT_FEATURE_GROUPS as EDGE_FEATURE_GROUPS,
    DEFAULT_MODELS as EDGE_MODEL_CHOICES,
    EDGE_HORIZONS,
    OPTIONAL_MODELS as EDGE_OPTIONAL_MODELS,
    PREDICTION_EDGE_IMPROVEMENT_PHASE_NAME,
    run_prediction_edge_improvement,
)
from src.ui_components import (
    inject_premium_css,
    render_asset_plan_card,
    render_active_vs_passive_card,
    render_asset_price_card,
    render_beginner_explanation_box,
    render_blocked_capital_banner,
    render_cost_assumption_inputs,
    render_cost_summary_card,
    render_disclaimer_banner,
    render_download_buttons,
    render_empty_state,
    render_glass_container,
    render_glossary_expander,
    render_hero_section,
    render_metric_grid,
    render_market_snapshot_grid,
    render_monitoring_card,
    render_navigation_card,
    render_opportunity_card,
    render_pipeline_stepper,
    render_prediction_snapshot_card,
    render_premium_header,
    render_research_disclaimer,
    render_risk_explanation_card,
    render_run_research_panel,
    render_safe_table,
    render_section_header,
    render_status_card,
    render_status_tabs,
    render_score_explainer_card,
    render_simple_plan_card,
)
from src.artifact_store import (
    build_input_source_table,
    get_artifact_registry,
    list_latest_artifacts,
    load_latest_artifact,
    resolve_artifact,
    save_phase_artifacts,
    validate_required_artifacts,
)

logger = get_logger(__name__)
cfg    = ConfigLoader()
DEFAULT_TARGET_COLUMN = get_asset_target(DEFAULT_ASSET)

# ════════════════════════════════════════════════════════════════
# Page Config
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Multi-Asset Research & Risk Intelligence",
    page_icon="📊",
    layout=cfg.get("dashboard.layout", "wide"),
    initial_sidebar_state=cfg.get("dashboard.sidebar_state", "expanded"),
)

inject_premium_css()


# ════════════════════════════════════════════════════════════════
# Cached Pipeline Builders
# ════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600 * 6)  # refresh cache every 6 hours
def load_raw_data(start_date: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Load market data from start_date through TODAY. end_date is intentionally
    not a parameter here — DataLoader defaults to the current date whenever
    end_date isn't explicitly provided, so every call automatically pulls the
    latest available trading data rather than a fixed historical snapshot.
    """
    loader = DataLoader(start_date=start_date, end_date=None)
    return loader.load_all(use_cache=use_cache)


def _target_prefix(target_col: str) -> str:
    """Convert a close column such as Gold_Close or BTC_Close to its OHLCV prefix."""
    return str(target_col).replace("_Close", "")


def _safe_filename_part(value: str) -> str:
    """Normalize asset/model labels for downloadable file names."""
    safe = str(value).strip().lower().replace("&", "and")
    for ch in [" ", "/", "\\", ":", "*", "?", "\"", "<", ">", "|"]:
        safe = safe.replace(ch, "_")
    return "_".join(part for part in safe.split("_") if part) or "asset"


def _table_research_explanation(title: str) -> str:
    name = str(title).lower()
    if "baseline" in name:
        return "Compare the model or policy with serious simple references. A negative gap does not prove edge."
    if "leakage" in name:
        return "Any failed chronology or leakage check invalidates the affected research conclusion."
    if "prediction log" in name:
        return "Each row should represent a prediction formed only from information available on its prediction date."
    if "quality gate" in name:
        return "Failed gates remain binding and must not be inferred away from stronger-looking metrics."
    if "rejected" in name or "rejection" in name:
        return "Rejected candidates stay visible so weak or unstable evidence is not hidden."
    if "cost" in name:
        return "This table shows whether modeled execution costs erase the paper-research edge."
    return "Read this table as research evidence for the selected asset/horizon, not as an execution instruction."


@st.cache_data(show_spinner=False)
def _load_cached_market_history() -> pd.DataFrame:
    """Load the local master dataset only; never trigger a download from a primary page."""
    path = Path(__file__).resolve().parent / "data" / "processed" / "master_dataset.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
    except Exception:
        return pd.DataFrame()


def _load_phase26_table(artifact_name: str) -> pd.DataFrame:
    try:
        table = load_latest_artifact(PHASE26_PRODUCT_EXPERIENCE, artifact_name, required=False)
    except Exception:
        table = None
    return table if isinstance(table, pd.DataFrame) else pd.DataFrame()


def _load_phase29_table(filename: str) -> pd.DataFrame:
    path = Path(__file__).resolve().parent / "artifacts" / "latest" / PHASE29_FINAL_USER_EXPERIENCE / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=300)
def _latest_user_price_snapshot() -> pd.DataFrame:
    return get_latest_asset_prices(_load_cached_market_history())


def _phase29_placeholder_snapshot(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        return pd.DataFrame()
    frame = prices.copy()
    defaults = {
        "BestHorizon": 0, "PredictedPrice": np.nan, "PredictedMovePct": np.nan,
        "Status": "Not Enough Evidence", "OpportunityScore": 0.0, "OpportunityGrade": "F",
        "RiskLabel": "Not Enough Evidence", "CostVerdict": "MissingEstimate",
        "PassiveBenchmarkName": "Passive benchmark pending", "SimplePlan": "Run Full Research to build a complete paper-research plan.",
    }
    for column, value in defaults.items():
        frame[column] = value
    return frame


def _navigate_primary(destination: str) -> None:
    """Switch primary product pages from a CTA callback and rerun immediately."""
    if destination in set(PRIMARY_USER_PAGES) | {"Cost-Aware Plan", "Paper Research Journey"}:
        st.session_state.primary_product_navigation = destination
        st.rerun()


def _navigate_to_plan(asset: str, destination: str, horizon: Optional[int] = None) -> None:
    """Preserve plan context while moving between primary user pages."""
    set_plan_navigation_state(st.session_state, asset, destination, horizon)
    st.rerun()


def _status_card_style(status: str) -> str:
    return {
        "Track": "positive",
        "Watch": "info",
        "Wait": "neutral",
        "Avoid": "critical",
        "High Risk": "critical",
        "Data Issue": "warning",
        "Not Enough Evidence": "warning",
    }.get(str(status), "neutral")


def _render_asset_plan_cards(plans: pd.DataFrame, *, show_advanced: bool = False) -> None:
    if not isinstance(plans, pd.DataFrame) or plans.empty:
        render_empty_state(
            "No asset plans yet",
            "Generate a research plan to turn the latest saved evidence into conservative, plain-language cards.",
        )
        return
    for _, plan in plans.iterrows():
        render_asset_plan_card(plan.to_dict(), show_advanced=show_advanced)
        continue
        with st.container(border=True):
            left, right = st.columns([0.72, 0.28])
            with left:
                st.markdown(f"### {plan.get('Asset', '')} · {int(plan.get('Horizon', 0))}D")
                st.caption(str(plan.get("Summary", "")))
            with right:
                render_status_card(
                    "Status",
                    plan.get("Status", "Not Enough Evidence"),
                    f"Confidence: {plan.get('Confidence', 'Low')}",
                    _status_card_style(str(plan.get("Status", ""))),
                )
            detail_a, detail_b = st.columns(2)
            with detail_a:
                st.markdown(f"**Why:** {plan.get('Why', '')}")
                st.markdown(f"**Main risk:** {plan.get('MainRisk', '')}")
                st.markdown(f"**What to watch:** {plan.get('WhatToWatch', '')}")
            with detail_b:
                st.markdown(f"**Tracking condition:** {plan.get('TrackingCondition', '')}")
                st.markdown(f"**Invalidation condition:** {plan.get('InvalidationCondition', '')}")
                st.markdown(f"**Recheck:** {plan.get('RecheckWhen', '')}")
            if show_advanced:
                with st.expander("Advanced evidence", expanded=False):
                    st.write(plan.get("TechnicalEvidenceSummary", "No technical summary is available."))
                    st.caption(f"Sources: {plan.get('AdvancedEvidenceReferences', 'No saved references')}")


def _render_phase29_snapshot(snapshot: pd.DataFrame) -> None:
    if not isinstance(snapshot, pd.DataFrame) or snapshot.empty:
        render_empty_state("No final snapshot", "Use Run Full Research to combine prices, saved estimates, risk, benchmarks, and costs.")
        return

    render_section_header(
        "Current multi-asset snapshot",
        "Numbers use the latest available project dataset until an explicit refresh is requested.",
    )
    render_market_snapshot_grid(
        snapshot,
        on_view_plan=lambda asset, horizon: _navigate_to_plan(asset, "Asset Plans", horizon),
    )
    advanced_mode = st.toggle("Advanced mode", value=False, key="phase29_advanced_mode")

    assets = snapshot["Asset"].astype(str).tolist()
    default_asset = st.session_state.get("phase29_selected_plan_asset", DEFAULT_ASSET)
    selected_asset_index = assets.index(default_asset) if default_asset in assets else 0
    view_asset = st.selectbox("View plan", assets, index=selected_asset_index, key="phase29_main_plan_asset")
    st.session_state.phase29_selected_plan_asset = view_asset
    selected = snapshot[snapshot["Asset"].astype(str).eq(view_asset)].iloc[0].to_dict()
    selected_a, selected_b = st.columns(2)
    with selected_a:
        render_prediction_snapshot_card(selected)
        render_cost_summary_card(selected)
    with selected_b:
        render_active_vs_passive_card(selected)
        render_score_explainer_card(selected)
    render_simple_plan_card(selected)
    if advanced_mode:
        with st.expander("Advanced snapshot fields", expanded=False):
            st.dataframe(pd.DataFrame([selected]), width="stretch", hide_index=True)

    render_section_header("Compare all assets", "Filter and sort the same snapshot without changing its underlying evidence.")
    compare_a, compare_b, compare_c, compare_d = st.columns(4)
    with compare_a:
        status_values = ["All"] + sorted(snapshot["Status"].dropna().astype(str).unique().tolist())
        status_filter = st.selectbox("Status", status_values, key="phase29_status_filter")
    with compare_b:
        risk_values = ["All"] + sorted(snapshot.get("RiskLabel", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        risk_filter = st.selectbox("Risk", risk_values, key="phase29_risk_filter")
    with compare_c:
        freshness_values = ["All"] + sorted(snapshot["DataFreshness"].dropna().astype(str).unique().tolist())
        freshness_filter = st.selectbox("Freshness", freshness_values, key="phase29_freshness_filter")
    with compare_d:
        snapshot_sort = st.selectbox(
            "Sort", ["OpportunityScore", "PredictedMovePct", "CostVerdict", "ActiveMinusPassiveNetPct"],
            key="phase29_snapshot_sort",
        )
    view = snapshot.copy()
    if status_filter != "All":
        view = view[view["Status"].astype(str).eq(status_filter)]
    if risk_filter != "All" and "RiskLabel" in view:
        view = view[view["RiskLabel"].astype(str).eq(risk_filter)]
    if freshness_filter != "All":
        view = view[view["DataFreshness"].astype(str).eq(freshness_filter)]
    view = view.sort_values(snapshot_sort, ascending=False, na_position="last")
    columns = [column for column in (
        "Asset", "LatestPrice", "LatestPriceDate", "Status", "OpportunityScore", "BestHorizon",
        "PredictedPrice", "PredictedMovePct", "CostDragPct", "BreakEvenReturnPct",
        "NetActiveEstimatePct", "NetPassiveEstimatePct", "ActiveMinusPassiveNetPct",
        "CostVerdict", "PassiveBenchmarkName", "TrustScore",
    ) if column in view.columns]
    st.dataframe(view[columns], width="stretch", hide_index=True)

    ranked = snapshot.sort_values("OpportunityScore", ascending=False, na_position="last")
    cost_blocked = snapshot[snapshot.get("CostVerdict", pd.Series(dtype=str)).eq("CostsTooHighForSignal")].head(3)
    gaps = pd.to_numeric(snapshot.get("ActiveMinusPassiveNetPct", pd.Series(np.nan, index=snapshot.index)), errors="coerce")
    passive_stronger = snapshot[gaps.lt(0)].head(3)
    insight_a, insight_b, insight_c = st.columns(3)
    with insight_a:
        render_beginner_explanation_box("Top 3 closest to track", "; ".join(ranked.head(3)["Asset"].astype(str)) or "No ranked assets yet.")
    with insight_b:
        render_beginner_explanation_box("Top cost-blocked ideas", "; ".join(cost_blocked["Asset"].astype(str)) if not cost_blocked.empty else "No complete cost-blocked estimate yet.")
    with insight_c:
        render_beginner_explanation_box("Passive benchmark stronger", "; ".join(passive_stronger["Asset"].astype(str)) if not passive_stronger.empty else "No complete passive comparison yet.")
    render_download_buttons({
        "Final snapshot": (snapshot, "phase29_all_asset_prediction_snapshot.csv"),
        "Cost-aware plans": (snapshot, "phase29_cost_aware_asset_plans.csv"),
    })


@st.cache_data(show_spinner=False)
def build_features(df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COLUMN) -> pd.DataFrame:
    prefix = _target_prefix(target_col)
    ti = TechnicalIndicators(prefix=prefix)
    df = ti.add_all(df)

    # Forward-fill supporting market/macro columns before feature engineering.
    # Some assets update later than the selected target. Without this,
    # FeatureEngineer.dropna() can remove the latest target rows and make the
    # forecast start from an older date. Forward-fill is time-series safe
    # because it only carries past known values forward.
    df = df.sort_index().ffill()

    fe = FeatureEngineer(target_col=target_col)
    df = fe.build_features(df)

    # Phase 5: add market-aware feature intelligence.
    # These features use only current/past information, so they are valid
    # for the next-day target created in preprocessing.py.
    df = add_phase5_feature_intelligence(df, target_col=target_col)
    return df


def get_preprocessor_and_data(df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COLUMN):
    """Not cached — Preprocessor holds unpicklable scaler state we reuse live."""
    pp = Preprocessor(target_col=target_col)
    data = pp.run(df)
    return pp, data


def _asset_mismatch(selected_asset: str) -> bool:
    trained_asset = st.session_state.get("trained_asset")
    return bool(st.session_state.get("trained", False) and trained_asset and trained_asset != selected_asset)


def _stop_if_asset_mismatch(selected_asset: str) -> None:
    if _asset_mismatch(selected_asset):
        st.warning(
            f"Models in memory were trained for **{st.session_state.trained_asset}**, "
            f"but sidebar asset is **{selected_asset}**. Go to **Train Models** and train again."
        )
        st.stop()


def _safe_test_rmse(model_result, data) -> float:
    """Use held-out test error for uncertainty bands, never training error."""
    try:
        rmse = model_result.metrics_test.get("RMSE")
        if rmse is not None and np.isfinite(float(rmse)) and float(rmse) > 0:
            return float(rmse)
    except Exception:
        pass

    actual = np.asarray(data.prices_test, dtype=float)
    pred = np.asarray(model_result.predictions_test, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    if mask.sum() == 0:
        raise ValueError("Cannot calculate RMSE because test predictions are invalid.")
    return float(np.sqrt(np.mean((actual[mask] - pred[mask]) ** 2)))


def _build_backtest_frame(data, model_result, target_col: str = DEFAULT_TARGET_COLUMN) -> pd.DataFrame:
    """
    Build a backtest DataFrame from real held-out test predictions.

    The model predicts the target test price. To convert that to a trading
    signal, each predicted price is compared with the previous known actual
    price anchor.
    """
    actual_prices = np.asarray(data.prices_test, dtype=float)
    predicted_prices = np.asarray(model_result.predictions_test, dtype=float)

    if len(actual_prices) != len(predicted_prices):
        raise ValueError("Actual and predicted test arrays have different lengths.")

    first_anchor = float(data.last_price_before_test)
    anchors = np.concatenate([[first_anchor], actual_prices[:-1]])

    return pd.DataFrame(
        {
            target_col: anchors,
            "Predicted_Price": predicted_prices,
            "Actual_Next_Price": actual_prices,
            "Predicted_Return": predicted_prices / anchors - 1.0,
            "Actual_Next_Return": actual_prices / anchors - 1.0,
        },
        index=pd.to_datetime(data.test_index),
    )


# ════════════════════════════════════════════════════════════════
# Session State Initialization
# ════════════════════════════════════════════════════════════════

if "trained" not in st.session_state:
    st.session_state.trained = False
if "trainer" not in st.session_state:
    st.session_state.trainer = None
if "pp" not in st.session_state:
    st.session_state.pp = None
if "data" not in st.session_state:
    st.session_state.data = None
if "df_features" not in st.session_state:
    st.session_state.df_features = None
if "backtest_df" not in st.session_state:
    st.session_state.backtest_df = None
if "trained_asset" not in st.session_state:
    st.session_state.trained_asset = None
if "selected_asset" not in st.session_state:
    st.session_state.selected_asset = DEFAULT_ASSET
if "selected_horizon" not in st.session_state:
    st.session_state.selected_horizon = DEFAULT_HORIZON
if "phase23_multiasset_workflow_report" not in st.session_state:
    st.session_state.phase23_multiasset_workflow_report = None
if "phase26_research_snapshot" not in st.session_state:
    st.session_state.phase26_research_snapshot = None
if "phase26_asset_plans" not in st.session_state:
    st.session_state.phase26_asset_plans = None
if "phase26_portfolio_plan" not in st.session_state:
    st.session_state.phase26_portfolio_plan = None
if "phase29_user_report" not in st.session_state:
    st.session_state.phase29_user_report = None
if "phase29_selected_plan_asset" not in st.session_state:
    st.session_state.phase29_selected_plan_asset = DEFAULT_ASSET
if "directional_results" not in st.session_state:
    st.session_state.directional_results = None
if "directional_asset" not in st.session_state:
    st.session_state.directional_asset = None
if "multiasset_validation_report" not in st.session_state:
    st.session_state.multiasset_validation_report = None
if "multiasset_validation_settings" not in st.session_state:
    st.session_state.multiasset_validation_settings = None
if "direct_forecast_report" not in st.session_state:
    st.session_state.direct_forecast_report = None
if "direct_forecast_settings" not in st.session_state:
    st.session_state.direct_forecast_settings = None
if "direct_horizon_scan_report" not in st.session_state:
    st.session_state.direct_horizon_scan_report = None
if "direct_horizon_scan_settings" not in st.session_state:
    st.session_state.direct_horizon_scan_settings = None
if "signal_engine_output" not in st.session_state:
    st.session_state.signal_engine_output = None
if "signal_engine_result" not in st.session_state:
    st.session_state.signal_engine_result = None
if "signal_engine_settings" not in st.session_state:
    st.session_state.signal_engine_settings = None
if "signal_research_scan_report" not in st.session_state:
    st.session_state.signal_research_scan_report = None
if "signal_research_scan_settings" not in st.session_state:
    st.session_state.signal_research_scan_settings = None
if "candidate_diagnostics_report" not in st.session_state:
    st.session_state.candidate_diagnostics_report = None
if "candidate_diagnostics_settings" not in st.session_state:
    st.session_state.candidate_diagnostics_settings = None
if "risk_control_upgrade_report" not in st.session_state:
    st.session_state.risk_control_upgrade_report = None
if "risk_control_upgrade_settings" not in st.session_state:
    st.session_state.risk_control_upgrade_settings = None
if "walk_forward_validation_report" not in st.session_state:
    st.session_state.walk_forward_validation_report = None
if "walk_forward_validation_settings" not in st.session_state:
    st.session_state.walk_forward_validation_settings = None
if "meta_signal_report" not in st.session_state:
    st.session_state.meta_signal_report = None
if "meta_signal_settings" not in st.session_state:
    st.session_state.meta_signal_settings = None
if "meta_decision_audit_report" not in st.session_state:
    st.session_state.meta_decision_audit_report = None
if "meta_decision_audit_settings" not in st.session_state:
    st.session_state.meta_decision_audit_settings = None
if "meta_reliability_grading_report" not in st.session_state:
    st.session_state.meta_reliability_grading_report = None
if "meta_reliability_grading_settings" not in st.session_state:
    st.session_state.meta_reliability_grading_settings = None
if "evidence_expansion_report" not in st.session_state:
    st.session_state.evidence_expansion_report = None
if "evidence_expansion_settings" not in st.session_state:
    st.session_state.evidence_expansion_settings = None
if "evidence_quality_diagnostics_report" not in st.session_state:
    st.session_state.evidence_quality_diagnostics_report = None
if "evidence_quality_diagnostics_settings" not in st.session_state:
    st.session_state.evidence_quality_diagnostics_settings = None
if "signal_policy_sensitivity_report" not in st.session_state:
    st.session_state.signal_policy_sensitivity_report = None
if "signal_policy_sensitivity_settings" not in st.session_state:
    st.session_state.signal_policy_sensitivity_settings = None
if "probability_calibration_report" not in st.session_state:
    st.session_state.probability_calibration_report = None
if "probability_calibration_settings" not in st.session_state:
    st.session_state.probability_calibration_settings = None
if "trade_evidence_ledger_report" not in st.session_state:
    st.session_state.trade_evidence_ledger_report = None
if "trade_evidence_ledger_settings" not in st.session_state:
    st.session_state.trade_evidence_ledger_settings = None
if "raw_trade_log_exporter_report" not in st.session_state:
    st.session_state.raw_trade_log_exporter_report = None
if "raw_trade_log_exporter_settings" not in st.session_state:
    st.session_state.raw_trade_log_exporter_settings = None
if "true_raw_trade_log_report" not in st.session_state:
    st.session_state.true_raw_trade_log_report = None
if "true_raw_trade_log_settings" not in st.session_state:
    st.session_state.true_raw_trade_log_settings = None
if "forward_paper_evidence_report" not in st.session_state:
    st.session_state.forward_paper_evidence_report = None
if "forward_paper_evidence_settings" not in st.session_state:
    st.session_state.forward_paper_evidence_settings = None
if "actionable_research_plan_report" not in st.session_state:
    st.session_state.actionable_research_plan_report = None
if "actionable_research_plan_settings" not in st.session_state:
    st.session_state.actionable_research_plan_settings = None
if "daily_research_control_center_report" not in st.session_state:
    st.session_state.daily_research_control_center_report = None
if "daily_research_control_center_settings" not in st.session_state:
    st.session_state.daily_research_control_center_settings = None
if "portfolio_capital_simulator_report" not in st.session_state:
    st.session_state.portfolio_capital_simulator_report = None
if "portfolio_capital_simulator_settings" not in st.session_state:
    st.session_state.portfolio_capital_simulator_settings = None
if "risk_warning_intelligence_report" not in st.session_state:
    st.session_state.risk_warning_intelligence_report = None
if "risk_warning_intelligence_settings" not in st.session_state:
    st.session_state.risk_warning_intelligence_settings = None
if "dynamic_risk_sizing_report" not in st.session_state:
    st.session_state.dynamic_risk_sizing_report = None
if "dynamic_risk_sizing_settings" not in st.session_state:
    st.session_state.dynamic_risk_sizing_settings = None
if "market_regime_intelligence_report" not in st.session_state:
    st.session_state.market_regime_intelligence_report = None
if "market_regime_intelligence_settings" not in st.session_state:
    st.session_state.market_regime_intelligence_settings = None
if "strategy_benchmark_arena_report" not in st.session_state:
    st.session_state.strategy_benchmark_arena_report = None
if "strategy_benchmark_arena_settings" not in st.session_state:
    st.session_state.strategy_benchmark_arena_settings = None
if "historical_model_replay_report" not in st.session_state:
    st.session_state.historical_model_replay_report = None
if "historical_model_replay_settings" not in st.session_state:
    st.session_state.historical_model_replay_settings = None
if "replay_benchmark_audit_report" not in st.session_state:
    st.session_state.replay_benchmark_audit_report = None
if "replay_benchmark_audit_settings" not in st.session_state:
    st.session_state.replay_benchmark_audit_settings = None
if "signal_policy_edge_lab_report" not in st.session_state:
    st.session_state.signal_policy_edge_lab_report = None
if "signal_policy_edge_lab_settings" not in st.session_state:
    st.session_state.signal_policy_edge_lab_settings = None
if "true_historical_ml_replay_report" not in st.session_state:
    st.session_state.true_historical_ml_replay_report = None
if "true_historical_ml_replay_settings" not in st.session_state:
    st.session_state.true_historical_ml_replay_settings = None
if "unified_risk_command_center_report" not in st.session_state:
    st.session_state.unified_risk_command_center_report = None
if "unified_risk_command_center_settings" not in st.session_state:
    st.session_state.unified_risk_command_center_settings = None
if "prediction_edge_improvement_report" not in st.session_state:
    st.session_state.prediction_edge_improvement_report = None
if "prediction_edge_improvement_settings" not in st.session_state:
    st.session_state.prediction_edge_improvement_settings = None
if "artifact_store_last_save" not in st.session_state:
    st.session_state.artifact_store_last_save = None
if "phase5_audit" not in st.session_state:
    st.session_state.phase5_audit = None
if "phase5_features_preview" not in st.session_state:
    st.session_state.phase5_features_preview = None


# ════════════════════════════════════════════════════════════════
# Sidebar Navigation
# ════════════════════════════════════════════════════════════════

st.sidebar.markdown("## Market Research Assistant")
st.sidebar.caption("Simple plans first. Technical evidence remains available when needed.")
st.sidebar.markdown("---")

NAVIGATION_GROUPS = {
    "Overview Command Center": ["Overview Command Center", "Guided Research Workflow"],
    "Forecasting & Prediction": [
        "📊 Dataset Explorer", "📈 Technical Indicators", "🤖 Train Models",
        "🏆 Compare Models", "🔮 Prediction", "🎯 Directional Models",
        "🧠 Feature Intelligence", "🎯 Direct Forecast Models",
        "🧭 Direct Horizon Scanner", "📅 30-Day Forecast",
        "Walk-Forward ML Replay", "Model Edge Benchmark Lab",
    ],
    "Signal Research": [
        "📡 Signal Engine", "🧪 Signal Research Scanner", "📈 Signal Policy Sensitivity",
        "Signal Policy & Edge Repair Lab", "🧠 Regime-Aware Meta Signal",
        "🔬 Candidate Deep Diagnostics", "🛡️ Risk-Controlled Upgrade",
        "🧭 Actionable Research Plan",
    ],
    "Validation & Evidence": [
        "🧪 Research Validation", "🌐 Multi-Asset Matrix", "🧭 Walk-Forward Validation",
        "🧾 Meta Decision Audit", "🏷️ Meta Reliability Grading", "🧪 Evidence Expansion",
        "🔎 Evidence Quality Diagnostics", "🎯 Probability Calibration",
        "📈 Forward Paper Evidence", "🗂️ Evidence Store", "🧾 True Raw Trade Logs",
        "📜 Raw Trade Log Exporter", "📒 Trade Evidence Ledger",
    ],
    "Risk Intelligence": [
        "🧠 Daily Research Control Center", "💼 Portfolio & Capital Simulator",
        "⚠️ Risk & Warning Intelligence", "📐 Dynamic Risk Sizing",
        "🌍 Market Regime Intelligence",
    ],
    "Benchmarking & Replay": [
        "📉 Backtesting", "🏁 Strategy Benchmark Arena", "🕰️ Historical Model Replay",
        "Walk-Forward ML Replay", "Unified Risk Command Center",
        "Model Edge Benchmark Lab",
    ],
    "Reports & Exports": [
        "Unified Risk Command Center", "🧭 Actionable Research Plan",
        "🗂️ Evidence Store", "ℹ️ About Project",
    ],
}

# Primary users see plain product language. Internal route labels remain unchanged below.
ADVANCED_FRIENDLY_ROUTES = dict(ADVANCED_DIAGNOSTIC_PAGES)
NAVIGATION_GROUPS = {
    "Data & Features": [
        "Overview Command Center", "Guided Research Workflow", "Dataset Explorer",
        "Technical Indicators", "Feature Intelligence", "Evidence Store", "About Project",
    ],
    "Forecasting & Models": [
        "Train Models", "Compare Models", "Prediction", "Directional Models",
        "Direct Forecast Models", "Direct Horizon Scanner", "30-Day Forecast", "Model Edge Benchmark Lab",
    ],
    "Signals & Plans": [
        "Signal Engine", "Signal Research Scanner", "Signal Policy Sensitivity",
        "Signal Policy & Edge Repair Lab", "Regime-Aware Meta Signal", "Candidate Deep Diagnostics",
        "Risk-Controlled Upgrade", "Actionable Research Plan", "Daily Research Control Center",
    ],
    "Risk & Regime": [
        "Risk & Warning Intelligence", "Dynamic Risk Sizing", "Market Regime Intelligence",
        "Portfolio & Capital Simulator", "Unified Risk Command Center",
    ],
    "Backtesting & Replay": [
        "Backtesting", "Strategy Benchmark Arena", "Historical Model Replay",
        "Walk-Forward ML Replay", "Walk-Forward Validation",
    ],
    "Evidence & Quality Gates": [
        "Research Validation", "Multi-Asset Matrix", "Meta Decision Audit", "Meta Reliability Grading",
        "Evidence Expansion", "Evidence Quality Diagnostics", "Probability Calibration",
        "Forward Paper Evidence", "True Raw Trade Logs", "Raw Trade Log Exporter", "Trade Evidence Ledger",
    ],
}

LEGACY_PRIMARY_NAVIGATION = list(PRIMARY_USER_PAGES) + ["Advanced Diagnostics"]
PRIMARY_PRODUCT_PAGES = [
    "Market Research Assistant",
    "Asset Plans",
    "Forecast Explorer",
    "Cost-Aware Plan",
    "Portfolio Summary",
    "Paper Research Journey",
    "About / Methodology",
]

experience_page = st.sidebar.radio(
    "Navigate",
    PRIMARY_PRODUCT_PAGES + ["Advanced Diagnostics"],
    key="primary_product_navigation",
)
is_advanced_diagnostic = experience_page == "Advanced Diagnostics"
if is_advanced_diagnostic:
    navigation_group = st.sidebar.selectbox("Diagnostic area", list(NAVIGATION_GROUPS), key="navigation_group")
    group_pages = NAVIGATION_GROUPS[navigation_group]
    friendly_page = group_pages[0] if len(group_pages) == 1 else st.sidebar.selectbox("Diagnostic page", group_pages, key=f"page_{navigation_group}")
    page_label = ADVANCED_FRIENDLY_ROUTES.get(friendly_page, friendly_page)
else:
    page_label = experience_page

asset_names = get_supported_assets()
PAGE_ROUTE_ALIASES = {
    "Signal Policy & Edge Repair Lab": "Phase 19: Signal Policy & Edge Repair Lab",
    "Walk-Forward ML Replay": "Phase 20: True Historical ML Replay",
    "Unified Risk Command Center": "Phase 21: Unified Risk Command Center",
    "Model Edge Benchmark Lab": "Phase 22: Prediction Edge Improvement",
}
page = PAGE_ROUTE_ALIASES.get(page_label, page_label)
default_asset_index = asset_names.index(st.session_state.selected_asset) if st.session_state.selected_asset in asset_names else 0
selected_asset = st.sidebar.selectbox("Research Asset", asset_names, index=default_asset_index)
st.session_state.selected_asset = selected_asset
target_col = get_asset_target(selected_asset)

horizon_options = get_available_horizons()
default_horizon_index = horizon_options.index(st.session_state.selected_horizon) if st.session_state.selected_horizon in horizon_options else horizon_options.index(DEFAULT_HORIZON)
selected_horizon = st.sidebar.selectbox(
    "Research Horizon",
    horizon_options,
    index=default_horizon_index,
    format_func=lambda value: f"{int(value)}D",
)
validate_asset_horizon(selected_asset, selected_horizon)
st.session_state.selected_horizon = int(selected_horizon)
st.sidebar.caption("Specialized scanners may override the central horizon for batch research.")

if _asset_mismatch(selected_asset):
    st.sidebar.warning(
        f"Current trained models are for {st.session_state.trained_asset}. "
        f"Train again to use {selected_asset}."
    )

st.sidebar.markdown("---")
st.sidebar.caption("Research assistant | Real-money decisions disabled")

if is_advanced_diagnostic:
    st.info("Advanced diagnostic page. Normal users should use Market Research Assistant or Asset Plans first.")


# ════════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════════

if page == "Market Research Assistant":
    render_hero_section(
        "Multi-Asset Research Intelligence",
        "Track market ideas with forecasts, costs, risk, and benchmarks in one place",
        "Run the research engine, compare active estimates against passive benchmarks, and get a simple paper-research plan.",
    )
    render_disclaimer_banner()
    render_blocked_capital_banner()

    hero_a, hero_b, hero_c, hero_d = st.columns(4)
    with hero_a:
        run_full_clicked = st.button("Run Full Research", type="primary", width="stretch", key="phase29_run_full")
    with hero_b:
        refresh_market_clicked = st.button("Refresh Market Data", width="stretch", key="phase29_refresh_market")
    with hero_c:
        st.button(
            "View Cost-Aware Plan", width="stretch", key="phase29_open_cost_plan",
            on_click=_navigate_to_plan,
            args=(selected_asset, "Cost-Aware Plan", int(selected_horizon)),
        )
    with hero_d:
        st.button(
            "Open Paper Research Journey", width="stretch", key="phase29_open_paper_journey",
            on_click=_navigate_to_plan,
            args=(selected_asset, "Paper Research Journey", int(selected_horizon)),
        )

    render_run_research_panel()
    if run_full_clicked or refresh_market_clicked:
        with st.status("Building the final research snapshot...", expanded=True) as run_status:
            for step in (
                "Loading latest prices", "Checking saved forecasts", "Checking risk",
                "Comparing passive benchmarks", "Estimating costs", "Building final plans",
            ):
                st.write(step)
            try:
                phase29_report = run_full_user_research(
                    selected_assets=get_supported_assets(), selected_horizons=get_available_horizons(),
                    amount=10000, cost_assumptions=default_cost_assumptions(),
                    refresh=bool(refresh_market_clicked),
                )
                st.session_state.phase29_user_report = phase29_report
                st.session_state.phase26_research_snapshot = phase29_report.get("ResearchSnapshot")
                st.session_state.phase26_asset_plans = phase29_report.get("AssetPlans")
                _latest_user_price_snapshot.clear()
                for warning in phase29_report.get("Warnings", []):
                    st.warning(warning)
                st.write("Saving snapshot")
                run_status.update(label="Final research snapshot ready", state="complete", expanded=False)
            except Exception as exc:
                st.warning(f"Some research evidence could not be combined: {exc}")
                run_status.update(label="Completed with missing evidence", state="error", expanded=True)

    phase29_report = st.session_state.get("phase29_user_report")
    phase29_snapshot = (
        phase29_report.get("AllAssetPredictionSnapshot", pd.DataFrame())
        if isinstance(phase29_report, dict)
        else _load_phase29_table("phase29_all_asset_prediction_snapshot.csv")
    )
    if not isinstance(phase29_snapshot, pd.DataFrame) or phase29_snapshot.empty:
        phase29_snapshot = _phase29_placeholder_snapshot(_latest_user_price_snapshot())
    _render_phase29_snapshot(phase29_snapshot)
    hero_generate_clicked = False

    render_glass_container(
        "Research controls",
        "Choose a focus or keep the full multi-asset view. The assistant reads saved evidence and does not rerun expensive models on page load.",
    )

    assistant_a, assistant_b, assistant_c = st.columns(3)
    with assistant_a:
        assistant_asset = st.selectbox("Asset", ["All assets"] + get_supported_assets(), index=0, key="phase26_assistant_asset")
    with assistant_b:
        assistant_horizon = st.selectbox(
            "Horizon",
            ["All horizons"] + [f"{value}D" for value in get_available_horizons()],
            index=0,
            key="phase26_assistant_horizon",
        )
    with assistant_c:
        show_advanced_evidence = st.checkbox("Show advanced evidence", value=False, key="phase26_show_advanced_evidence")

    action_a, action_b = st.columns(2)
    with action_a:
        generate_plan_clicked = st.button("Generate Research Plan", type="primary", width="stretch")
    with action_b:
        refresh_snapshot_clicked = st.button("Refresh Research Snapshot", width="stretch")

    if hero_generate_clicked or generate_plan_clicked or refresh_snapshot_clicked:
        with st.spinner("Reviewing saved research evidence..."):
            phase26_snapshot = run_research_engine(
                selected_assets=get_supported_assets(),
                selected_horizons=get_available_horizons(),
                refresh=bool(refresh_snapshot_clicked),
            )
            phase26_plans = generate_all_asset_plans(phase26_snapshot)
            phase26_portfolio = generate_portfolio_plan(phase26_plans)
            save_product_experience_artifacts(phase26_snapshot, phase26_plans, phase26_portfolio)
            save_premium_product_artifacts(
                phase26_plans,
                phase26_portfolio,
                app_source=Path(__file__).read_text(encoding="utf-8"),
            )
            st.session_state.phase26_research_snapshot = phase26_snapshot
            st.session_state.phase26_asset_plans = phase26_plans
            st.session_state.phase26_portfolio_plan = phase26_portfolio

    phase26_snapshot = st.session_state.get("phase26_research_snapshot")
    if not isinstance(phase26_snapshot, pd.DataFrame):
        phase26_snapshot = load_latest_research_snapshot()
    phase26_plans = st.session_state.get("phase26_asset_plans")
    if not isinstance(phase26_plans, pd.DataFrame):
        phase26_plans = _load_phase26_table("phase26_asset_plans")

    if phase26_plans.empty:
        render_empty_state(
            "No research plan generated yet",
            "Use Generate Research Plan to review the latest saved evidence. Missing evidence will stay visible rather than being treated as neutral.",
        )
    else:
        phase26_plans = rank_asset_plans(phase26_plans)
        phase26_portfolio = generate_portfolio_plan(phase26_plans)
        display_plans = phase26_plans.copy()
        if assistant_asset != "All assets":
            display_plans = display_plans[display_plans["Asset"].astype(str).eq(assistant_asset)]
        if assistant_horizon != "All horizons":
            chosen_horizon = int(str(assistant_horizon).replace("D", ""))
            display_plans = display_plans[pd.to_numeric(display_plans["Horizon"], errors="coerce").eq(chosen_horizon)]
        display_plans = rank_asset_plans(display_plans)

        counts = display_plans["Status"].value_counts()
        closest = display_plans.iloc[0] if not display_plans.empty else pd.Series(dtype=object)
        closest_label = f"{closest.get('Asset', 'None')} {int(closest.get('Horizon', 0))}D" if not closest.empty else "None"
        render_section_header("Research snapshot", "Opportunity scores rank research closeness, not investment attractiveness.")
        render_metric_grid(
            [
                {"title": "Closest to Track", "value": closest_label, "subtitle": f"Score {float(closest.get('OpportunityScore', 0)):.0f}/100 · status stays {closest.get('Status', 'Unknown')}", "status": _status_card_style(str(closest.get("Status", "")))},
                {"title": "Watchlist", "value": int(counts.get("Watch", 0)), "subtitle": "Interesting, but evidence remains limited", "status": "info"},
                {"title": "High Risk", "value": int(counts.get("Avoid", 0) + counts.get("High Risk", 0)), "subtitle": "Risk evidence dominates opportunity evidence", "status": "critical" if counts.get("Avoid", 0) + counts.get("High Risk", 0) else "neutral"},
                {"title": "Data Issues", "value": int(counts.get("Data Issue", 0)), "subtitle": "Repair data before interpretation", "status": "warning" if counts.get("Data Issue", 0) else "neutral"},
            ]
        )

        portfolio_row = phase26_portfolio.iloc[0]
        render_section_header("Portfolio condition", "A plain-language reading of the full research set.")
        condition_a, condition_b = st.columns(2)
        with condition_a:
            render_risk_explanation_card(
                str(portfolio_row.get("OverallResearchCondition", "Evidence constrained")),
                str(portfolio_row.get("WhySystemIsCautious", "Risk and evidence limits keep the system cautious.")),
                f"Main risk theme: {portfolio_row.get('MainRiskTheme', portfolio_row.get('MainMarketRisk', 'Unknown'))}",
            )
        with condition_b:
            render_monitoring_card(
                f"Best available opportunity: {portfolio_row.get('ClosestToTrack', closest_label)}",
                str(portfolio_row.get("WhatUserShouldMonitorNext", "Review forward outcomes and risk warnings.")),
                str(portfolio_row.get("NextReviewTrigger", "After the next saved evidence refresh.")),
            )

        risk_share = display_plans["Status"].isin(["High Risk", "Avoid"]).mean() if not display_plans.empty else 0.0
        if risk_share >= 0.5:
            render_section_header("Why are many assets marked High Risk?")
            render_risk_explanation_card(
                "The system is intentionally conservative",
                str(display_plans.iloc[0].get("WhyEverythingIsHighRisk", "Risk warnings are stronger than opportunity signals.")),
                "A forecast alone is not enough: benchmark support, regime stability, data freshness, and repeatable evidence must also improve.",
            )

        render_section_header(
            "Closest to becoming trackable",
            "These are the least blocked research combinations. Their underlying statuses are not upgraded by the ranking.",
        )
        opportunity_rows = display_plans.head(6)
        for row_start in range(0, len(opportunity_rows), 2):
            opportunity_columns = st.columns(2)
            for column, (_, opportunity) in zip(opportunity_columns, opportunity_rows.iloc[row_start:row_start + 2].iterrows()):
                with column:
                    render_opportunity_card(opportunity.to_dict())

        with st.expander("What changed since the previous snapshot?", expanded=False):
            st.info("No earlier product snapshot is loaded for comparison. A later refresh can be compared without changing research calculations.")

        show_more_plans = st.checkbox("Show more asset plan cards", value=False, key="phase27_show_more_plans")
        render_section_header("Asset plans", "Each card explains the blocker, improvement needed, and next review trigger.")
        _render_asset_plan_cards(display_plans if show_more_plans else display_plans.head(6), show_advanced=show_advanced_evidence)
        if show_advanced_evidence:
            with st.expander("Raw evidence snapshot", expanded=False):
                if phase26_snapshot.empty:
                    st.info("No normalized evidence rows are available.")
                else:
                    st.dataframe(phase26_snapshot.head(1000), width="stretch", hide_index=True)


elif page == "Paper Research Journey":


    st.markdown("<div class='premium-pill'>Paper Research Journey</div>", unsafe_allow_html=True)


    st.markdown("# Build trust through paper research")


    st.caption("Track ideas safely, compare them against passive benchmarks, and learn when the system is useful â€” without real-money decisions.")


    st.info("This is a research assistant, not financial advice. It does not execute trades or approve real-money decisions.")



    try:


        from src.paper_research_journey import (


            build_paper_research_journey,


            create_paper_research_plan,


            generate_passive_benchmark_guides,


            save_phase28_artifacts,


        )


        from src.user_plan_generator import generate_all_asset_plans


    except Exception as exc:


        st.error(f"Paper Research Journey is unavailable: {exc}")


        st.stop()



    try:


        plans = st.session_state.get("phase26_asset_plans")
        if not isinstance(plans, pd.DataFrame):
            plans = _load_phase26_table("phase26_asset_plans")
        if not isinstance(plans, pd.DataFrame) or plans.empty:
            plans = generate_all_asset_plans(pd.DataFrame())


        journey = build_paper_research_journey(plans)


        save_phase28_artifacts(journey)


    except Exception as exc:


        st.warning(f"Using limited paper research view because latest plan artifacts were incomplete: {exc}")


        journey = build_paper_research_journey([])



    trust = journey.get("TrustScorecard", {})


    c1, c2, c3, c4 = st.columns(4)


    c1.metric("Trust score", trust.get("TrustScore", 10))


    c2.metric("Trust label", trust.get("TrustLabel", "New"))


    c3.metric("Completed paper plans", trust.get("CompletedPaperPlans", 0))


    c4.metric("Benchmark wins", trust.get("BenchmarkBeatenCount", 0))



    st.markdown("## Why use this if passive benchmarks often win?")


    st.write("Passive benchmarks are hard to beat. The value of this platform is not to force active ideas; it is to structure paper tests, compare every idea against a benchmark, and avoid trusting weak forecasts blindly.")



    st.markdown("## Closest candidates to track in paper research")


    candidates = journey.get("Candidates", [])


    if not candidates:


        st.warning("No candidates are available yet. Generate an asset plan first, then return here.")


    else:


        for row in candidates[:8]:


            with st.container(border=True):


                top = st.columns([1.2, 0.8, 0.8, 1.2])


                top[0].markdown(f"### {row.get('Asset')} Â· {row.get('Horizon')}")


                top[1].metric("Opportunity", row.get("OpportunityScore", 0))


                top[2].write(f"**Status:** {row.get('Status', 'Not Enough Evidence')}")


                top[3].write(f"**Recheck:** {row.get('RecheckPriority', 'Medium')}")


                st.write(f"**What is blocking it:** {row.get('MainBlocker', 'Evidence is not strong enough yet.')}")


                st.write(f"**What must improve:** {row.get('WhatMustImprove', 'Benchmark evidence and risk conditions must improve.')}")


                st.write(f"**What to monitor next:** {row.get('WhatUserShouldMonitorNext', 'Monitor risk warnings, data freshness, and benchmark gap.')}")


                st.write(f"**Passive benchmark:** {row.get('PassiveBenchmarkName', 'Passive benchmark reference')}")


                st.caption(row.get("BenchmarkWarning", "Benchmark comparison is research-only and does not approve real-money decisions."))



    st.markdown("## Start a simulated paper research plan")


    assets = sorted({row.get("Asset", "Gold") for row in candidates}) or ["Gold", "Silver", "Crude Oil", "Bitcoin", "S&P 500", "Gold ETF"]


    selected_asset = st.selectbox("Asset", assets, key="phase28_asset")


    matching = [row for row in candidates if row.get("Asset") == selected_asset]


    horizons = sorted({row.get("Horizon", "30D") for row in matching}) or ["1D", "5D", "10D", "20D", "30D"]


    selected_horizon = st.selectbox("Horizon", horizons, key="phase28_horizon")


    amount = st.number_input("Simulated amount", min_value=0.0, value=10000.0, step=1000.0, key="phase28_amount")


    notes = st.text_area("Research notes", value="", key="phase28_notes")

    with st.expander("Cost-aware paper assumptions", expanded=False):
        st.info(COST_DISCLAIMER)
        paper_cost_assumptions = render_cost_assumption_inputs(
            default_cost_assumptions(selected_asset), key_prefix="phase28_cost",
        )


    if st.button("Create paper research plan", key="phase28_create_plan"):


        base = next((row for row in candidates if row.get("Asset") == selected_asset and row.get("Horizon") == selected_horizon), {})


        plan = create_paper_research_plan(selected_asset, selected_horizon, base, simulated_amount=amount, notes=notes)
        cost_enriched = generate_cost_aware_asset_plan(
            {**base, "Asset": selected_asset, "Horizon": selected_horizon},
            amount=amount, cost_assumptions=paper_cost_assumptions,
        )
        for field in (
            "EstimatedRoundTripCost", "CostDragPct", "BreakEvenReturnPct", "GrossActiveEstimatePct",
            "NetActiveEstimatePct", "GrossPassiveEstimatePct", "NetPassiveEstimatePct",
            "ActiveMinusPassiveNetPct", "CostVerdict", "CostWarning", "ActiveVsPassiveLesson",
        ):
            plan[field] = cost_enriched.get(field)
        plan["CostAssumptions"] = paper_cost_assumptions
        plan["TrustCostNote"] = "Trust should improve only when repeated net outcomes survive costs and compare well with the passive benchmark."


        st.session_state.setdefault("phase28_active_paper_plans", []).append(plan)


        st.success("Paper research plan created for simulated tracking.")


        st.json(plan)



    active = st.session_state.get("phase28_active_paper_plans", [])


    st.markdown("## Active paper research plans")


    if not active:


        st.info("No active paper research plans in this session yet. Create one above and review it after the horizon matures.")


    else:


        for plan in active:


            with st.expander(f"{plan.get('Asset')} Â· {plan.get('Horizon')} Â· {plan.get('PlanId')}"):


                st.write(f"**Status at start:** {plan.get('StatusAtStart')}")


                st.write(f"**Benchmark:** {plan.get('BenchmarkToCompare')}")


                st.write(f"**Cost verdict:** {plan.get('CostVerdict', 'MissingEstimate')}")


                st.write(f"**Break-even return:** {plan.get('BreakEvenReturnPct', 'Not available')}%")


                st.write(f"**What to monitor:** {plan.get('WhatUserShouldMonitorNext')}")


                st.write(f"**Invalidation condition:** {plan.get('InvalidationCondition')}")


                st.caption(plan.get("BenchmarkWarning"))



    st.markdown("## What is the passive benchmark?")


    guides = generate_passive_benchmark_guides()


    for guide in guides:


        with st.expander(f"{guide['Asset']} â€” {guide['PassiveBenchmarkName']}"):


            st.write(guide["Explanation"])


            st.write(f"**How to follow in research mode:** {guide['HowToFollow']}")


            st.write(f"**What to compare:** {guide['WhatToCompare']}")


            st.warning(guide["Warning"])



elif page == "Asset Plans":
    render_premium_header(
        "Asset Plans",
        "Compare every configured asset and horizon through conservative, plain-language research cards.",
        "Opportunity ranking",
    )
    render_disclaimer_banner()
    asset_plans = st.session_state.get("phase26_asset_plans")
    if not isinstance(asset_plans, pd.DataFrame):
        asset_plans = _load_phase26_table("phase26_asset_plans")
    portfolio_plan = st.session_state.get("phase26_portfolio_plan")
    if not isinstance(portfolio_plan, pd.DataFrame):
        portfolio_plan = _load_phase26_table("phase26_portfolio_plan")
    if asset_plans.empty:
        render_empty_state(
            "No asset plans available",
            "Open Market Research Assistant and generate a plan from the latest saved evidence first.",
        )
    else:
        ranked_plans = rank_asset_plans(asset_plans)
        portfolio_plan = generate_portfolio_plan(ranked_plans)
        filter_choice = render_status_tabs(
            ["All", "Closest to Track", "Watch", "Wait", "High Risk", "Data Issues"],
            key="phase27_asset_status_filter",
        )
        filter_a, filter_b, filter_c = st.columns(3)
        with filter_a:
            asset_focus = st.selectbox("Asset focus", ["All assets"] + get_supported_assets(), key="phase26_asset_plan_focus")
        with filter_b:
            sort_choice = st.selectbox(
                "Sort by", ["OpportunityScore", "RecheckPriority", "Confidence", "Asset", "Horizon"],
                key="phase27_asset_plan_sort",
            )
        with filter_c:
            show_plan_evidence = st.checkbox("Show advanced evidence", value=False, key="phase27_asset_plan_evidence")

        display_asset_plans = ranked_plans.copy()
        if asset_focus != "All assets":
            display_asset_plans = display_asset_plans[display_asset_plans["Asset"].astype(str).eq(asset_focus)]
        status_filters = {
            "Watch": ["Watch"], "Wait": ["Wait", "Not Enough Evidence"],
            "High Risk": ["High Risk", "Avoid"], "Data Issues": ["Data Issue"],
        }
        if filter_choice == "Closest to Track":
            display_asset_plans = display_asset_plans.nsmallest(8, "ClosestToTrackRank")
        elif filter_choice in status_filters:
            display_asset_plans = display_asset_plans[display_asset_plans["Status"].isin(status_filters[filter_choice])]

        if sort_choice == "OpportunityScore":
            display_asset_plans = display_asset_plans.sort_values("OpportunityScore", ascending=False)
        elif sort_choice == "RecheckPriority":
            priority_order = {"High": 0, "Medium": 1, "Low": 2}
            display_asset_plans = display_asset_plans.assign(
                _sort=display_asset_plans["RecheckPriority"].map(priority_order).fillna(3)
            ).sort_values(["_sort", "OpportunityScore"], ascending=[True, False]).drop(columns="_sort")
        elif sort_choice == "Confidence":
            confidence_order = {"Higher": 0, "Moderate": 1, "Low": 2}
            display_asset_plans = display_asset_plans.assign(
                _sort=display_asset_plans["Confidence"].map(confidence_order).fillna(3)
            ).sort_values(["_sort", "OpportunityScore"], ascending=[True, False]).drop(columns="_sort")
        else:
            display_asset_plans = display_asset_plans.sort_values(sort_choice)

        if display_asset_plans.empty:
            render_empty_state("No plans match this view", "Change the status or asset filter to review another part of the research set.")
        else:
            _render_asset_plan_cards(display_asset_plans, show_advanced=show_plan_evidence)
            detail_options = [f"{row.Asset} · {int(row.Horizon)}D" for row in display_asset_plans.itertuples()]
            detail_choice = st.selectbox("Beginner plan detail", detail_options, key="phase29_asset_plan_detail")
            detail_index = detail_options.index(detail_choice)
            detail_row = display_asset_plans.iloc[detail_index].to_dict()
            saved_final = _load_phase29_table("phase29_all_asset_prediction_snapshot.csv")
            final_match = saved_final[
                saved_final["Asset"].astype(str).eq(str(detail_row.get("Asset")))
                & pd.to_numeric(saved_final["BestHorizon"], errors="coerce").eq(int(detail_row.get("Horizon", 0)))
            ] if not saved_final.empty else pd.DataFrame()
            if not final_match.empty:
                detail_row.update(final_match.iloc[0].to_dict())
            else:
                detail_row = generate_cost_aware_asset_plan(detail_row, amount=10000, cost_assumptions=default_cost_assumptions(str(detail_row.get("Asset"))))
                detail_row["SimplePlan"] = generate_final_user_plan(detail_row)
            detail_a, detail_b = st.columns(2)
            with detail_a:
                render_score_explainer_card(detail_row)
                render_cost_summary_card(detail_row)
            with detail_b:
                render_active_vs_passive_card(detail_row)
                render_simple_plan_card(detail_row)

        high_risk_explanations = build_high_risk_explanations(ranked_plans)
        monitoring_plan = build_monitoring_plan(ranked_plans)
        render_section_header("Downloads", "Export the rankings, complete plan cards, and portfolio interpretation.")
        render_download_buttons(
            {
                "Opportunity rankings": (ranked_plans, "phase27_opportunity_rankings.csv"),
                "Asset plan cards": (ranked_plans, "phase27_asset_plan_cards.csv"),
                "Portfolio summary": (portfolio_plan, "phase27_portfolio_summary.csv"),
                "High-risk explanations": (high_risk_explanations, "phase27_high_risk_explanations.csv"),
                "Monitoring plan": (monitoring_plan, "phase27_monitoring_plan.csv"),
            }
        )


elif page == "Forecast Explorer":
    render_premium_header(
        "Forecast Explorer",
        "Inspect correctly routed price history and saved forecast evidence for one asset and horizon.",
        "Research estimates",
    )
    render_disclaimer_banner()
    render_glass_container("Forecasts are research estimates, not guarantees.", "Use the forecast with benchmark, regime, risk, and data-freshness evidence rather than as a standalone decision.")
    forecast_a, forecast_b, forecast_c = st.columns(3)
    with forecast_a:
        explorer_asset = st.selectbox("Asset", get_supported_assets(), index=get_supported_assets().index(selected_asset), key="phase26_forecast_asset")
    with forecast_b:
        explorer_horizon = st.selectbox(
            "Horizon",
            get_available_horizons(),
            index=get_available_horizons().index(int(selected_horizon)),
            format_func=lambda value: f"{int(value)}D",
            key="phase26_forecast_horizon",
        )
    with forecast_c:
        history_window = st.selectbox("History window", ["90 days", "180 days", "1 year", "3 years", "Full history"], index=1, key="phase26_history_window")
    validate_asset_horizon(explorer_asset, explorer_horizon)
    explorer_target = get_asset_target(explorer_asset)
    market_history = _load_cached_market_history()
    history_rows = {"90 days": 90, "180 days": 180, "1 year": 252, "3 years": 756, "Full history": None}[history_window]
    if market_history.empty or explorer_target not in market_history.columns:
        render_empty_state(
            f"No cached {explorer_asset} history",
            "Refresh data from Advanced Diagnostics before using this view. The app will not substitute another asset.",
        )
    else:
        selected_history = market_history[[explorer_target]].dropna()
        if history_rows is not None:
            selected_history = selected_history.tail(history_rows)
        render_section_header(f"{explorer_asset} price history", f"Target column: {explorer_target} · window: {history_window}")
        st.line_chart(selected_history, width="stretch")

    render_section_header("How to read this forecast")
    how_a, how_b = st.columns(2)
    with how_a:
        render_navigation_card(
            "Forecast line and range",
            "The line represents the saved research estimate. Any range represents uncertainty, not a guaranteed path.",
            "Compare the selected horizon with current history",
        )
    with how_b:
        render_navigation_card(
            "Why it should not be used alone",
            "A plausible forecast can still fail benchmarks, occur in an unstable regime, or carry unacceptable drawdown and data risk.",
            "Review the matching Asset Plan",
        )

    explorer_snapshot = st.session_state.get("phase26_research_snapshot")
    if not isinstance(explorer_snapshot, pd.DataFrame):
        explorer_snapshot = load_latest_research_snapshot()
    explorer_evidence = collect_asset_horizon_evidence(explorer_asset, int(explorer_horizon), explorer_snapshot)
    explorer_plans = st.session_state.get("phase26_asset_plans")
    if not isinstance(explorer_plans, pd.DataFrame):
        explorer_plans = _load_phase26_table("phase26_asset_plans")
    if not explorer_plans.empty:
        explorer_plans = rank_asset_plans(explorer_plans)
    current_plan = explorer_plans[
        explorer_plans.get("Asset", pd.Series(dtype=str)).astype(str).eq(explorer_asset)
        & pd.to_numeric(explorer_plans.get("Horizon", pd.Series(dtype=float)), errors="coerce").eq(int(explorer_horizon))
    ] if not explorer_plans.empty else pd.DataFrame()
    if not current_plan.empty:
        row = current_plan.iloc[0]
        render_status_card("Current research status", row.get("Status", "Not Enough Evidence"), row.get("Summary", ""), _status_card_style(str(row.get("Status", ""))))

    final_report = st.session_state.get("phase29_user_report")
    final_snapshot = (
        final_report.get("AllAssetPredictionSnapshot", pd.DataFrame())
        if isinstance(final_report, dict)
        else _load_phase29_table("phase29_all_asset_prediction_snapshot.csv")
    )
    matching_final = final_snapshot[final_snapshot["Asset"].astype(str).eq(explorer_asset)] if isinstance(final_snapshot, pd.DataFrame) and not final_snapshot.empty else pd.DataFrame()
    if not matching_final.empty and int(matching_final.iloc[0].get("BestHorizon", 0) or 0) == int(explorer_horizon):
        render_prediction_snapshot_card(matching_final.iloc[0].to_dict())
        render_cost_summary_card(matching_final.iloc[0].to_dict())
    else:
        latest_rows = _latest_user_price_snapshot()
        latest_match = latest_rows[latest_rows["Asset"].astype(str).eq(explorer_asset)] if not latest_rows.empty else pd.DataFrame()
        if not latest_match.empty:
            render_metric_grid([{
                "title": "Current price", "value": f"{float(latest_match.iloc[0].get('LatestPrice')):,.2f}",
                "subtitle": f"Latest available dataset price · {latest_match.iloc[0].get('LatestPriceDate', '')}", "status": "info",
            }])
        st.info("No matching saved predicted price is available for this asset and horizon. The app does not substitute another horizon or asset.")

    forecast_mask = explorer_evidence.get("Metric", pd.Series(dtype=str)).astype(str).str.contains("forecast|predicted|probability", case=False, regex=True, na=False)
    forecast_table = explorer_evidence.loc[forecast_mask, ["Asset", "Horizon", "Metric", "Value", "Status", "Freshness"]].copy()
    render_safe_table(
        forecast_table,
        f"{explorer_asset} {int(explorer_horizon)}D Forecast Evidence",
        "No saved forecast evidence is available for this asset and horizon. This is not treated as a neutral forecast.",
    )
    if not forecast_table.empty:
        st.download_button(
            "Download forecast evidence",
            data=forecast_table.to_csv(index=False).encode("utf-8"),
            file_name=f"{_safe_filename_part(explorer_asset)}_{int(explorer_horizon)}d_forecast_evidence.csv",
            mime="text/csv",
        )
    with st.expander("Show technical evidence", expanded=False):
        st.dataframe(explorer_evidence, width="stretch", hide_index=True)


elif page == "Cost-Aware Plan":
    render_hero_section(
        "Paper-simulation cost reality",
        "Understand the cost before tracking an idea",
        "Adjust brokerage, spread, slippage, taxes, fees, and other assumptions to see how much movement is needed just to break even.",
    )
    render_disclaimer_banner()
    st.info(COST_DISCLAIMER)

    saved_snapshot = st.session_state.get("phase29_user_report")
    cost_snapshot = (
        saved_snapshot.get("AllAssetPredictionSnapshot", pd.DataFrame())
        if isinstance(saved_snapshot, dict)
        else _load_phase29_table("phase29_all_asset_prediction_snapshot.csv")
    )
    asset_plans = st.session_state.get("phase26_asset_plans")
    if not isinstance(asset_plans, pd.DataFrame):
        asset_plans = _load_phase26_table("phase26_asset_plans")
    if not isinstance(asset_plans, pd.DataFrame):
        asset_plans = pd.DataFrame()

    cost_a, cost_b, cost_c = st.columns(3)
    if st.session_state.get("phase29_cost_asset") not in get_supported_assets():
        st.session_state.phase29_cost_asset = selected_asset
    if st.session_state.get("phase29_cost_horizon") not in get_available_horizons():
        st.session_state.phase29_cost_horizon = int(selected_horizon)
    with cost_a:
        cost_asset = st.selectbox("Asset", get_supported_assets(), key="phase29_cost_asset")
    with cost_b:
        cost_horizon = st.selectbox(
            "Horizon", get_available_horizons(), format_func=lambda value: f"{int(value)}D",
            key="phase29_cost_horizon",
        )
    with cost_c:
        simulated_amount = st.number_input("Simulated amount", min_value=0.0, value=10000.0, step=1000.0, key="phase29_cost_amount")

    render_section_header("Editable cost assumptions", "These are paper-simulation inputs, not broker quotes.")
    assumptions = render_cost_assumption_inputs(default_cost_assumptions(cost_asset), key_prefix="phase29_cost")
    assumptions["Amount"] = simulated_amount

    selected_plan = {}
    if not asset_plans.empty:
        matches = asset_plans[
            asset_plans["Asset"].astype(str).eq(cost_asset)
            & pd.to_numeric(asset_plans["Horizon"], errors="coerce").eq(int(cost_horizon))
        ]
        if not matches.empty:
            selected_plan = matches.iloc[0].to_dict()
    if isinstance(cost_snapshot, pd.DataFrame) and not cost_snapshot.empty:
        snapshot_match = cost_snapshot[cost_snapshot["Asset"].astype(str).eq(cost_asset)]
        if not snapshot_match.empty and int(snapshot_match.iloc[0].get("BestHorizon", 0) or 0) == int(cost_horizon):
            selected_plan.update(snapshot_match.iloc[0].to_dict())
    research_snapshot = st.session_state.get("phase26_research_snapshot")
    if not isinstance(research_snapshot, pd.DataFrame):
        research_snapshot = load_latest_research_snapshot()
    exact_estimates = resolve_horizon_estimates(
        cost_asset,
        int(cost_horizon),
        research_snapshot=research_snapshot,
        master_dataset=_load_cached_market_history(),
    )
    selected_plan.update({key: value for key, value in exact_estimates.items() if value is not None})
    selected_plan.update({"Asset": cost_asset, "Horizon": int(cost_horizon)})
    cost_plan = generate_cost_aware_asset_plan(
        selected_plan, amount=simulated_amount, cost_assumptions=assumptions,
    )
    cost_plan["SimplePlan"] = generate_final_user_plan(cost_plan)

    render_section_header("Cost summary", "Gross estimates are reduced by the entered round-trip assumptions.")
    render_metric_grid([
        {"title": "Simulated amount", "value": f"{simulated_amount:,.2f}", "subtitle": "Paper simulation only", "status": "neutral"},
        {"title": "Round-trip cost", "value": f"{float(cost_plan.get('EstimatedRoundTripCost', 0)):,.2f}", "subtitle": COST_DISCLAIMER, "status": "warning"},
        {"title": "Cost drag", "value": f"{float(cost_plan.get('CostDragPct', 0)):.2f}%", "subtitle": str(cost_plan.get("CostVerdict", "MissingEstimate")), "status": "warning"},
        {"title": "Break-even return", "value": f"{float(cost_plan.get('BreakEvenReturnPct', 0)):.2f}%", "subtitle": "Movement needed to cover assumptions", "status": "info"},
        {"title": "Predicted active move", "value": "Run Full Research first" if pd.isna(pd.to_numeric(cost_plan.get("GrossActiveEstimatePct"), errors="coerce")) else f"{float(cost_plan.get('GrossActiveEstimatePct')):.2f}%", "subtitle": str(cost_plan.get("ActiveEstimateExplanation", "No saved estimate for this horizon yet")), "status": "neutral"},
        {"title": "Passive benchmark move", "value": "No forward estimate" if pd.isna(pd.to_numeric(cost_plan.get("GrossPassiveEstimatePct"), errors="coerce")) else f"{float(cost_plan.get('GrossPassiveEstimatePct')):.2f}%", "subtitle": str(cost_plan.get("PassiveEstimateExplanation", "The passive benchmark remains a comparison reference")), "status": "neutral"},
        {"title": "Net active estimate", "value": "Run Full Research first" if pd.isna(pd.to_numeric(cost_plan.get("NetActiveEstimatePct"), errors="coerce")) else f"{float(cost_plan.get('NetActiveEstimatePct')):.2f}%", "subtitle": str(cost_plan.get("ActiveEstimateExplanation", "No saved estimate for this horizon yet")), "status": "info"},
        {"title": "Net passive estimate", "value": "Awaiting benchmark estimate" if pd.isna(pd.to_numeric(cost_plan.get("NetPassiveEstimatePct"), errors="coerce")) else f"{float(cost_plan.get('NetPassiveEstimatePct')):.2f}%", "subtitle": str(cost_plan.get("PassiveEstimateExplanation", "The passive benchmark remains a comparison reference")), "status": "info"},
    ])
    if not bool(cost_plan.get("ActiveEstimateAvailable", False)):
        st.info(str(cost_plan.get("ActiveEstimateExplanation", "Run Full Research first to generate an active estimate.")))
    if not bool(cost_plan.get("PassiveEstimateAvailable", False)):
        st.info(str(cost_plan.get("PassiveEstimateExplanation", "No passive benchmark estimate is available for this horizon yet. The benchmark is still shown as a comparison reference.")))
    if str(cost_plan.get("EstimateComparisonStatus", "")) == "MissingEstimate":
        render_beginner_explanation_box(
            "Missing estimate",
            str(cost_plan.get("CostComparisonExplanation", "Cost comparison will appear after active/passive estimates are available.")),
        )
    detail_a, detail_b = st.columns(2)
    with detail_a:
        render_cost_summary_card(cost_plan)
        render_score_explainer_card(cost_plan)
    with detail_b:
        render_active_vs_passive_card(cost_plan)
        render_beginner_explanation_box("What this means", str(cost_plan.get("BeginnerExplanation", "A complete estimate is not available yet.")))
    render_simple_plan_card(cost_plan)
    with st.expander("Advanced calculations", expanded=False):
        st.json({key: cost_plan.get(key) for key in (
            "EstimatedEntryCost", "EstimatedExitCost", "EstimatedRoundTripCost", "CostDragPct",
            "BreakEvenReturnPct", "GrossActiveEstimatePct", "NetActiveEstimatePct",
            "GrossPassiveEstimatePct", "NetPassiveEstimatePct", "ActiveMinusPassiveNetPct",
        )})
        st.json(assumptions)
    render_download_buttons({"Cost-aware plan": (pd.DataFrame([cost_plan]), f"phase29_{_safe_filename_part(cost_asset)}_{int(cost_horizon)}d_cost_plan.csv")})


elif page == "Portfolio Summary":
    render_premium_header(
        "Portfolio Summary",
        "A cross-asset interpretation of opportunity closeness, active blockers, and what deserves the next review.",
        "Portfolio research condition",
    )
    render_disclaimer_banner()
    render_blocked_capital_banner()
    portfolio_plans = st.session_state.get("phase26_asset_plans")
    if not isinstance(portfolio_plans, pd.DataFrame):
        portfolio_plans = _load_phase26_table("phase26_asset_plans")
    portfolio_summary = st.session_state.get("phase26_portfolio_plan")
    if not isinstance(portfolio_summary, pd.DataFrame):
        portfolio_summary = _load_phase26_table("phase26_portfolio_plan")
    if portfolio_plans.empty or portfolio_summary.empty:
        render_empty_state("No portfolio summary", "Generate a research plan first so the assistant can summarize the saved cross-asset evidence.")
    else:
        portfolio_plans = rank_asset_plans(portfolio_plans)
        portfolio_summary = generate_portfolio_plan(portfolio_plans)
        summary = portfolio_summary.iloc[0]
        final_portfolio_snapshot = _load_phase29_table("phase29_all_asset_prediction_snapshot.csv")
        if not final_portfolio_snapshot.empty:
            render_section_header("Current market snapshot", "Latest available prices with final cost and benchmark context.")
            render_market_snapshot_grid(
                final_portfolio_snapshot,
                on_view_plan=lambda asset, horizon: _navigate_to_plan(asset, "Asset Plans", horizon),
            )
            portfolio_gaps = pd.to_numeric(final_portfolio_snapshot.get("ActiveMinusPassiveNetPct"), errors="coerce")
            portfolio_cost_drag = pd.to_numeric(final_portfolio_snapshot.get("CostDragPct"), errors="coerce")
            render_metric_grid([
                {"title": "Cost-blocked ideas", "value": int(final_portfolio_snapshot["CostVerdict"].eq("CostsTooHighForSignal").sum()), "subtitle": "Estimated move does not clear costs", "status": "warning"},
                {"title": "Passive benchmark stronger", "value": int(portfolio_gaps.lt(0).sum()), "subtitle": "After entered cost assumptions", "status": "info"},
                {"title": "Average cost drag", "value": f"{float(portfolio_cost_drag.mean()):.2f}%" if portfolio_cost_drag.notna().any() else "Not available", "subtitle": COST_DISCLAIMER, "status": "warning"},
                {"title": "What to monitor next", "value": "Outcomes + costs", "subtitle": "Then compare with the same passive dates", "status": "neutral"},
            ])
        render_metric_grid(
            [
                {"title": "Overall research condition", "value": str(summary.get("OverallResearchCondition", "Evidence constrained")), "subtitle": "No real-money approval", "status": "warning"},
                {"title": "Closest to Track", "value": str(summary.get("ClosestToTrack", "None")), "subtitle": f"Opportunity score {float(summary.get('ClosestOpportunityScore', 0)):.0f}/100", "status": "info"},
                {"title": "Watch", "value": int(summary.get("WatchCount", 0)), "subtitle": "Interesting but not strong enough", "status": "info"},
                {"title": "High Risk", "value": int(summary.get("AvoidHighRiskCount", 0)), "subtitle": "Risk or weak evidence dominates", "status": "critical"},
                {"title": "Data Issues", "value": int(summary.get("DataIssueCount", 0)), "subtitle": "Repair before interpretation", "status": "warning"},
            ]
        )
        summary_a, summary_b = st.columns(2)
        with summary_a:
            render_risk_explanation_card(
                "Why the system is cautious",
                str(summary.get("WhySystemIsCautious", "Evidence remains conditional.")),
                f"Main risk theme: {summary.get('MainRiskTheme', summary.get('MainMarketRisk', 'Unknown'))}",
            )
        with summary_b:
            render_monitoring_card(
                "What to monitor next",
                str(summary.get("WhatUserShouldMonitorNext", summary.get("WhatToRecheckNext", "Review the next evidence update."))),
                str(summary.get("NextReviewTrigger", "After the next saved evidence refresh.")),
            )
        render_glass_container(
            "A useful cautious result",
            "If most assets are High Risk, the best action is not to force a decision. The useful output is knowing what to monitor and what conditions would need to improve.",
        )
        render_section_header("Closest research opportunities", "The ranking does not override any High Risk or Data Issue status.")
        for row_start in range(0, min(6, len(portfolio_plans)), 2):
            columns = st.columns(2)
            for column, (_, opportunity) in zip(columns, portfolio_plans.iloc[row_start:row_start + 2].iterrows()):
                with column:
                    render_opportunity_card(opportunity.to_dict())
        for status_group, labels in (
            ("Track candidates", ["Track"]),
            ("Watchlist", ["Watch"]),
            ("Wait", ["Wait", "Not Enough Evidence"]),
            ("Avoid / High Risk", ["Avoid", "High Risk"]),
            ("Data Issues", ["Data Issue"]),
        ):
            group = portfolio_plans[portfolio_plans["Status"].isin(labels)]
            with st.expander(status_group, expanded=status_group == "Track candidates"):
                if group.empty:
                    st.info(f"No plans currently fall into {status_group.lower()}.")
                else:
                    st.dataframe(group[["Asset", "Horizon", "Status", "Confidence", "Summary", "MainRisk", "RecheckWhen"]], width="stretch", hide_index=True)


elif page == "About / Methodology":
    render_premium_header(
        "About / Methodology",
        "How the research assistant turns technical evidence into simple, conservative plans.",
        "Transparent by design",
    )
    render_disclaimer_banner()
    render_section_header("What the assistant does")
    st.write(
        "The assistant combines saved forecast, signal, validation, benchmark, replay, risk, regime, portfolio, "
        "and freshness evidence. It then assigns a conservative status and explains what would change it."
    )
    render_section_header("Status meanings")
    st.dataframe(
        pd.DataFrame(
            [
                ("Track", "Worth monitoring in research mode."),
                ("Watch", "Interesting, but not strong enough yet."),
                ("Wait", "There is no clear reason to prioritize it now."),
                ("Avoid", "Current evidence or risk is poor."),
                ("High Risk", "Risk warnings dominate the evidence."),
                ("Data Issue", "The available evidence cannot be trusted yet."),
                ("Not Enough Evidence", "More mature or repeated evidence is required."),
            ],
            columns=["Status", "Meaning"],
        ),
        width="stretch",
        hide_index=True,
    )
    render_section_header("What remains advanced")
    st.write(
        "Model training, leakage audits, calibration, detailed replay, benchmark diagnostics, quality gates, "
        "and raw evidence tables remain available under Advanced Diagnostics."
    )


elif page == "Overview Command Center":
    st.markdown('<p class="main-header">Multi-Asset Market Research &amp; Risk Intelligence Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Unified evidence, model validation, signal research, and risk oversight</p>', unsafe_allow_html=True)
    render_research_disclaimer()
    render_blocked_capital_banner()

    phase21_summary = load_latest_artifact(UNIFIED_RISK_COMMAND_CENTER_PHASE_NAME, "phase21_unified_summary", required=False)
    phase21_candidates = load_latest_artifact(UNIFIED_RISK_COMMAND_CENTER_PHASE_NAME, "phase21_paper_tracking_candidates", required=False)
    phase21_risks = load_latest_artifact(UNIFIED_RISK_COMMAND_CENTER_PHASE_NAME, "phase21_risk_register", required=False)
    phase21_gates = load_latest_artifact(UNIFIED_RISK_COMMAND_CENTER_PHASE_NAME, "phase21_quality_gates", required=False)
    phase22_summary = load_latest_artifact(PREDICTION_EDGE_IMPROVEMENT_PHASE_NAME, "phase22_prediction_edge_summary", required=False)
    phase22_scorecard = load_latest_artifact(PREDICTION_EDGE_IMPROVEMENT_PHASE_NAME, "phase22_asset_horizon_model_scorecard", required=False)
    phase22_gates = load_latest_artifact(PREDICTION_EDGE_IMPROVEMENT_PHASE_NAME, "phase22_quality_gates", required=False)

    command_verdict = "InsufficientEvidence"
    broad_edge = "NoBroadEdgeProven"
    best_evidence = "No saved unified or benchmark summary"
    if isinstance(phase21_summary, pd.DataFrame) and not phase21_summary.empty:
        row = phase21_summary.iloc[0]
        command_verdict = str(row.get("CommandCenterVerdict", command_verdict))
        broad_edge = str(row.get("BroadEdgeStatus", broad_edge))
        best_evidence = str(row.get("BestEvidenceSource", "Unified command-center evidence"))
    elif isinstance(phase22_summary, pd.DataFrame) and not phase22_summary.empty:
        row = phase22_summary.iloc[0]
        command_verdict = str(row.get("FinalVerdict", command_verdict))
        broad_edge = str(row.get("BroadEdgeStatus", broad_edge))
        best_evidence = "Model benchmark expansion"

    candidate_table = phase21_candidates if isinstance(phase21_candidates, pd.DataFrame) else pd.DataFrame()
    if candidate_table.empty and isinstance(phase22_scorecard, pd.DataFrame) and not phase22_scorecard.empty:
        label_column = phase22_scorecard.get("ResearchLabel", pd.Series(index=phase22_scorecard.index, dtype=str)).astype(str)
        candidate_table = phase22_scorecard[label_column.eq("PaperTrack")].copy()

    gate_frames = []
    for source_name, gate_table in (("Unified Risk", phase21_gates), ("Model Edge", phase22_gates)):
        if isinstance(gate_table, pd.DataFrame) and not gate_table.empty:
            frame = gate_table.copy()
            frame.insert(0, "EvidenceSource", source_name)
            gate_frames.append(frame)
    latest_gates = pd.concat(gate_frames, ignore_index=True) if gate_frames else pd.DataFrame()
    passed_gates = int(latest_gates["Passed"].astype(bool).sum()) if not latest_gates.empty and "Passed" in latest_gates.columns else 0
    total_gates = len(latest_gates)
    risk_table = phase21_risks if isinstance(phase21_risks, pd.DataFrame) else pd.DataFrame()
    workflow_report = st.session_state.get("phase23_multiasset_workflow_report")
    overview_freshness = (
        workflow_report.data_freshness_table
        if workflow_report is not None
        else build_data_freshness_table(None)
    )
    current_feeds = int(overview_freshness["FreshnessStatus"].eq("Current").sum())
    stale_feeds = int(overview_freshness["IsStale"].astype(bool).sum())

    render_metric_grid(
        [
            {"title": "Unified verdict", "value": command_verdict, "subtitle": best_evidence, "status": "info"},
            {"title": "Broad edge", "value": broad_edge, "subtitle": "Baseline breadth remains the governing hurdle", "status": "warning" if broad_edge == "NoBroadEdgeProven" else "positive"},
            {"title": "Paper-track rows", "value": len(candidate_table), "subtitle": "Conservative research candidates only", "status": "positive" if len(candidate_table) else "neutral"},
            {"title": "Latest quality gates", "value": f"{passed_gates}/{total_gates}", "subtitle": "Saved unified and benchmark evidence", "status": "positive" if total_gates and passed_gates == total_gates else "warning"},
            {"title": "Data freshness", "value": f"{current_feeds}/{len(get_supported_assets())}", "subtitle": f"Current feeds; {stale_feeds} stale or missing", "status": "positive" if stale_feeds == 0 else "warning"},
            {"title": "Open risk rows", "value": len(risk_table), "subtitle": "Warnings remain visible in the risk register", "status": "warning" if len(risk_table) else "neutral"},
        ]
    )

    if phase21_summary is None and phase22_summary is None:
        st.warning("No saved unified or model-benchmark summary was found. Run those research pages to populate the command center.")

    render_section_header("Research Pipeline", "Evidence moves forward only after chronology, baseline, and risk checks.")
    pipeline_steps = ["Data", "Features", "Forecasts", "Signals", "Validation", "Risk", "Benchmarking", "Unified Verdict"]
    render_pipeline_stepper(pipeline_steps)

    overview_left, overview_right = st.columns([1.15, 0.85])
    with overview_left:
        render_safe_table(candidate_table.head(10), "Best Paper-Track Candidates", "No conservative candidate is available in the latest saved evidence.")
    with overview_right:
        render_safe_table(risk_table.head(10), "Main Risks", "No unified risk register is currently available.")

    render_safe_table(latest_gates, "Latest Quality Gates", "No saved unified or benchmark quality-gate tables were found.")

    render_section_header("Workspace Guide", "Use the grouped sidebar to move from research creation to evidence review.")
    guide = pd.DataFrame(
        [
            ("Forecasting & Prediction", "Models, direct horizons, and true historical replay", "Direct Horizon Scanner"),
            ("Signal Research", "Probability signals, policies, and candidate diagnostics", "Signal Engine"),
            ("Validation & Evidence", "Walk-forward tests, calibration, ledgers, and artifacts", "Walk-Forward Validation"),
            ("Risk Intelligence", "Portfolio constraints, warnings, sizing, and regimes", "Risk & Warning Intelligence"),
            ("Benchmarking & Replay", "Serious baselines, proxy replay, and true ML replay", "Strategy Benchmark Arena"),
            ("Reports & Exports", "Unified verdicts, plans, and downloadable evidence", "Unified Risk Command Center"),
        ],
        columns=["Workspace", "Purpose", "Suggested Starting Page"],
    )
    st.dataframe(guide, width="stretch", hide_index=True)

    render_safe_table(
        overview_freshness,
        "Data Freshness",
        "No market-data freshness evidence is available yet.",
    )
    if overview_freshness["FreshnessStatus"].eq("MissingData").any():
        st.caption("Open Guided Research Workflow and refresh from the project dataset to populate observed dates.")


# ================================================================
# PAGE: GUIDED RESEARCH WORKFLOW
# ================================================================

elif page == "Guided Research Workflow":
    st.markdown('<p class="main-header">Guided Research Workflow</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">A step-by-step path from data health to an evidence-bound verdict</p>', unsafe_allow_html=True)
    render_research_disclaimer()
    render_blocked_capital_banner()

    wf_col_a, wf_col_b, wf_col_c = st.columns(3)
    with wf_col_a:
        workflow_asset = st.selectbox(
            "Workflow asset",
            get_supported_assets(),
            index=get_supported_assets().index(selected_asset),
            key="phase23_workflow_asset",
        )
    with wf_col_b:
        workflow_horizon = st.selectbox(
            "Workflow horizon",
            get_available_horizons(),
            index=get_available_horizons().index(int(selected_horizon)),
            format_func=lambda value: f"{int(value)}D",
            key="phase23_workflow_horizon",
        )
    with wf_col_c:
        refresh_project_data = st.checkbox(
            "Refresh from project dataset",
            value=False,
            help="Uses the project data loader only when the workflow audit is run.",
            key="phase23_refresh_project_data",
        )
    validate_asset_horizon(workflow_asset, workflow_horizon)

    if st.button("Build workflow status and save audit", type="primary"):
        audit_market_data = None
        if refresh_project_data:
            try:
                audit_market_data = load_raw_data("2015-01-01", use_cache=True)
            except Exception as exc:
                st.warning(f"Market-data refresh was unavailable; missing freshness remains visible: {exc}")
        workflow_report = run_multiasset_workflow_audit(
            market_data=audit_market_data,
            app_source=Path(__file__).read_text(encoding="utf-8"),
            autosave=True,
        )
        st.session_state.phase23_multiasset_workflow_report = workflow_report
        st.session_state.artifact_store_last_save = workflow_report.saved_artifacts

    workflow_report = st.session_state.get("phase23_multiasset_workflow_report")
    if workflow_report is None:
        workflow_report = run_multiasset_workflow_audit(
            market_data=None,
            app_source=Path(__file__).read_text(encoding="utf-8"),
            autosave=False,
        )
        st.info("No saved workflow audit is loaded. The tables below show the safe missing-evidence state.")

    latest_evidence = list_latest_artifacts()
    latest_evidence_count = len(latest_evidence) if isinstance(latest_evidence, pd.DataFrame) else 0
    failed_workflow_gates = int((~workflow_report.quality_gates_table["Passed"].astype(bool)).sum())
    freshness_current = int(workflow_report.data_freshness_table["FreshnessStatus"].eq("Current").sum())
    render_metric_grid(
        [
            {"title": "Selected context", "value": f"{workflow_asset} {int(workflow_horizon)}D", "subtitle": get_asset_target(workflow_asset), "status": "info"},
            {"title": "Workflow steps", "value": len(workflow_report.workflow_steps_table), "subtitle": "Ordered research checks", "status": "positive"},
            {"title": "Current asset feeds", "value": f"{freshness_current}/{len(get_supported_assets())}", "subtitle": "Simple calendar freshness check", "status": "warning" if freshness_current < len(get_supported_assets()) else "positive"},
            {"title": "Workflow gate failures", "value": failed_workflow_gates, "subtitle": f"{latest_evidence_count} saved artifacts indexed", "status": "critical" if failed_workflow_gates else "positive"},
        ]
    )

    workflow_records = workflow_report.workflow_steps_table.to_dict("records")
    workflow_labels = [str(step["StepName"]) for step in workflow_records]
    render_section_header("Research Path", "The highlighted first step is the correct starting point for a fresh evidence cycle.")
    render_pipeline_stepper(workflow_labels, active_step=0)
    render_metric_grid(
        [
            {
                "title": f"Step {int(step['StepNumber'])}",
                "value": step["StepName"],
                "subtitle": f"Next: {step['NextRecommendedPage']}",
                "status": "info" if int(step["StepNumber"]) == 1 else "neutral",
            }
            for step in workflow_records
        ]
    )

    render_section_header("Recommended Run Order", "Complete each step before treating later outputs as meaningful evidence.")
    for step in workflow_records:
        with st.expander(f"{int(step['StepNumber'])}. {step['StepName']}", expanded=int(step["StepNumber"]) == 1):
            st.write(step["WhatItDoes"])
            st.markdown(f"**Run or check:** {step['RunOrCheck']}")
            st.markdown(f"**How to read it:** {step['OutputMeaning']}")
            st.warning(step["WeakEvidenceWarning"])
            st.caption(f"Next recommended page: {step['NextRecommendedPage']}")

    render_safe_table(
        workflow_report.data_freshness_table,
        "Data Freshness Panel",
        "No freshness rows are available.",
    )
    st.caption("Freshness uses a simple weekday/calendar tolerance and does not claim full exchange-holiday awareness.")

    phase23_tabs = st.tabs(
        ["Next Actions", "Page Audit", "Multi-Asset Coverage", "Quality Gates", "Glossary"]
    )
    phase23_tables = [
        workflow_report.next_actions_table,
        workflow_report.page_audit_table,
        workflow_report.multiasset_coverage_table,
        workflow_report.quality_gates_table,
        workflow_report.glossary_terms_table,
    ]
    for tab, table in zip(phase23_tabs, phase23_tables):
        with tab:
            st.dataframe(table, width="stretch", hide_index=True)

    render_glossary_expander(
        glossary_entries(["Asset", "Horizon", "Baseline", "Leakage", "Walk-forward validation", "PaperTrack", "RealCapitalBlocked"])
    )
    render_download_buttons(
        {
            "Page audit": (workflow_report.page_audit_table, "phase23_page_audit.csv"),
            "Multi-asset coverage": (workflow_report.multiasset_coverage_table, "phase23_multiasset_coverage.csv"),
            "Workflow steps": (workflow_report.workflow_steps_table, "phase23_workflow_steps.csv"),
            "Glossary": (workflow_report.glossary_terms_table, "phase23_glossary_terms.csv"),
            "Data freshness": (workflow_report.data_freshness_table, "phase23_data_freshness.csv"),
            "Quality gates": (workflow_report.quality_gates_table, "phase23_quality_gates.csv"),
            "Next actions": (workflow_report.next_actions_table, "phase23_next_actions.csv"),
        }
    )


# ════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ════════════════════════════════════════════════════════════════

elif page == "ℹ️ About Project":
    st.markdown('<p class="main-header">About the Research Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-asset forecasting, evidence validation, and risk intelligence</p>', unsafe_allow_html=True)
    render_research_disclaimer()
    render_blocked_capital_banner()

    render_section_header("Mission")
    st.write(
        "The platform studies Gold, Silver, Crude Oil, Bitcoin, S&P 500, and Gold ETF through "
        "time-ordered forecasting, signal research, realistic paper backtests, and explicit risk gates."
    )
    render_section_header("Research Architecture")
    st.code(
        "Data -> Features -> Forecasts -> Signals -> Validation -> Risk -> "
        "Benchmarking -> True ML Replay -> Unified Verdict",
        language="text",
    )
    render_section_header("Evidence Standard")
    st.write(
        "Train-only preprocessing, chronological validation, baseline comparisons, transaction-cost "
        "stress, rejection visibility, and forward-paper evidence govern every serious conclusion."
    )
    render_section_header("Core Technology")
    st.dataframe(
        pd.DataFrame(
            [
                ("Data", "yfinance, FRED, pandas"),
                ("Models", "scikit-learn and optional research libraries"),
                ("Validation", "walk-forward replay, calibration, baseline audits"),
                ("Interface", "Streamlit and Plotly"),
                ("Evidence", "versioned CSV artifact store"),
            ],
            columns=["Layer", "Tools"],
        ),
        width="stretch",
        hide_index=True,
    )


# ════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ════════════════════════════════════════════════════════════════

elif page == "📊 Dataset Explorer":
    st.markdown('<p class="main-header">📊 Dataset Explorer</p>', unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("Loading data..."):
        df_raw = load_raw_data("2015-01-01", use_cache=True)

    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📈 Summary Statistics", "⬇️ Download"])

    with tab1:
        n_rows = st.slider("Rows to display", 10, 200, 50)
        st.dataframe(df_raw.tail(n_rows), width="stretch")

    with tab2:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df_raw.describe().T, width="stretch")

        st.markdown("#### Missing Values")
        null_counts = df_raw.isnull().sum()
        null_df = null_counts[null_counts > 0].to_frame("Missing Count")
        if null_df.empty:
            st.success("No missing values in the dataset ✓")
        else:
            st.dataframe(null_df, width="stretch")

    with tab3:
        csv = df_raw.to_csv().encode("utf-8")
        st.download_button(
            "📥 Download Full Dataset (CSV)", data=csv,
            file_name="multi_asset_master_dataset.csv", mime="text/csv",
        )


# ════════════════════════════════════════════════════════════════
# PAGE: TECHNICAL INDICATORS
# ════════════════════════════════════════════════════════════════

elif page == "📈 Technical Indicators":
    st.markdown(f'<p class="main-header">📈 Technical Indicator Viewer — {selected_asset}</p>', unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("Computing indicators..."):
        df_raw = load_raw_data("2015-01-01", use_cache=True)
        df_ind = build_features(df_raw, target_col=target_col)

    import plotly.graph_objects as go

    n_days = st.slider("Days to display", 30, 500, 250)

    tab1, tab2, tab3 = st.tabs(["📊 Moving Averages", "📉 RSI", "📈 MACD"])

    with tab1:
        recent = df_ind.tail(n_days)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent.index, y=recent[target_col], name=f"{selected_asset} Price", line=dict(color="white")))
        for p, color in zip([20, 50, 200], ["#FF6B6B", "#4ECDC4", "#FFD93D"]):
            col = f"SMA_{p}"
            if col in recent.columns:
                fig.add_trace(go.Scatter(x=recent.index, y=recent[col], name=f"SMA-{p}", line=dict(color=color)))
        fig.update_layout(template="plotly_dark", title="Price with Moving Averages")
        st.plotly_chart(fig, width="stretch")

    with tab2:
        recent = df_ind.tail(n_days)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent.index, y=recent["RSI"], line=dict(color="#9B59B6")))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
        fig.update_layout(template="plotly_dark", title="RSI", yaxis_range=[0, 100])
        st.plotly_chart(fig, width="stretch")

    with tab3:
        recent = df_ind.tail(n_days)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=recent.index, y=recent["MACD_Hist"], name="Histogram", marker_color="gray", opacity=0.4))
        fig.add_trace(go.Scatter(x=recent.index, y=recent["MACD"], name="MACD", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=recent.index, y=recent["MACD_Signal"], name="Signal", line=dict(color="#D4AF37")))
        fig.update_layout(template="plotly_dark", title="MACD")
        st.plotly_chart(fig, width="stretch")


# ════════════════════════════════════════════════════════════════
# PAGE: TRAIN MODELS
# ════════════════════════════════════════════════════════════════

elif page == "🤖 Train Models":
    st.markdown('<p class="main-header">🤖 Train Models</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("Select which model families to train, then click **Start Training**.")
    st.info(f"Training target asset: **{selected_asset}** (`{target_col}`)")
    col1, col2 = st.columns(2)
    with col1:
        train_ml = st.checkbox("Train ML Models (Linear Regression, Trees, Boosting, SVR)", value=True)
    with col2:
        train_dl = st.checkbox("Train DL Models (LSTM, BiLSTM, GRU, CNN-LSTM, Transformer)", value=False)
        if train_dl:
            st.caption("⚠️ DL training is slower — consider reducing epochs below for a quick demo run.")
            dl_epochs = st.slider("DL Epochs (demo)", 3, 100, 10)

    if st.button("🚀 Start Training", type="primary"):
        progress = st.progress(0, text="Loading data...")

        df_raw = load_raw_data("2015-01-01", use_cache=True)
        progress.progress(20, text="Engineering features...")
        df_features = build_features(df_raw, target_col=target_col)
        progress.progress(40, text="Preprocessing...")
        pp, data = get_preprocessor_and_data(df_features, target_col=target_col)

        st.session_state.pp = pp
        st.session_state.data = data
        st.session_state.df_features = df_features
        st.session_state.trained_asset = selected_asset

        progress.progress(55, text="Training ML models...")
        trainer = ModelTrainer(use_optuna=False, target_scaler=data.target_scaler, preprocessor=pp)

        if train_ml:
            trainer.train_all_ml(data)

        progress.progress(85, text="Finalizing...")
        st.session_state.trainer = trainer
        st.session_state.trained = True
        progress.progress(100, text="Done!")

        st.success(f"✔ Training complete for {selected_asset} — {len(trainer.results)} model(s) trained.")
        board = trainer.get_leaderboard("test")
        st.dataframe(board, width="stretch")

    elif st.session_state.trained:
        st.info("Models already trained this session. Go to **Compare Models** to view results, or retrain above.")


# ════════════════════════════════════════════════════════════════
# PAGE: COMPARE MODELS
# ════════════════════════════════════════════════════════════════

elif page == "🏆 Compare Models":
    st.markdown('<p class="main-header">🏆 Model Comparison</p>', unsafe_allow_html=True)
    st.markdown("---")

    if not st.session_state.trained:
        st.warning("⚠️ No models trained yet. Go to **Train Models** first.")
    else:
        _stop_if_asset_mismatch(selected_asset)
        trainer = st.session_state.trainer
        viz = Visualizer()

        metric = st.selectbox("Sort/compare by metric", ["RMSE", "MAE", "MAPE", "R2", "DirectionalAccuracy"])
        board = trainer.get_leaderboard("test")

        st.dataframe(board, width="stretch")

        st.markdown("### Baseline Checks")
        data = st.session_state.data
        try:
            baseline_board = price_baseline_leaderboard(data)
            st.caption("Baselines use only known price anchors. Naive baseline means: tomorrow's price = today's price.")
            st.dataframe(baseline_board, width="stretch")

            summary = model_vs_naive_summary(board, baseline_board)
            if summary:
                improvement = summary["rmse_improvement_pct"]
                if improvement > 0:
                    st.success(
                        f"Best model **{summary['best_model']}** beats Naive RMSE by **{improvement:.2f}%** "
                        f"({summary['best_model_rmse']} vs {summary['naive_rmse']})."
                    )
                else:
                    st.error(
                        f"Best model **{summary['best_model']}** does NOT beat Naive RMSE "
                        f"({summary['best_model_rmse']} vs {summary['naive_rmse']})."
                    )
        except Exception as exc:
            st.warning(f"Could not calculate baseline checks: {exc}")

        fig = viz.plot_model_comparison_plotly(board, metric=metric)
        st.plotly_chart(fig, width="stretch")

        best_name, best_result = trainer.get_best_model("test")
        st.success(f"🏆 Best Model: **{best_name}** — RMSE = ${best_result.metrics_test['RMSE']:.2f}, R² = {best_result.metrics_test['R2']:.4f}")

        st.markdown(f"### Actual vs Predicted — {selected_asset} (Best Model)")
        data = st.session_state.data
        fig2 = viz.plot_actual_vs_predicted_plotly(
            data.prices_test, best_result.predictions_test, data.test_index,
            title=f"{selected_asset} / {best_name} — Actual vs Predicted",
        )
        st.plotly_chart(fig2, width="stretch")


# ════════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ════════════════════════════════════════════════════════════════

elif page == "🔮 Prediction":
    st.info(
        "Foundational diagnostic view - this selected-asset estimate is not the final evidence layer. "
        "Check Direct Horizon Scanner, Walk-Forward ML Replay, and Unified Risk Command Center next."
    )
    st.markdown(f'<p class="main-header">🔮 Next-Day Prediction — {selected_asset}</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "This system is for educational and research purposes only. "
        "Predictions are estimates and should not be considered financial advice."
    )

    if not st.session_state.trained:
        st.warning("⚠️ No models trained yet. Go to **Train Models** first.")
    else:
        _stop_if_asset_mismatch(selected_asset)
        trainer = st.session_state.trainer
        pp = st.session_state.pp
        data = st.session_state.data

        model_name = st.selectbox("Select a model", list(trainer.results.keys()))
        result = trainer.results[model_name]

        predictor = Predictor(model=result.model, preprocessor=pp, is_sequence_model=False)

        # Use the latest available feature row for the live next-day forecast.
        # After the next-day target fix, data.X_test[-1] is the last row with a
        # known next-day label, not necessarily the newest feature row.
        target_col = getattr(pp, "target_col", get_asset_target(selected_asset))
        latest_features = st.session_state.df_features[data.feature_cols].tail(1)
        X_latest = data.feature_scaler.transform(latest_features.values)
        last_price = float(st.session_state.df_features[target_col].dropna().iloc[-1])
        next_price = predictor.predict_next_day(X_latest, last_price)

        try:
            rmse = _safe_test_rmse(result, data)

            prediction_range = calculate_prediction_range(
                last_price=last_price,
                predicted_price=next_price,
                rmse=rmse,
                confidence_level=0.68,
                model_used=model_name,
            )

            trading_signal = generate_trading_signal(
                predicted_return_pct=prediction_range.predicted_return_pct,
                lower_return_pct=prediction_range.lower_return_pct,
                upper_return_pct=prediction_range.upper_return_pct,
            )

            st.markdown("### Prediction Range")

            col1, col2, col3 = st.columns(3)
            col1.metric(f"Last Known {selected_asset} Price", f"${prediction_range.last_price:,.2f}")
            col2.metric(
                f"Predicted Next-Day {selected_asset} Price",
                f"${prediction_range.predicted_price:,.2f}",
                f"{prediction_range.predicted_return_pct:+.2f}%",
            )
            col3.metric("Model Used", prediction_range.model_used)

            col4, col5, col6 = st.columns(3)
            col4.metric("Expected Lower Bound", f"${prediction_range.lower_bound:,.2f}")
            col5.metric("Expected Upper Bound", f"${prediction_range.upper_bound:,.2f}")
            col6.metric("Confidence Level", f"{prediction_range.confidence_level:.0f}%")

            st.caption(
                f"Range is based on held-out test RMSE = ${prediction_range.error_used:,.2f}. "
                "It is an uncertainty estimate, not a guaranteed interval."
            )

            st.markdown("### Trading Signal")

            sig1, sig2, sig3 = st.columns(3)
            sig1.metric("Signal", trading_signal.signal)
            sig2.metric("Signal Confidence", trading_signal.confidence_label)
            sig3.metric("Risk Label", trading_signal.risk_label)

            st.caption(trading_signal.explanation)

        except Exception as exc:
            st.error(f"Could not calculate prediction range/signal: {exc}")

            col1, col2, col3 = st.columns(3)
            col1.metric(f"Last Known {selected_asset} Price", f"${last_price:,.2f}")
            col2.metric(f"Predicted Next-Day {selected_asset} Price", f"${next_price:,.2f}", f"{((next_price / last_price) - 1) * 100:+.2f}%")
            col3.metric("Model Used", model_name)

        st.markdown("### Test-Set Predictions (CSV Export)")

        target_col = getattr(pp, "target_col", get_asset_target(selected_asset))

        export_df = pd.DataFrame({
            "Date": data.test_index,
            "Actual_Price": data.prices_test,
            "Predicted_Price": result.predictions_test,
        })

        # Store real test-set predictions for the Backtesting page.
        try:
            st.session_state.backtest_df = _build_backtest_frame(
                data=data,
                model_result=result,
                target_col=target_col,
            )
        except Exception as exc:
            st.session_state.backtest_df = None
            st.warning(f"Could not prepare backtesting data: {exc}")

        st.dataframe(export_df.tail(20), width="stretch")

        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Predictions (CSV)",
            data=csv,
            file_name=f"{model_name}_predictions.csv",
            mime="text/csv",
        )


# ════════════════════════════════════════════════════════════════
# PAGE: BACKTESTING
# ════════════════════════════════════════════════════════════════

elif page == "📉 Backtesting":
    st.info(
        "Legacy diagnostic view - this basic selected-asset backtest is retained for comparison. "
        "Use Signal Engine and Walk-Forward Validation for cost-aware chronological evidence."
    )
    st.markdown('<p class="main-header">📉 Strategy Backtesting</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "This backtest is for educational and research purposes only. "
        "It does not include slippage, liquidity limits, taxes, or real execution constraints."
    )

    if not st.session_state.trained:
        st.warning("⚠️ No models trained yet. Go to **Train Models** first.")
    else:
        _stop_if_asset_mismatch(selected_asset)
        trainer = st.session_state.trainer
        pp = st.session_state.pp
        data = st.session_state.data

        model_name = st.selectbox("Select model for backtest", list(trainer.results.keys()))
        result = trainer.results[model_name]

        threshold_pct = st.slider(
            "Signal threshold (%)",
            min_value=0.0,
            max_value=3.0,
            value=0.2,
            step=0.1,
            help="Long only when predicted return is greater than this threshold.",
        )

        transaction_cost_pct = st.number_input(
            "Transaction cost per trade (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.1,
            step=0.05,
        )

        allow_short = st.checkbox("Allow shorting", value=False)

        target_col = getattr(pp, "target_col", get_asset_target(selected_asset))

        try:
            backtest_df = _build_backtest_frame(
                data=data,
                model_result=result,
                target_col=target_col,
            )

            bt = run_backtest_from_predictions(
                backtest_df,
                price_col=target_col,
                predicted_price_col="Predicted_Price",
                threshold=threshold_pct / 100.0,
                transaction_cost=transaction_cost_pct / 100.0,
                allow_short=allow_short,
            )

            st.session_state.backtest_df = backtest_df

        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.stop()

        metrics = bt.metrics
        equity = bt.equity_curve
        trades = bt.trades

        st.markdown("### Backtest Metrics")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
        c2.metric("Annualized Return", f"{metrics['annualized_return_pct']:.2f}%")
        c3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        c4.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Win Rate", f"{metrics['win_rate_pct']:.2f}%")
        c6.metric("Trades", int(metrics["number_of_trades"]))
        c7.metric("Passive Hold Return", f"{metrics['buy_hold_return_pct']:.2f}%")
        c8.metric("Strategy - Passive Hold", f"{metrics['strategy_minus_buy_hold_pct']:.2f}%")

        st.markdown("### Strategy vs Passive Hold")
        st.line_chart(equity[["strategy_equity", "buy_hold_equity"]])

        st.markdown("### Drawdown")
        st.line_chart(equity[["strategy_drawdown", "buy_hold_drawdown"]])

        st.markdown("### Position")
        st.line_chart(equity[["position"]])

        with st.expander("View backtest input data"):
            st.dataframe(backtest_df.tail(30), width="stretch")

        st.markdown("### Trade Summary")
        if trades.empty:
            st.info("No trades were generated with this threshold.")
        else:
            st.dataframe(trades, width="stretch")



# ════════════════════════════════════════════════════════════════
# PAGE: DIRECTIONAL MODELS
# ════════════════════════════════════════════════════════════════

elif page == "🎯 Directional Models":
    st.markdown(f'<p class="main-header">🎯 Directional Models — {selected_asset}</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.info(
        "This page trains separate Up/Down classifiers. Regression models optimize RMSE; "
        "directional classifiers directly optimize tomorrow's direction."
    )

    if not st.session_state.trained:
        st.warning("⚠️ No preprocessing/training session found. Go to **Train Models** first.")
    else:
        _stop_if_asset_mismatch(selected_asset)
        pp = st.session_state.pp
        data = st.session_state.data

        st.markdown("### Directional Baselines")
        try:
            dir_base = directional_baseline_leaderboard(data, pp)
            st.dataframe(dir_base, width="stretch")
        except Exception as exc:
            st.warning(f"Could not calculate directional baselines: {exc}")

        c_train, c_note = st.columns([1, 2])
        with c_train:
            train_dir = st.button("🚀 Train Directional Models", type="primary")
        with c_note:
            st.caption("Start here after training the normal ML models. This may take a little time if XGBoost/LightGBM/CatBoost classifiers are available.")

        if train_dir or st.session_state.directional_results is None or st.session_state.directional_asset != selected_asset:
            with st.spinner("Training Up/Down directional classifiers..."):
                dir_results = train_directional_models(data, pp, include_heavy=True)
                st.session_state.directional_results = dir_results
                st.session_state.directional_asset = selected_asset
            st.success(f"Directional models trained for {selected_asset}.")

        dir_results = st.session_state.directional_results

        if dir_results:
            st.markdown("### Directional Model Leaderboard")
            dir_board = directional_leaderboard(dir_results)
            st.dataframe(dir_board, width="stretch")

            valid_names = [name for name, res in dir_results.items() if len(res.probabilities_test) == len(data.prices_test)]
            if not valid_names:
                st.error("No valid directional model produced probability outputs.")
                st.stop()

            default_idx = 0
            if not dir_board.empty and "Model" in dir_board.columns:
                best_candidate = str(dir_board.iloc[0]["Model"])
                if best_candidate in valid_names:
                    default_idx = valid_names.index(best_candidate)

            st.markdown("### Probability-Based Backtest")
            model_name = st.selectbox("Select directional model", valid_names, index=default_idx)
            probability_threshold_pct = st.slider(
                "Probability threshold (%)",
                min_value=50,
                max_value=75,
                value=55,
                step=1,
                help="Long when P(up) is above this threshold. With shorting ON, short when P(up) is below 100-threshold.",
            )
            transaction_cost_pct = st.number_input(
                "Transaction cost per trade (%)",
                min_value=0.0,
                max_value=5.0,
                value=0.10,
                step=0.05,
            )
            allow_short = st.checkbox("Allow shorting for directional model", value=False)

            selected_result = dir_results[model_name]
            try:
                metrics, equity = run_directional_probability_backtest(
                    data,
                    selected_result.probabilities_test,
                    probability_threshold=probability_threshold_pct / 100.0,
                    transaction_cost=transaction_cost_pct / 100.0,
                    allow_short=allow_short,
                )
            except Exception as exc:
                st.error(f"Directional backtest failed: {exc}")
                st.stop()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
            c2.metric("Annualized Return", f"{metrics['annualized_return_pct']:.2f}%")
            c3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            c4.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Win Rate", f"{metrics['win_rate_pct']:.2f}%")
            c6.metric("Trades", int(metrics["number_of_trades"]))
            c7.metric("Passive Hold Return", f"{metrics['buy_hold_return_pct']:.2f}%")
            c8.metric("Strategy - Passive Hold", f"{metrics['strategy_minus_buy_hold_pct']:.2f}%")

            st.markdown("### Directional Strategy vs Passive Hold")
            st.line_chart(equity[["strategy_equity", "buy_hold_equity"]])

            st.markdown("### Directional Position")
            st.line_chart(equity[["position"]])

            with st.expander("View directional backtest input data"):
                st.dataframe(equity.tail(50), width="stretch")



# ════════════════════════════════════════════════════════════════
# PAGE: FEATURE INTELLIGENCE
# ════════════════════════════════════════════════════════════════

elif page == "🧠 Feature Intelligence":
    st.markdown(f'<p class="main-header">🧠 Phase 5 Feature Intelligence — {selected_asset}</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 5 does not try to hide weak validation. It adds better market features first: "
        "volatility regime, trend strength, DXY/VIX/TNX pressure, rolling correlations, relative strength, "
        "drawdowns, and breakout/breakdown pressure. Then Phase 4A tells us honestly whether they help."
    )

    st.markdown("### Build Feature Audit")
    st.caption(
        "This page checks whether the Phase 5 FI_* features are being created for the selected target asset. "
        "It is a feature-quality page, not a trading-signal page."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        start_for_features = st.text_input("Start date", value="2015-01-01", key="phase5_start_date")
    with c2:
        force_refresh_features = st.checkbox("Force fresh market download", value=False, key="phase5_force_refresh")

    if st.button("🔬 Run Phase 5 Feature Audit", type="primary"):
        with st.spinner("Building Phase 5 features..."):
            try:
                raw = load_raw_data(start_for_features, use_cache=not force_refresh_features)
                features = build_features(raw, target_col=target_col)
                audit = build_feature_intelligence_report(features, target_col=target_col)
                st.session_state.phase5_audit = audit
                st.session_state.phase5_features_preview = features[phase5_feature_columns(features)].tail(30)
            except Exception as exc:
                st.error(f"Phase 5 feature audit failed: {exc}")
                st.stop()

    audit = st.session_state.get("phase5_audit")
    preview = st.session_state.get("phase5_features_preview")

    if audit is None:
        st.info("Run the audit to confirm Phase 5 feature generation.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Target", audit.target_col)
        m2.metric("Rows", audit.total_rows)
        m3.metric("Total Columns", audit.total_columns)
        m4.metric("Phase 5 Features", audit.phase5_columns)

        if audit.phase5_columns < 40:
            st.warning("Few Phase 5 features were created. Check available asset/macro columns in the dataset.")
        else:
            st.success("Phase 5 feature intelligence is active.")

        tab1, tab2, tab3 = st.tabs(["Feature Families", "Missingness", "Preview"])
        with tab1:
            st.markdown("### Feature Family Counts")
            st.dataframe(audit.family_counts, width="stretch")
            st.markdown("### Sample Phase 5 Columns")
            st.write(audit.sample_columns)
        with tab2:
            st.markdown("### Missingness After Cleaning")
            st.caption("This should usually be near zero because the feature module forward-fills past values and drops only early warm-up rows.")
            st.dataframe(audit.missing_summary, width="stretch")
        with tab3:
            st.markdown("### Latest FI_* Feature Values")
            if preview is not None and not preview.empty:
                st.dataframe(preview, width="stretch")
            else:
                st.info("No preview data available.")

    st.markdown("---")
    st.info(
        "After this audit passes, go to **🌐 Multi-Asset Matrix** and run all assets again with "
        "**Use Phase 5 feature intelligence** enabled. The goal is not instant high trust; the goal is to see whether "
        "RMSE-vs-naive, direction accuracy, and trust score improve honestly."
    )


# ════════════════════════════════════════════════════════════════
# PAGE: RESEARCH VALIDATION
# ════════════════════════════════════════════════════════════════

elif page == "🧪 Research Validation":
    st.info(
        "Foundational diagnostic view - use Walk-Forward ML Replay and Unified Risk Command Center "
        "for the newer multi-asset historical evidence chain."
    )
    st.markdown(f'<p class="main-header">🧪 Research-Grade Validation — {selected_asset}</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "This page is the hard-truth layer. It checks baselines, leakage risk, regime weakness, "
        "walk-forward stability, and a conservative model trust score. High R² alone is not treated as proof."
    )

    if not st.session_state.trained:
        st.warning("⚠️ Train ML models first. Go to **Train Models**.")
    else:
        _stop_if_asset_mismatch(selected_asset)
        trainer = st.session_state.trainer
        pp = st.session_state.pp
        data = st.session_state.data
        df_features = st.session_state.df_features

        tab1, tab2, tab3, tab4 = st.tabs([
            "Trust Score",
            "Regime Performance",
            "Walk-Forward",
            "Leakage Audit",
        ])

        with tab1:
            st.markdown("### Conservative Model Trust Score")
            st.caption(
                "Score uses RMSE improvement over Naive, directional accuracy, long-only backtest risk, "
                "drawdown, and overfit gap. It intentionally gives little credit to high R² alone."
            )
            try:
                report = build_validation_report(trainer, data, df_features)
                st.dataframe(report.trust_scores, width="stretch")
                if not report.trust_scores.empty:
                    best = report.trust_scores.iloc[0]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Best Trust Score", f"{float(best['TrustScore']):.2f}/100")
                    c2.metric("Best Candidate", str(best["Model"]))
                    c3.metric("Verdict", str(best["Verdict"]))
                    if float(best["TrustScore"]) < 55:
                        st.error("No model is currently strong enough for confident trading signals. This is useful truth, not failure.")
                    elif float(best["TrustScore"]) < 75:
                        st.warning("Best model is a medium-trust research candidate. Use risk controls and further validation.")
                    else:
                        st.success("A high-trust candidate exists, but still requires live paper-trading validation.")

                st.markdown("### Price Baselines")
                st.dataframe(report.baseline_board, width="stretch")
            except Exception as exc:
                st.error(f"Validation report failed: {exc}")

        with tab2:
            st.markdown("### Regime-wise Model Weakness")
            st.caption("A serious model should not only look good overall; it should reveal where it fails: bull, bear, sideways, high-volatility, low-volatility.")
            model_name = st.selectbox("Select model for regime analysis", list(trainer.results.keys()), key="regime_model")
            result = trainer.results[model_name]
            try:
                regime_board = regime_performance(data, result.predictions_test)
                st.dataframe(regime_board, width="stretch")
                if not regime_board.empty:
                    worst = regime_board.sort_values("DirectionalAccuracy", ascending=True).iloc[0]
                    st.warning(
                        f"Weakest regime for **{model_name}**: {worst['RegimeType']} = **{worst['Regime']}** "
                        f"with directional accuracy **{worst['DirectionalAccuracy']:.2f}%**."
                    )
            except Exception as exc:
                st.error(f"Regime analysis failed: {exc}")

        with tab3:
            st.markdown("### Walk-Forward Validation")
            st.caption(
                "This retrains the selected model across rolling time splits. It is slower but much closer to real research than a single train/test split."
            )
            wf_model_name = st.selectbox("Select model for walk-forward", list(trainer.results.keys()), key="wf_model")
            n_splits = st.slider("Walk-forward folds", min_value=3, max_value=8, value=5, step=1)
            max_train = st.selectbox("Training window", ["Expanding", "Last 1000 rows", "Last 1500 rows"], index=0)
            max_train_size = None if max_train == "Expanding" else int(max_train.split()[1])

            if st.button("🚀 Run Walk-Forward Validation", type="primary"):
                with st.spinner("Running walk-forward validation. This can take time for heavier models..."):
                    try:
                        base_result = trainer.results[wf_model_name]
                        folds, summary = walk_forward_validate_model(
                            base_result.model,
                            wf_model_name,
                            data,
                            pp,
                            n_splits=n_splits,
                            max_train_size=max_train_size,
                        )
                        st.session_state.phase4_wf_folds = folds
                        st.session_state.phase4_wf_summary = summary
                    except Exception as exc:
                        st.error(f"Walk-forward validation failed: {exc}")
                        st.stop()

            if "phase4_wf_summary" in st.session_state:
                summary = st.session_state.phase4_wf_summary
                folds = st.session_state.phase4_wf_folds
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean RMSE", f"{summary['MeanRMSE']:.2f}")
                c2.metric("RMSE Std", f"{summary['StdRMSE']:.2f}")
                c3.metric("Mean Direction", f"{summary['MeanDirectionalAccuracy']:.2f}%")
                c4.metric("Stability Score", f"{summary['StabilityScore']:.2f}/100")
                st.dataframe(folds, width="stretch")

        with tab4:
            st.markdown("### Leakage / Alignment Audit")
            st.caption("This checks target alignment, scaler leakage, target exclusion, and extreme feature correlation with next-day target.")
            try:
                report = build_validation_report(trainer, data, df_features)
                st.dataframe(report.leakage_report, width="stretch")
                bad = report.leakage_report[report.leakage_report["Status"].isin(["FAIL", "ERROR"])]
                review = report.leakage_report[report.leakage_report["Status"].eq("REVIEW")]
                if not bad.empty:
                    st.error("Critical validation issue found. Do not trust model results until fixed.")
                elif not review.empty:
                    st.warning("Some checks need review. This does not always mean leakage, but it must be inspected.")
                else:
                    st.success("Core leakage/alignment checks passed.")
            except Exception as exc:
                st.error(f"Leakage audit failed: {exc}")



# ════════════════════════════════════════════════════════════════
# PAGE: MULTI-ASSET VALIDATION MATRIX
# ════════════════════════════════════════════════════════════════

elif page == "🌐 Multi-Asset Matrix":
    st.markdown('<p class="main-header">🌐 Multi-Asset Research Validation Matrix</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "This is the serious all-asset layer. It validates every configured asset, not only Gold. "
        "The goal is to identify which assets are actually modelable, which are weak, and which must not be trusted for signals."
    )

    st.markdown("### Validation Scope")
    all_assets = get_asset_names()
    selected_assets_for_matrix = st.multiselect(
        "Assets to validate",
        all_assets,
        default=all_assets,
        help="For a serious final run, keep all assets selected. For quick debugging, choose 1-2 assets.",
    )

    model_depth = st.selectbox(
        "Model depth",
        [
            "fast",
            "core",
            "full",
        ],
        index=0,
        help=(
            "fast = Linear Regression + Decision Tree; "
            "core = Linear/RF/XGBoost/LightGBM/CatBoost; "
            "full = all ML models including SVR. Use core/full for serious validation."
        ),
    )

    use_phase5_matrix = st.checkbox(
        "Use Phase 5 feature intelligence",
        value=True,
        help="Adds FI_* features: regimes, cross-asset relationships, macro pressure, volatility, and trend intelligence.",
    )

    include_wf_matrix = st.checkbox(
        "Include walk-forward check for best model per asset (slower)",
        value=False,
    )
    wf_splits_matrix = st.slider("Walk-forward folds", 3, 6, 3, disabled=not include_wf_matrix)

    st.caption(
        "Recommendation: use **fast** first to confirm the pipeline, then use **core** for a serious all-asset run. "
        "Full mode may take longer."
    )

    run_matrix = st.button("🚀 Run Multi-Asset Validation", type="primary")

    if run_matrix:
        if not selected_assets_for_matrix:
            st.error("Select at least one asset.")
            st.stop()

        progress = st.progress(0, text="Loading raw market data...")

        def _matrix_progress(i, total, msg):
            pct = int((i - 1) / max(total, 1) * 100)
            progress.progress(pct, text=msg)

        with st.spinner("Running multi-asset research validation. This may take time..."):
            try:
                df_raw = load_raw_data("2015-01-01", use_cache=True)
                report = run_multiasset_validation(
                    raw_df=df_raw,
                    asset_names=selected_assets_for_matrix,
                    model_set=model_depth,
                    include_walk_forward=include_wf_matrix,
                    walk_forward_splits=wf_splits_matrix,
                    use_phase5_features=use_phase5_matrix,
                    progress_callback=_matrix_progress,
                )
                st.session_state.multiasset_validation_report = report
                st.session_state.multiasset_validation_settings = {
                    "assets": selected_assets_for_matrix,
                    "model_depth": model_depth,
                    "include_walk_forward": include_wf_matrix,
                    "walk_forward_splits": wf_splits_matrix,
                    "use_phase5_features": use_phase5_matrix,
                }
                progress.progress(100, text="Multi-asset validation complete.")
            except Exception as exc:
                st.error(f"Multi-asset validation failed: {exc}")
                st.stop()

    report = st.session_state.multiasset_validation_report

    if report is None:
        st.info("Run the validation to generate the all-asset trust matrix.")
    else:
        settings = st.session_state.get("multiasset_validation_settings") or {}
        st.caption(f"Last run settings: {settings}")

        summary = report.asset_summary
        status = summarize_asset_status(summary)

        st.markdown("### Asset-Level Verdict")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("High Trust", status.get("High", 0))
        c2.metric("Medium Trust", status.get("Medium", 0))
        c3.metric("Low Trust", status.get("Low", 0))
        c4.metric("Do Not Trust", status.get("DoNotTrust", 0))

        if summary is not None and not summary.empty:
            st.dataframe(summary, width="stretch")

            do_not = summary[summary["AssetVerdict"].astype(str).str.contains("Do not trust", case=False, na=False)]
            if not do_not.empty:
                st.error(
                    "Some assets are currently marked **Do Not Trust for Signals**. "
                    "That is not failure; it tells us where the model needs better features or a different objective."
                )
            else:
                st.success("No selected asset was marked hard Do-Not-Trust in this run.")
        else:
            st.warning("No asset summary was produced.")

        tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs([
            "Model Leaderboard",
            "Price Baselines",
            "Leakage Matrix",
            "Walk-Forward",
            "Errors",
        ])

        with tab_a:
            st.markdown("### All Models Across All Assets")
            st.caption("Sorts all asset/model pairs by conservative trust score. High R² alone does not dominate this score.")
            if report.model_leaderboard is not None and not report.model_leaderboard.empty:
                st.dataframe(report.model_leaderboard, width="stretch")
            else:
                st.info("No model leaderboard available.")

        with tab_b:
            st.markdown("### Baselines Across All Assets")
            st.caption("This shows whether an ML model beats simple logic such as tomorrow=today or moving averages.")
            if report.baseline_leaderboard is not None and not report.baseline_leaderboard.empty:
                st.dataframe(report.baseline_leaderboard, width="stretch")
            else:
                st.info("No baseline table available.")

        with tab_c:
            st.markdown("### Leakage / Alignment Checks Across Assets")
            if report.leakage_matrix is not None and not report.leakage_matrix.empty:
                st.dataframe(report.leakage_matrix, width="stretch")
                failures = report.leakage_matrix[report.leakage_matrix["Status"].isin(["FAIL", "ERROR"])]
                reviews = report.leakage_matrix[report.leakage_matrix["Status"].eq("REVIEW")]
                if not failures.empty:
                    st.error("Critical leakage/alignment failures exist. Do not trust affected assets.")
                elif not reviews.empty:
                    st.warning("Some checks need manual review. This is normal in serious research.")
                else:
                    st.success("Core leakage/alignment checks passed for selected assets.")
            else:
                st.info("No leakage matrix available.")

        with tab_d:
            st.markdown("### Walk-Forward Summary")
            if report.walk_forward_summary is not None and not report.walk_forward_summary.empty:
                st.dataframe(report.walk_forward_summary, width="stretch")
            else:
                st.info("Walk-forward was not run, or no walk-forward summary was produced.")

        with tab_e:
            st.markdown("### Assets / Models That Failed")
            if report.errors is not None and not report.errors.empty:
                st.dataframe(report.errors, width="stretch")
            else:
                st.success("No asset-level errors in the last multi-asset validation run.")


# ════════════════════════════════════════════════════════════════
# PAGE: DIRECT FORECAST MODELS
# ════════════════════════════════════════════════════════════════

elif page == "🎯 Direct Forecast Models":
    st.markdown('<p class="main-header">🎯 Direct Forecast Models</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 6 trains direct horizon models for future returns and direction. "
        "It does not recursively roll a 30-day price forecast forward, and it must beat naive baselines before it is useful."
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        direct_asset = st.selectbox(
            "Asset",
            get_asset_names(),
            index=get_asset_names().index(selected_asset) if selected_asset in get_asset_names() else 0,
            key="direct_forecast_asset",
        )
    with col_b:
        direct_horizon_label = st.selectbox(
            "Horizon",
            [f"{h}D" for h in DIRECT_FORECAST_HORIZONS],
            index=1,
            key="direct_forecast_horizon",
        )
        direct_horizon = int(str(direct_horizon_label).replace("D", ""))
    with col_c:
        direct_depth = st.selectbox(
            "Model depth",
            ["fast", "core"],
            index=0,
            help="fast = linear/logistic + decision tree. core adds random forest and gradient boosting.",
            key="direct_forecast_depth",
        )
    with col_d:
        direct_use_phase5 = st.checkbox(
            "Use Phase 5 feature intelligence",
            value=True,
            help="Adds FI_* market intelligence features before direct horizon targets are created.",
            key="direct_forecast_phase5",
        )

    run_direct = st.button("🚀 Run Direct Forecast Models", type="primary")

    if run_direct:
        with st.spinner("Training direct horizon return and direction models..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                direct_report = run_direct_forecast_report(
                    raw_df=raw_df,
                    asset_name=direct_asset,
                    horizon=direct_horizon,
                    model_depth=direct_depth,
                    use_phase5_features=direct_use_phase5,
                )
                st.session_state.direct_forecast_report = direct_report
                st.session_state.direct_forecast_settings = {
                    "asset": direct_asset,
                    "horizon": direct_horizon,
                    "model_depth": direct_depth,
                    "use_phase5_features": direct_use_phase5,
                }
            except Exception as exc:
                st.error(f"Direct forecast model run failed: {exc}")
                st.stop()

    direct_report = st.session_state.direct_forecast_report

    if direct_report is None:
        st.info("Run direct forecast models to generate a horizon-specific leaderboard.")
    else:
        settings = st.session_state.get("direct_forecast_settings") or {}
        st.caption(f"Last run settings: {settings}")

        leaderboard = direct_report.leaderboard
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{direct_report.rows:,}")
        c2.metric("Features", f"{direct_report.feature_count:,}")
        c3.metric("Test Rows", f"{direct_report.test_rows:,}")
        c4.metric("Horizon", f"{direct_report.horizon}D")

        if leaderboard is None or leaderboard.empty:
            st.warning("No direct model leaderboard was produced.")
        else:
            best = leaderboard.iloc[0]
            st.markdown("### Direct Horizon Leaderboard")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Best Trust Score", f"{float(best['TrustScore']):.2f}/100")
            b2.metric("Best RMSE vs Naive", f"{float(best['RMSE_vs_Naive_%']):+.2f}%")
            b3.metric("Best Direction vs Baseline", f"{float(best['Direction_vs_Baseline_%']):+.2f}%")
            b4.metric("Best Verdict", str(best["Verdict"]))

            st.dataframe(leaderboard, width="stretch")

            failed_return = leaderboard[leaderboard["RMSE_vs_Naive_%"].astype(float) < 0]
            failed_direction = leaderboard[leaderboard["Direction_vs_Baseline_%"].astype(float) <= 0]
            if len(failed_return) == len(leaderboard):
                st.error("All direct models failed the zero-return RMSE baseline. Do not use this horizon for signals.")
            elif not failed_return.empty:
                st.warning("Some direct models failed the zero-return RMSE baseline.")

            if len(failed_direction) == len(leaderboard):
                st.error("All direct models failed the direction baseline. Directional signal is not trustworthy.")
            elif not failed_direction.empty:
                st.warning("Some direct models failed the direction baseline.")

            do_not = leaderboard[leaderboard["Verdict"].astype(str).str.contains("Do not trust", case=False, na=False)]
            if not do_not.empty:
                st.error("One or more direct horizon models are marked Do Not Trust for Signals.")

            csv = leaderboard.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Direct Leaderboard CSV",
                data=csv,
                file_name=f"direct_forecast_{direct_report.asset}_{direct_report.horizon}D.csv".replace(" ", "_"),
                mime="text/csv",
            )

        tab_base, tab_errors, tab_alignment = st.tabs(["Baselines", "Errors", "Alignment"])

        with tab_base:
            st.markdown("### Naive Baselines")
            if direct_report.baseline_board is not None and not direct_report.baseline_board.empty:
                st.dataframe(direct_report.baseline_board, width="stretch")
            else:
                st.warning("No baseline board was produced.")

        with tab_errors:
            st.markdown("### Model Failures")
            if direct_report.errors is not None and not direct_report.errors.empty:
                st.dataframe(direct_report.errors, width="stretch")
            else:
                st.success("No model-level errors in the last direct forecast run.")

        with tab_alignment:
            dataset = direct_report.dataset
            if dataset is not None:
                st.markdown("### Direct Target Alignment")
                st.write(
                    {
                        "return_target": dataset.return_target_col,
                        "direction_target": dataset.direction_target_col,
                        "volatility_target": dataset.volatility_target_col,
                        "dropped_tail_rows": dataset.dropped_tail_rows,
                        "train_start": str(dataset.train_index.min().date()) if len(dataset.train_index) else "",
                        "train_end": str(dataset.train_index.max().date()) if len(dataset.train_index) else "",
                        "test_start": str(dataset.test_index.min().date()) if len(dataset.test_index) else "",
                        "test_end": str(dataset.test_index.max().date()) if len(dataset.test_index) else "",
                    }
                )
                leaked = [c for c in dataset.feature_cols if c.startswith(("future_return_", "future_direction_", "future_realized_vol_"))]
                if leaked:
                    st.error(f"Future target columns leaked into features: {leaked}")
                else:
                    st.success("No future target columns are present in the model feature set.")
            else:
                st.info("No dataset metadata available.")


# ════════════════════════════════════════════════════════════════
# PAGE: DIRECT HORIZON SCANNER
# ════════════════════════════════════════════════════════════════

elif page == "🧭 Direct Horizon Scanner":
    st.markdown('<p class="main-header">🧭 Direct Horizon Scanner</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 6B scans direct forecast models across assets and horizons. "
        "Do Not Trust results are shown plainly; this page is for finding honest pockets of signal, not decorating weak models."
    )

    all_scan_assets = get_asset_names()
    scan_col_a, scan_col_b, scan_col_c = st.columns([2, 2, 1])
    with scan_col_a:
        scan_assets = st.multiselect(
            "Assets",
            all_scan_assets,
            default=all_scan_assets,
            help="Serious validation should include every configured asset.",
            key="direct_horizon_scan_assets",
        )
    with scan_col_b:
        scan_horizon_labels = st.multiselect(
            "Horizons",
            [f"{h}D" for h in DIRECT_FORECAST_HORIZONS],
            default=[f"{h}D" for h in DIRECT_FORECAST_HORIZONS],
            key="direct_horizon_scan_horizons",
        )
        scan_horizons = [int(str(label).replace("D", "")) for label in scan_horizon_labels]
    with scan_col_c:
        scan_depth = st.selectbox(
            "Model depth",
            ["core", "fast"],
            index=0,
            help="core is the default for scanner runs; fast is useful for smoke tests.",
            key="direct_horizon_scan_depth",
        )
        scan_use_phase5 = st.checkbox(
            "Use Phase 5",
            value=True,
            help="Adds FI_* market intelligence features before direct horizon targets are created.",
            key="direct_horizon_scan_phase5",
        )

    run_scan = st.button("🚀 Run Asset × Horizon Scan", type="primary")

    if run_scan:
        if not scan_assets:
            st.error("Select at least one asset.")
            st.stop()
        if not scan_horizons:
            st.error("Select at least one horizon.")
            st.stop()

        progress = st.progress(0, text="Preparing direct horizon scan...")

        def _scan_progress(done, total, msg):
            pct = int(done / max(total, 1) * 100)
            progress.progress(pct, text=msg)

        with st.spinner("Running asset × horizon direct forecast scanner..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                scan_report = run_asset_horizon_scan(
                    raw_df=raw_df,
                    asset_names=scan_assets,
                    horizons=scan_horizons,
                    model_depth=scan_depth,
                    use_phase5_features=scan_use_phase5,
                    progress_callback=_scan_progress,
                )
                st.session_state.direct_horizon_scan_report = scan_report
                st.session_state.direct_horizon_scan_settings = scan_report.settings
                progress.progress(100, text="Direct horizon scan complete.")
            except Exception as exc:
                st.error(f"Direct horizon scan failed: {exc}")
                st.stop()

    scan_report = st.session_state.direct_horizon_scan_report
    if scan_report is None:
        st.info("Run the scanner to evaluate every selected asset × horizon combination.")
    else:
        settings = st.session_state.get("direct_horizon_scan_settings") or {}
        st.caption(f"Last run settings: {settings}")

        counts = scan_report.status_counts or {}
        count_a, count_b, count_c, count_d = st.columns(4)
        count_a.metric("High", counts.get("High", 0))
        count_b.metric("Medium", counts.get("Medium", 0))
        count_c.metric("Low", counts.get("Low", 0))
        count_d.metric("Do Not Trust", counts.get("DoNotTrust", 0))

        summary = scan_report.asset_horizon_summary
        if summary is None or summary.empty:
            st.warning("No scanner summary was produced.")
        else:
            total_rows = len(summary)
            if counts.get("DoNotTrust", 0) == total_rows:
                st.error("All scanned asset-horizon combinations are Do Not Trust for Signals.")

            st.markdown("### Asset × Horizon Summary")
            st.dataframe(summary, width="stretch")

            csv = summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Scanner CSV",
                data=csv,
                file_name="direct_horizon_scanner.csv",
                mime="text/csv",
            )

        tab_top, tab_worst, tab_errors, tab_leakage = st.tabs([
            "Top Promising",
            "Worst Failed",
            "Errors",
            "Leakage Check",
        ])

        with tab_top:
            st.markdown("### Top Promising Asset-Horizon Combinations")
            if scan_report.top_promising is not None and not scan_report.top_promising.empty:
                st.dataframe(scan_report.top_promising, width="stretch")
            else:
                st.warning("No non-DoNotTrust combinations were found in this scan.")

        with tab_worst:
            st.markdown("### Worst Failed Combinations")
            if scan_report.worst_failed is not None and not scan_report.worst_failed.empty:
                st.dataframe(scan_report.worst_failed, width="stretch")
            else:
                st.info("No failed combinations available.")

        with tab_errors:
            st.markdown("### Scan Errors")
            if scan_report.errors is not None and not scan_report.errors.empty:
                st.dataframe(scan_report.errors, width="stretch")
            else:
                st.success("No scanner errors in the last run.")

        with tab_leakage:
            st.markdown("### Future Target Feature Check")
            if summary is not None and not summary.empty and "FeatureLeakageCount" in summary.columns:
                leaks = summary[summary["FeatureLeakageCount"].fillna(0).astype(float) > 0]
                if leaks.empty:
                    st.success("No future_return_*, future_direction_*, or future_realized_vol_* columns were used as features.")
                else:
                    st.error("Future target columns leaked into scanner features. Do not trust this run.")
                    st.dataframe(leaks[["Asset", "Horizon", "FeatureLeakageColumns"]], width="stretch")
            else:
                st.info("No leakage metadata available.")


# ════════════════════════════════════════════════════════════════
# PAGE: SIGNAL RESEARCH SCANNER
# ════════════════════════════════════════════════════════════════

elif page == "🧪 Signal Research Scanner":
    st.markdown('<p class="main-header">🧪 Signal Research Scanner</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 7D is a validation-locked research scanner. It uses non-overlapping realistic trades only, "
        "selects thresholds and cooldowns from validation evidence only, and never labels results production-ready."
    )

    all_signal_scan_assets = get_asset_names()
    scan_col_a, scan_col_b, scan_col_c = st.columns([2, 2, 1])
    with scan_col_a:
        signal_scan_assets = st.multiselect(
            "Assets",
            all_signal_scan_assets,
            default=all_signal_scan_assets,
            help="Serious signal research should include every configured asset.",
            key="signal_research_scan_assets",
        )
    with scan_col_b:
        signal_scan_horizon_labels = st.multiselect(
            "Horizons",
            [f"{h}D" for h in DIRECT_FORECAST_HORIZONS],
            default=[f"{h}D" for h in DIRECT_FORECAST_HORIZONS],
            key="signal_research_scan_horizons",
        )
        signal_scan_horizons = [int(str(label).replace("D", "")) for label in signal_scan_horizon_labels]
    with scan_col_c:
        signal_scan_depth = st.selectbox(
            "Model depth",
            ["core", "fast"],
            index=0,
            help="core is the serious default; fast is useful for smoke tests.",
            key="signal_research_scan_depth",
        )
        signal_scan_phase5 = st.checkbox(
            "Use Phase 5",
            value=True,
            help="Adds FI_* features before direct horizon targets are created.",
            key="signal_research_scan_phase5",
        )

    rule_col_a, rule_col_b, rule_col_c = st.columns([2, 2, 1])
    with rule_col_a:
        threshold_candidates = st.multiselect(
            "Threshold candidates",
            [0.50, 0.55, 0.60, 0.65, 0.70],
            default=[0.50, 0.55, 0.60, 0.65, 0.70],
            format_func=lambda x: f"{x:.2f}",
            key="signal_research_thresholds",
        )
    with rule_col_b:
        cooldown_candidates = st.multiselect(
            "Cooldown candidates",
            [0, 2, 5],
            default=[0, 2, 5],
            format_func=lambda x: f"{int(x)} rows",
            key="signal_research_cooldowns",
        )
    with rule_col_c:
        validation_segment_pct = st.slider(
            "Validation segment %",
            min_value=30,
            max_value=70,
            value=50,
            step=5,
            help="Chronological share of Phase 6 out-of-sample rows used for threshold and cooldown selection.",
            key="signal_research_validation_segment",
        )

    st.caption(
        "Signal mode: long_only. Backtest style: non_overlapping_realistic. Threshold policy: validation_locked. "
        "Cooldown selection basis: validation score only."
    )

    run_signal_scan = st.button("🚀 Run Signal Research Scan", type="primary")

    if run_signal_scan:
        if not signal_scan_assets:
            st.error("Select at least one asset.")
            st.stop()
        if not signal_scan_horizons:
            st.error("Select at least one horizon.")
            st.stop()
        if not threshold_candidates:
            st.error("Select at least one threshold candidate.")
            st.stop()
        if not cooldown_candidates:
            st.error("Select at least one cooldown candidate.")
            st.stop()

        progress = st.progress(0, text="Preparing signal research scan...")

        def _signal_scan_progress(done, total, msg):
            pct = int(done / max(total, 1) * 100)
            progress.progress(pct, text=msg)

        with st.spinner("Running validation-locked signal research scanner..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                signal_scan_report = run_signal_research_scan(
                    raw_df=raw_df,
                    asset_names=signal_scan_assets,
                    horizons=signal_scan_horizons,
                    model_depth=signal_scan_depth,
                    use_phase5_features=signal_scan_phase5,
                    signal_mode="long_only",
                    threshold_candidates=threshold_candidates,
                    cooldown_candidates=cooldown_candidates,
                    validation_fraction=float(validation_segment_pct) / 100.0,
                    progress_callback=_signal_scan_progress,
                )
                st.session_state.signal_research_scan_report = signal_scan_report
                st.session_state.signal_research_scan_settings = signal_scan_report.settings
                progress.progress(100, text="Signal research scan complete.")
            except Exception as exc:
                st.error(f"Signal research scan failed: {exc}")
                st.stop()

    signal_scan_report = st.session_state.signal_research_scan_report
    if signal_scan_report is None:
        st.info("Run the scanner to evaluate validation-locked realistic signals across selected assets and horizons.")
    else:
        settings = st.session_state.get("signal_research_scan_settings") or {}
        st.caption(f"Last run settings: {settings}")

        counts = signal_scan_report.verdict_counts or {}
        count_cols = st.columns(max(1, min(4, len(counts) or 1)))
        if counts:
            for idx, (label, value) in enumerate(counts.items()):
                count_cols[idx % len(count_cols)].metric(label, value)
        else:
            count_cols[0].metric("Results", 0)

        full_results = signal_scan_report.full_results
        top_candidates = signal_scan_report.top_robust_candidates
        failed_candidates = signal_scan_report.failed_candidates

        if full_results is None or full_results.empty:
            st.warning("No signal scanner results were produced.")
        else:
            if top_candidates is None or top_candidates.empty:
                st.error("All scanned combinations failed the robust validation-locked signal criteria.")

            st.markdown("### Full Signal Research Results")
            st.dataframe(full_results, width="stretch")
            full_csv = full_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Full Signal Scan CSV",
                data=full_csv,
                file_name="signal_research_scanner.csv",
                mime="text/csv",
            )

        tab_top, tab_failed, tab_candidates, tab_errors = st.tabs([
            "Top Robust Candidates",
            "Failed / Weak",
            "Cooldown Candidates",
            "Errors",
        ])

        with tab_top:
            st.markdown("### Top Robust Candidates")
            if top_candidates is not None and not top_candidates.empty:
                st.dataframe(top_candidates, width="stretch")
            else:
                st.warning("No robust research candidates survived validation-locked realistic evaluation.")

        with tab_failed:
            st.markdown("### Failed And Weak Combinations")
            if failed_candidates is not None and not failed_candidates.empty:
                st.dataframe(failed_candidates, width="stretch")
            else:
                st.success("No failed or weak combinations in this scan.")

        with tab_candidates:
            st.markdown("### Cooldown Candidate Diagnostics")
            candidate_results = signal_scan_report.candidate_results
            if candidate_results is not None and not candidate_results.empty:
                st.caption("Cooldown candidates are validation-only diagnostics. Locked test is evaluated once after the cooldown is selected.")
                st.dataframe(candidate_results, width="stretch")
                candidate_csv = candidate_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Cooldown Candidate CSV",
                    data=candidate_csv,
                    file_name="signal_research_cooldown_candidates.csv",
                    mime="text/csv",
                )
            else:
                st.info("No cooldown candidate diagnostics available.")

        with tab_errors:
            st.markdown("### Scan Errors")
            if signal_scan_report.errors is not None and not signal_scan_report.errors.empty:
                st.dataframe(signal_scan_report.errors, width="stretch")
            else:
                st.success("No scanner errors in the last run.")


# ════════════════════════════════════════════════════════════════
# PAGE: CANDIDATE DEEP DIAGNOSTICS
# ════════════════════════════════════════════════════════════════

elif page == "🔬 Candidate Deep Diagnostics":
    st.markdown('<p class="main-header">🔬 Candidate Deep Diagnostics</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 7E diagnoses validation-locked signal candidates. Thresholds and cooldowns are selected from validation evidence only; "
        "locked-test data is used only for post-selection diagnostics. Nothing here is production-ready."
    )

    diag_assets = get_asset_names()
    diag_default_asset = diag_assets.index(selected_asset) if selected_asset in diag_assets else 0
    diag_col_a, diag_col_b, diag_col_c, diag_col_d = st.columns(4)
    with diag_col_a:
        diag_asset = st.selectbox(
            "Asset",
            diag_assets,
            index=diag_default_asset,
            key="candidate_diag_asset",
        )
    with diag_col_b:
        diag_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        diag_horizon_default = diag_horizon_labels.index("5D") if "5D" in diag_horizon_labels else 0
        diag_horizon_label = st.selectbox(
            "Horizon",
            diag_horizon_labels,
            index=diag_horizon_default,
            key="candidate_diag_horizon",
        )
        diag_horizon = int(str(diag_horizon_label).replace("D", ""))
    with diag_col_c:
        diag_depth = st.selectbox(
            "Model depth",
            ["core", "fast"],
            index=0,
            key="candidate_diag_depth",
        )
    with diag_col_d:
        diag_phase5 = st.checkbox(
            "Use Phase 5",
            value=True,
            key="candidate_diag_phase5",
        )

    diag_rule_a, diag_rule_b, diag_rule_c, diag_rule_d = st.columns([1, 2, 2, 1])
    with diag_rule_a:
        diag_mode = st.selectbox(
            "Signal mode",
            ["long_only", "long_short", "avoid_only"],
            index=0,
            key="candidate_diag_mode",
        )
    with diag_rule_b:
        diag_thresholds = st.multiselect(
            "Threshold candidates",
            [0.50, 0.55, 0.60, 0.65, 0.70],
            default=[0.50, 0.55, 0.60, 0.65, 0.70],
            format_func=lambda x: f"{x:.2f}",
            key="candidate_diag_thresholds",
        )
    with diag_rule_c:
        diag_cooldowns = st.multiselect(
            "Cooldown candidates",
            [0, 2, 5],
            default=[0, 2, 5],
            format_func=lambda x: f"{int(x)} rows",
            key="candidate_diag_cooldowns",
        )
    with diag_rule_d:
        diag_validation_pct = st.slider(
            "Validation %",
            min_value=30,
            max_value=70,
            value=50,
            step=5,
            key="candidate_diag_validation_pct",
        )

    run_candidate_diag = st.button("🚀 Run Candidate Diagnostics", type="primary")

    if run_candidate_diag:
        if not diag_thresholds:
            st.error("Select at least one threshold candidate.")
            st.stop()
        if not diag_cooldowns:
            st.error("Select at least one cooldown candidate.")
            st.stop()

        with st.spinner("Running validation-locked candidate diagnostics..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                diag_report = run_candidate_deep_diagnostics(
                    raw_df=raw_df,
                    asset_name=diag_asset,
                    horizon=diag_horizon,
                    model_depth=diag_depth,
                    use_phase5_features=diag_phase5,
                    signal_mode=diag_mode,
                    threshold_candidates=diag_thresholds,
                    cooldown_candidates=diag_cooldowns,
                    validation_fraction=float(diag_validation_pct) / 100.0,
                )
                st.session_state.candidate_diagnostics_report = diag_report
                st.session_state.candidate_diagnostics_settings = diag_report.settings
            except Exception as exc:
                st.error(f"Candidate diagnostics failed: {exc}")
                st.stop()

    diag_report = st.session_state.candidate_diagnostics_report
    if diag_report is None:
        st.info("Run diagnostics for any configured asset and horizon. Silver 5D is only the default candidate view.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('candidate_diagnostics_settings') or {}}")

        if diag_report.warnings:
            for warning in diag_report.warnings:
                if warning.startswith("BenchmarkWeakness") or warning.startswith("DrawdownRisk") or warning.startswith("CostFragile"):
                    st.error(warning)
                else:
                    st.warning(warning)

        st.markdown("### Candidate Summary")
        st.dataframe(diag_report.candidate_summary, width="stretch")
        st.download_button(
            "📥 Export Candidate Summary CSV",
            data=diag_report.candidate_summary.to_csv(index=False).encode("utf-8"),
            file_name="candidate_diagnostics_summary.csv",
            mime="text/csv",
        )

        tab_trade, tab_time, tab_drawdown, tab_sensitivity, tab_probability = st.tabs([
            "Trade Diagnostics",
            "Time Returns",
            "Equity / Drawdown",
            "Sensitivity",
            "Probability",
        ])

        with tab_trade:
            st.markdown("### Trade Diagnostics")
            st.dataframe(diag_report.trade_diagnostics, width="stretch")
            st.download_button(
                "📥 Export Trade Diagnostics CSV",
                data=diag_report.trade_diagnostics.to_csv(index=False).encode("utf-8"),
                file_name="candidate_trade_diagnostics.csv",
                mime="text/csv",
            )

            st.markdown("### Trade Log")
            st.dataframe(diag_report.trade_log, width="stretch")
            st.download_button(
                "📥 Export Trade Log CSV",
                data=diag_report.trade_log.to_csv(index=False).encode("utf-8"),
                file_name="candidate_trade_log.csv",
                mime="text/csv",
            )

        with tab_time:
            st.markdown("### Monthly Returns")
            st.dataframe(diag_report.monthly_returns, width="stretch")
            st.download_button(
                "📥 Export Monthly Returns CSV",
                data=diag_report.monthly_returns.to_csv(index=False).encode("utf-8"),
                file_name="candidate_monthly_returns.csv",
                mime="text/csv",
            )

            st.markdown("### Quarterly Returns")
            st.dataframe(diag_report.quarterly_returns, width="stretch")
            st.download_button(
                "📥 Export Quarterly Returns CSV",
                data=diag_report.quarterly_returns.to_csv(index=False).encode("utf-8"),
                file_name="candidate_quarterly_returns.csv",
                mime="text/csv",
            )

        with tab_drawdown:
            st.markdown("### Equity Curve")
            if diag_report.equity_curve is not None and not diag_report.equity_curve.empty:
                equity_chart = diag_report.equity_curve.set_index("Date")[["Equity"]]
                st.line_chart(equity_chart)
            st.dataframe(diag_report.equity_curve, width="stretch")
            st.download_button(
                "📥 Export Equity Curve CSV",
                data=diag_report.equity_curve.to_csv(index=False).encode("utf-8"),
                file_name="candidate_equity_curve.csv",
                mime="text/csv",
            )

            st.markdown("### Drawdown Curve")
            if diag_report.drawdown_curve is not None and not diag_report.drawdown_curve.empty:
                dd_chart = diag_report.drawdown_curve.set_index("Date")[["Drawdown_%"]]
                st.line_chart(dd_chart)
            st.dataframe(diag_report.drawdown_curve, width="stretch")
            st.download_button(
                "📥 Export Drawdown Curve CSV",
                data=diag_report.drawdown_curve.to_csv(index=False).encode("utf-8"),
                file_name="candidate_drawdown_curve.csv",
                mime="text/csv",
            )

        with tab_sensitivity:
            st.markdown("### Cost Sensitivity")
            st.caption("Uses the validation-selected threshold and cooldown, then re-evaluates locked-test economics at each transaction cost.")
            st.dataframe(diag_report.cost_sensitivity, width="stretch")
            st.download_button(
                "📥 Export Cost Sensitivity CSV",
                data=diag_report.cost_sensitivity.to_csv(index=False).encode("utf-8"),
                file_name="candidate_cost_sensitivity.csv",
                mime="text/csv",
            )

            st.markdown("### Validation Split Sensitivity")
            st.caption("Each split re-runs validation-locked selection independently. Locked-test metrics are post-selection diagnostics.")
            st.dataframe(diag_report.validation_split_sensitivity, width="stretch")
            st.download_button(
                "📥 Export Split Sensitivity CSV",
                data=diag_report.validation_split_sensitivity.to_csv(index=False).encode("utf-8"),
                file_name="candidate_split_sensitivity.csv",
                mime="text/csv",
            )

        with tab_probability:
            st.markdown("### Probability Diagnostics")
            st.caption("These are descriptive P(up) diagnostics. They are not calibration claims.")
            st.dataframe(diag_report.probability_diagnostics, width="stretch")
            st.download_button(
                "📥 Export Probability Diagnostics CSV",
                data=diag_report.probability_diagnostics.to_csv(index=False).encode("utf-8"),
                file_name="candidate_probability_diagnostics.csv",
                mime="text/csv",
            )

            st.markdown("### Probability Bins")
            st.dataframe(diag_report.probability_bins, width="stretch")
            st.download_button(
                "📥 Export Probability Bins CSV",
                data=diag_report.probability_bins.to_csv(index=False).encode("utf-8"),
                file_name="candidate_probability_bins.csv",
                mime="text/csv",
            )


# ════════════════════════════════════════════════════════════════
# PAGE: RISK-CONTROLLED UPGRADE
# ════════════════════════════════════════════════════════════════

elif page == "🛡️ Risk-Controlled Upgrade":
    st.markdown('<p class="main-header">🛡️ Risk-Controlled Upgrade</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 7F tests risk controls as research variants. Variant selection is validation-only; locked test is used after selection. "
        "If risk control reduces drawdown but destroys return, that weakness stays visible."
    )

    rc_assets = get_asset_names()
    rc_default_asset = rc_assets.index(selected_asset) if selected_asset in rc_assets else 0
    rc_col_a, rc_col_b, rc_col_c, rc_col_d = st.columns(4)
    with rc_col_a:
        rc_asset = st.selectbox("Asset", rc_assets, index=rc_default_asset, key="risk_control_asset")
    with rc_col_b:
        rc_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        rc_default_horizon = rc_horizon_labels.index("5D") if "5D" in rc_horizon_labels else 0
        rc_horizon_label = st.selectbox("Horizon", rc_horizon_labels, index=rc_default_horizon, key="risk_control_horizon")
        rc_horizon = int(str(rc_horizon_label).replace("D", ""))
    with rc_col_c:
        rc_depth = st.selectbox("Model depth", ["core", "fast"], index=0, key="risk_control_depth")
    with rc_col_d:
        rc_phase5 = st.checkbox("Use Phase 5", value=True, key="risk_control_phase5")

    rc_rule_a, rc_rule_b, rc_rule_c, rc_rule_d = st.columns([1, 2, 2, 1])
    with rc_rule_a:
        rc_mode = st.selectbox("Signal mode", ["long_only", "long_short", "avoid_only"], index=0, key="risk_control_mode")
    with rc_rule_b:
        rc_thresholds = st.multiselect(
            "Threshold candidates",
            [0.50, 0.55, 0.60, 0.65, 0.70],
            default=[0.50, 0.55, 0.60, 0.65, 0.70],
            format_func=lambda x: f"{x:.2f}",
            key="risk_control_thresholds",
        )
    with rc_rule_c:
        rc_cooldowns = st.multiselect(
            "Cooldown candidates",
            [0, 2, 5],
            default=[0, 2, 5],
            format_func=lambda x: f"{int(x)} rows",
            key="risk_control_cooldowns",
        )
    with rc_rule_d:
        rc_validation_pct = st.slider("Validation %", min_value=30, max_value=70, value=50, step=5, key="risk_control_validation_pct")

    rc_variant_labels = {
        "Baseline signal": "baseline",
        "Volatility filter": "volatility_filter",
        "Drawdown stop rule": "drawdown_stop",
        "Loss-streak stop rule": "loss_streak_stop",
        "Probability band filter": "probability_band_filter",
        "Position sizing simulation": "position_sizing",
    }
    rc_selected_variant_labels = st.multiselect(
        "Risk-control variants to test",
        list(rc_variant_labels.keys()),
        default=list(rc_variant_labels.keys()),
        key="risk_control_variants",
    )
    rc_variant_keys = [rc_variant_labels[label] for label in rc_selected_variant_labels]

    cost_col_a, cost_col_b = st.columns([2, 1])
    with cost_col_a:
        rc_cost_pct_values = st.multiselect(
            "Cost stress options",
            [0.00, 0.05, 0.10, 0.20, 0.50],
            default=[0.00, 0.05, 0.10, 0.20, 0.50],
            format_func=lambda x: f"{x:.2f}%",
            key="risk_control_cost_stress",
        )
    with cost_col_b:
        rc_transaction_cost_pct = st.number_input(
            "Base transaction cost %",
            min_value=0.0,
            max_value=2.0,
            value=0.10,
            step=0.01,
            key="risk_control_transaction_cost",
        )

    run_risk_upgrade = st.button("🚀 Run Risk-Controlled Upgrade", type="primary")

    if run_risk_upgrade:
        if not rc_thresholds:
            st.error("Select at least one threshold candidate.")
            st.stop()
        if not rc_cooldowns:
            st.error("Select at least one cooldown candidate.")
            st.stop()
        if not rc_variant_keys:
            st.error("Select at least one risk-control variant.")
            st.stop()
        if not rc_cost_pct_values:
            st.error("Select at least one cost stress option.")
            st.stop()

        with st.spinner("Running validation-only risk-control upgrade..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                risk_report = run_risk_controlled_candidate_upgrade(
                    raw_df=raw_df,
                    asset_name=rc_asset,
                    horizon=rc_horizon,
                    model_depth=rc_depth,
                    use_phase5_features=rc_phase5,
                    signal_mode=rc_mode,
                    threshold_candidates=rc_thresholds,
                    cooldown_candidates=rc_cooldowns,
                    validation_fraction=float(rc_validation_pct) / 100.0,
                    transaction_cost=float(rc_transaction_cost_pct) / 100.0,
                    risk_variant_names=rc_variant_keys,
                    cost_values=[float(v) / 100.0 for v in rc_cost_pct_values],
                )
                st.session_state.risk_control_upgrade_report = risk_report
                st.session_state.risk_control_upgrade_settings = risk_report.settings
            except Exception as exc:
                st.error(f"Risk-controlled upgrade failed: {exc}")
                st.stop()

    risk_report = st.session_state.risk_control_upgrade_report
    if risk_report is None:
        st.info("Run the upgrade for any configured asset and horizon. Silver 5D is only the default candidate view.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('risk_control_upgrade_settings') or {}}")
        if risk_report.warnings:
            for warning in risk_report.warnings:
                if any(flag in warning for flag in ["CostFragile", "DrawdownRisk", "ReturnDestroyed", "NoImprovement"]):
                    st.warning(warning)
                else:
                    st.info(warning)

        selected_info = risk_report.selected_variant or {}
        st.markdown("### Selected Variant")
        st.json(selected_info)

        st.markdown("### Baseline vs Best Risk-Controlled Variant")
        st.dataframe(risk_report.baseline_vs_best, width="stretch")
        st.download_button(
            "📥 Export Baseline vs Best CSV",
            data=risk_report.baseline_vs_best.to_csv(index=False).encode("utf-8"),
            file_name="risk_control_baseline_vs_best.csv",
            mime="text/csv",
        )

        tab_variants, tab_costs = st.tabs(["Full Variant Table", "Cost Stress"])

        with tab_variants:
            st.markdown("### Full Variant Table")
            st.caption("Non-selected variants show validation metrics only. Locked test is evaluated for the baseline reference and validation-selected variant.")
            st.dataframe(risk_report.full_variant_table, width="stretch")
            st.download_button(
                "📥 Export Full Variant CSV",
                data=risk_report.full_variant_table.to_csv(index=False).encode("utf-8"),
                file_name="risk_control_full_variants.csv",
                mime="text/csv",
            )

        with tab_costs:
            st.markdown("### Cost / Slippage Stress")
            st.caption("Post-selection stress table. It is not used to choose the risk-control variant.")
            st.dataframe(risk_report.cost_stress_table, width="stretch")
            st.download_button(
                "📥 Export Cost Stress CSV",
                data=risk_report.cost_stress_table.to_csv(index=False).encode("utf-8"),
                file_name="risk_control_cost_stress.csv",
                mime="text/csv",
            )


# ════════════════════════════════════════════════════════════════
# PAGE: WALK-FORWARD VALIDATION
# ════════════════════════════════════════════════════════════════

elif page == "🧭 Walk-Forward Validation":
    st.markdown('<p class="main-header">🧭 Walk-Forward Validation</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 7G asks whether candidates survive repeated rolling validation/locked-test windows. "
        "Each window selects threshold and cooldown from validation rows only; locked-test rows are evaluation-only."
    )

    wf_assets_all = get_asset_names()
    wf_default_assets = [asset for asset in ["Silver", "Crude Oil"] if asset in wf_assets_all] or wf_assets_all
    wf_col_a, wf_col_b, wf_col_c = st.columns([2, 2, 1])
    with wf_col_a:
        wf_assets = st.multiselect(
            "Assets",
            wf_assets_all,
            default=wf_default_assets,
            key="walk_forward_assets",
        )
    with wf_col_b:
        wf_horizon_options = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        wf_default_horizons = [label for label in ["5D", "1D"] if label in wf_horizon_options] or wf_horizon_options[:1]
        wf_horizon_labels = st.multiselect(
            "Horizons",
            wf_horizon_options,
            default=wf_default_horizons,
            key="walk_forward_horizons",
        )
        wf_horizons = [int(str(label).replace("D", "")) for label in wf_horizon_labels]
    with wf_col_c:
        wf_depth = st.selectbox("Model depth", ["core", "fast"], index=0, key="walk_forward_depth")
        wf_phase5 = st.checkbox("Use Phase 5", value=True, key="walk_forward_phase5")

    wf_rule_a, wf_rule_b, wf_rule_c, wf_rule_d = st.columns([1, 2, 2, 1])
    with wf_rule_a:
        wf_mode = st.selectbox("Signal mode", ["long_only", "long_short", "avoid_only"], index=0, key="walk_forward_mode")
    with wf_rule_b:
        wf_thresholds = st.multiselect(
            "Threshold candidates",
            [0.50, 0.55, 0.60, 0.65, 0.70],
            default=[0.50, 0.55, 0.60, 0.65, 0.70],
            format_func=lambda x: f"{x:.2f}",
            key="walk_forward_thresholds",
        )
    with wf_rule_c:
        wf_cooldowns = st.multiselect(
            "Cooldown candidates",
            [0, 2, 5],
            default=[0, 2, 5],
            format_func=lambda x: f"{int(x)} rows",
            key="walk_forward_cooldowns",
        )
    with wf_rule_d:
        wf_cost_pct = st.number_input(
            "Transaction cost %",
            min_value=0.0,
            max_value=2.0,
            value=0.10,
            step=0.01,
            key="walk_forward_transaction_cost",
        )

    wf_win_a, wf_win_b, wf_win_c, wf_win_d, wf_win_e = st.columns(5)
    with wf_win_a:
        wf_validation_rows = st.number_input("Validation rows", min_value=30, max_value=500, value=180, step=10, key="walk_forward_validation_rows")
    with wf_win_b:
        wf_test_rows = st.number_input("Test rows", min_value=20, max_value=250, value=90, step=10, key="walk_forward_test_rows")
    with wf_win_c:
        wf_step_rows = st.number_input("Step rows", min_value=10, max_value=250, value=60, step=10, key="walk_forward_step_rows")
    with wf_win_d:
        wf_min_trades = st.number_input("Min trades/window", min_value=1, max_value=50, value=3, step=1, key="walk_forward_min_trades")
    with wf_win_e:
        wf_window_mode = st.selectbox("Window mode", ["rolling", "expanding"], index=0, key="walk_forward_window_mode")

    run_wf = st.button("🚀 Run Walk-Forward Validation", type="primary")

    if run_wf:
        if not wf_assets:
            st.error("Select at least one asset.")
            st.stop()
        if not wf_horizons:
            st.error("Select at least one horizon.")
            st.stop()
        if not wf_thresholds:
            st.error("Select at least one threshold candidate.")
            st.stop()
        if not wf_cooldowns:
            st.error("Select at least one cooldown candidate.")
            st.stop()

        progress = st.progress(0, text="Preparing walk-forward validation...")

        def _wf_progress(done, total, msg):
            pct = int(done / max(total, 1) * 100)
            progress.progress(pct, text=msg)

        with st.spinner("Running walk-forward validation..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                wf_report = run_walk_forward_risk_validation(
                    raw_df=raw_df,
                    asset_names=wf_assets,
                    horizons=wf_horizons,
                    model_depth=wf_depth,
                    use_phase5_features=wf_phase5,
                    signal_mode=wf_mode,
                    threshold_candidates=wf_thresholds,
                    cooldown_candidates=wf_cooldowns,
                    transaction_cost=float(wf_cost_pct) / 100.0,
                    validation_window=int(wf_validation_rows),
                    test_window=int(wf_test_rows),
                    step_size=int(wf_step_rows),
                    min_trades_per_window=int(wf_min_trades),
                    window_mode=wf_window_mode,
                    progress_callback=_wf_progress,
                )
                st.session_state.walk_forward_validation_report = wf_report
                st.session_state.walk_forward_validation_settings = wf_report.settings
                progress.progress(100, text="Walk-forward validation complete.")
            except Exception as exc:
                st.error(f"Walk-forward validation failed: {exc}")
                st.stop()

    wf_report = st.session_state.walk_forward_validation_report
    if wf_report is None:
        st.info("Run walk-forward validation to test whether candidates repeat across rolling historical windows.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('walk_forward_validation_settings') or {}}")

        counts = wf_report.verdict_counts or {}
        count_cols = st.columns(max(1, min(4, len(counts) or 1)))
        if counts:
            for idx, (label, value) in enumerate(counts.items()):
                count_cols[idx % len(count_cols)].metric(label, value)
        else:
            count_cols[0].metric("Results", 0)

        if wf_report.warnings:
            for warning in wf_report.warnings[:8]:
                st.warning(warning)
            if len(wf_report.warnings) > 8:
                st.caption(f"{len(wf_report.warnings) - 8} additional warnings are included in the tables.")

        st.markdown("### Aggregate Walk-Forward Summary")
        st.dataframe(wf_report.aggregate_summary, width="stretch")
        st.download_button(
            "📥 Export Aggregate CSV",
            data=wf_report.aggregate_summary.to_csv(index=False).encode("utf-8"),
            file_name="walk_forward_aggregate_summary.csv",
            mime="text/csv",
        )

        tab_windows, tab_errors = st.tabs(["Per-Window Results", "Errors"])
        with tab_windows:
            st.markdown("### Per-Window Results")
            st.dataframe(wf_report.window_results, width="stretch")
            st.download_button(
                "📥 Export Per-Window CSV",
                data=wf_report.window_results.to_csv(index=False).encode("utf-8"),
                file_name="walk_forward_window_results.csv",
                mime="text/csv",
            )

        with tab_errors:
            st.markdown("### Walk-Forward Errors")
            if wf_report.errors is not None and not wf_report.errors.empty:
                st.dataframe(wf_report.errors, width="stretch")
            else:
                st.success("No walk-forward errors in the last run.")


# ════════════════════════════════════════════════════════════════
# PAGE: REGIME-AWARE META SIGNAL
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

elif page == "🧠 Regime-Aware Meta Signal":
    st.markdown('<p class="main-header">🧠 Regime-Aware Meta Signal</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 8 is a rule-based meta layer. It consumes Phase 7G walk-forward aggregate reliability, "
        "adds current/historical regime context, and never labels results production-ready."
    )

    meta_assets_all = get_asset_names()
    meta_col_a, meta_col_b, meta_col_c = st.columns([2, 2, 1])
    with meta_col_a:
        meta_assets = st.multiselect(
            "Assets",
            meta_assets_all,
            default=meta_assets_all,
            key="meta_signal_assets",
        )
    with meta_col_b:
        meta_horizon_options = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        meta_horizon_labels = st.multiselect(
            "Horizons",
            meta_horizon_options,
            default=meta_horizon_options,
            key="meta_signal_horizons",
        )
        meta_horizons = [int(str(label).replace("D", "")) for label in meta_horizon_labels]
    with meta_col_c:
        meta_depth = st.selectbox("Model depth", ["core", "fast"], index=0, key="meta_signal_depth")
        meta_phase5 = st.checkbox("Use Phase 5", value=True, key="meta_signal_phase5")

    meta_rule_a, meta_rule_b = st.columns([1, 2])
    with meta_rule_a:
        meta_mode = st.selectbox("Signal mode", ["long_only", "long_short", "avoid_only"], index=0, key="meta_signal_mode")
    with meta_rule_b:
        st.caption(
            "Regime features use only current and historical prices: rolling returns, volatility, drawdown, "
            "moving-average trend, and recent passive-hold strength."
        )

    latest_wf_report = st.session_state.get("walk_forward_validation_report")
    latest_wf_summary = None
    if latest_wf_report is not None and getattr(latest_wf_report, "aggregate_summary", None) is not None:
        latest_wf_summary = latest_wf_report.aggregate_summary

    uploaded_wf_csv = st.file_uploader(
        "Upload Phase 7G aggregate CSV",
        type=["csv"],
        key="meta_signal_wf_upload",
        help="Use this when no walk-forward aggregate is available in the current session.",
    )

    source_options = []
    if latest_wf_summary is not None and not latest_wf_summary.empty:
        source_options.append("Latest Phase 7G session result")
    if uploaded_wf_csv is not None:
        source_options.append("Uploaded Phase 7G aggregate CSV")

    if source_options:
        meta_source = st.radio("Phase 7G reliability source", source_options, horizontal=True, key="meta_signal_source")
    else:
        meta_source = None
        st.info("Run Phase 7G Walk-Forward Validation first, or upload a Phase 7G aggregate CSV.")

    run_meta = st.button("🚀 Run Meta Signal Analysis", type="primary")

    if run_meta:
        if not meta_assets:
            st.error("Select at least one asset.")
            st.stop()
        if not meta_horizons:
            st.error("Select at least one horizon.")
            st.stop()
        if meta_source is None:
            st.error("Provide a Phase 7G walk-forward aggregate source.")
            st.stop()

        try:
            if meta_source == "Latest Phase 7G session result":
                wf_summary_for_meta = latest_wf_summary.copy()
            else:
                uploaded_wf_csv.seek(0)
                wf_summary_for_meta = pd.read_csv(uploaded_wf_csv)
        except Exception as exc:
            st.error(f"Could not read Phase 7G aggregate data: {exc}")
            st.stop()

        if wf_summary_for_meta is None or wf_summary_for_meta.empty:
            st.error("Phase 7G aggregate data is empty.")
            st.stop()

        with st.spinner("Running regime-aware meta signal analysis..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                meta_report = run_regime_aware_meta_signal(
                    raw_df=raw_df,
                    walk_forward_summary=wf_summary_for_meta,
                    asset_names=meta_assets,
                    horizons=meta_horizons,
                    model_depth=meta_depth,
                    use_phase5_features=meta_phase5,
                    signal_mode=meta_mode,
                )
                st.session_state.meta_signal_report = meta_report
                st.session_state.meta_signal_settings = meta_report.settings
            except Exception as exc:
                st.error(f"Meta signal analysis failed: {exc}")
                st.stop()

    meta_report = st.session_state.meta_signal_report
    if meta_report is None:
        st.info("Run meta signal analysis to convert Phase 7G reliability into regime-aware decisions.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('meta_signal_settings') or {}}")

        summary = meta_report.decision_summary
        metric_cols = st.columns(5)
        for idx, decision in enumerate(["Trade", "No Trade", "Defensive Only", "Research Only", "Avoid"]):
            match = summary[summary["MetaDecision"].eq(decision)]
            value = int(match.iloc[0]["Count"]) if not match.empty else 0
            metric_cols[idx].metric(decision, value)

        if meta_report.warnings:
            for warning in meta_report.warnings[:10]:
                st.warning(warning)
            if len(meta_report.warnings) > 10:
                st.caption(f"{len(meta_report.warnings) - 10} additional warnings are included in the decision table.")

        st.markdown("### Meta Decision Table")
        st.dataframe(meta_report.decision_table, width="stretch")
        st.download_button(
            "📥 Export Meta Decisions CSV",
            data=meta_report.decision_table.to_csv(index=False).encode("utf-8"),
            file_name="regime_aware_meta_signal_decisions.csv",
            mime="text/csv",
            key="meta_signal_decisions_download",
        )

        st.markdown("### Grouped Decisions")
        tabs = st.tabs(["Trade", "No Trade", "Defensive Only", "Research Only", "Avoid"])
        for tab, decision in zip(tabs, ["Trade", "No Trade", "Defensive Only", "Research Only", "Avoid"]):
            with tab:
                subset = meta_report.decision_table[meta_report.decision_table["MetaDecision"].eq(decision)]
                if subset.empty:
                    st.info(f"No {decision} rows in the latest meta analysis.")
                else:
                    st.dataframe(subset, width="stretch")

        diag_a, diag_b, diag_c = st.tabs(["Regimes", "Reliability / Risk", "Summary"])
        with diag_a:
            st.markdown("### Current Regime Features")
            st.dataframe(meta_report.regime_features, width="stretch")
            st.download_button(
                "📥 Export Regime Features CSV",
                data=meta_report.regime_features.to_csv(index=False).encode("utf-8"),
                file_name="regime_features.csv",
                mime="text/csv",
                key="meta_signal_regime_download",
            )
        with diag_b:
            cols = [
                "Asset",
                "Horizon",
                "MetaDecision",
                "MetaConfidenceScore",
                "MetaRiskScore",
                "SignalReliabilityScore",
                "WalkForwardReliabilityScore",
                "RegimeLabel",
                "BenchmarkRiskFlag",
                "CostFragilityFlag",
                "DrawdownRiskFlag",
                "StabilityFlag",
                "Warnings",
            ]
            available_cols = [col for col in cols if col in meta_report.decision_table.columns]
            st.dataframe(meta_report.decision_table[available_cols], width="stretch")
        with diag_c:
            st.markdown("### Decision Summary")
            st.dataframe(summary, width="stretch")
            st.download_button(
                "📥 Export Decision Summary CSV",
                data=summary.to_csv(index=False).encode("utf-8"),
                file_name="regime_aware_meta_signal_summary.csv",
                mime="text/csv",
                key="meta_signal_summary_download",
            )


# PAGE: META DECISION AUDIT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

elif page == "🧾 Meta Decision Audit":
    st.markdown('<p class="main-header">🧾 Meta Decision Audit</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 8A is audit and calibration only. It explains existing Phase 8 meta decisions, "
        "does not rerun models, does not retune thresholds, and never marks anything production-ready."
    )

    latest_meta_report = st.session_state.get("meta_signal_report")
    latest_meta_table = None
    if latest_meta_report is not None and getattr(latest_meta_report, "decision_table", None) is not None:
        latest_meta_table = latest_meta_report.decision_table

    uploaded_meta_csv = st.file_uploader(
        "Upload Phase 8 meta decision CSV",
        type=["csv"],
        key="meta_audit_upload",
        help="Use this when no Phase 8 meta decision table is available in the current session.",
    )

    audit_sources = []
    if latest_meta_table is not None and not latest_meta_table.empty:
        audit_sources.append("Latest Phase 8 session result")
    if uploaded_meta_csv is not None:
        audit_sources.append("Uploaded Phase 8 meta decision CSV")

    audit_col_a, audit_col_b = st.columns([1, 2])
    with audit_col_a:
        audit_mode = st.selectbox(
            "Calibration mode",
            ["Conservative", "Balanced", "Aggressive Research"],
            index=0,
            key="meta_audit_mode",
        )
    with audit_col_b:
        if audit_mode == "Conservative":
            st.caption("Strict audit mode: mirrors Phase 8's conservative posture.")
        elif audit_mode == "Balanced":
            st.caption("Research calibration mode: may surface more Research Only / Defensive Only rows, but Trade remains strict.")
        else:
            st.caption("Experimental research mode: may surface candidates for investigation, never production readiness.")

    if audit_sources:
        audit_source = st.radio("Phase 8 decision source", audit_sources, horizontal=True, key="meta_audit_source")
    else:
        audit_source = None
        st.info("Run Phase 8 Regime-Aware Meta Signal first, or upload a Phase 8 meta decision CSV.")

    run_audit = st.button("🚀 Run Meta Decision Audit", type="primary")

    if run_audit:
        if audit_source is None:
            st.error("Provide a Phase 8 meta decision source.")
            st.stop()

        try:
            if audit_source == "Latest Phase 8 session result":
                meta_table_for_audit = latest_meta_table.copy()
            else:
                uploaded_meta_csv.seek(0)
                meta_table_for_audit = pd.read_csv(uploaded_meta_csv)
        except Exception as exc:
            st.error(f"Could not read Phase 8 meta decision data: {exc}")
            st.stop()

        if meta_table_for_audit is None or meta_table_for_audit.empty:
            st.error("Phase 8 meta decision data is empty.")
            st.stop()

        with st.spinner("Running meta decision audit..."):
            try:
                audit_report = run_meta_decision_audit(meta_table_for_audit, calibration_mode=audit_mode)
                st.session_state.meta_decision_audit_report = audit_report
                st.session_state.meta_decision_audit_settings = audit_report.settings
            except Exception as exc:
                st.error(f"Meta decision audit failed: {exc}")
                st.stop()

    audit_report = st.session_state.meta_decision_audit_report
    if audit_report is None:
        st.info("Run the audit to see blocking rules, passing rules, near misses, and calibration tables.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('meta_decision_audit_settings') or {}}")

        if audit_report.warnings:
            for warning in audit_report.warnings:
                st.warning(warning)

        st.markdown("### Meta Decision Audit Table")
        st.dataframe(audit_report.audit_table, width="stretch")
        st.download_button(
            "📥 Export Audit CSV",
            data=audit_report.audit_table.to_csv(index=False).encode("utf-8"),
            file_name="meta_decision_audit.csv",
            mime="text/csv",
            key="meta_audit_table_download",
        )

        tab_blocking, tab_passing, tab_near, tab_rankings, tab_modes, tab_thresholds = st.tabs(
            ["Blocking Rules", "Passing Rules", "Near Misses", "Candidate Rankings", "Mode Comparison", "Thresholds"]
        )

        with tab_blocking:
            st.markdown("### Most Common Blocking Rules")
            st.dataframe(audit_report.common_blocking_rules, width="stretch")
            st.download_button(
                "📥 Export Blocking Rules CSV",
                data=audit_report.common_blocking_rules.to_csv(index=False).encode("utf-8"),
                file_name="meta_audit_blocking_rules.csv",
                mime="text/csv",
                key="meta_audit_blocking_download",
            )

            cols = ["Asset", "Horizon", "Current MetaDecision", "Calibrated MetaDecision", "MainBlockingRule", "BlockingRules"]
            available = [col for col in cols if col in audit_report.audit_table.columns]
            st.dataframe(audit_report.audit_table[available], width="stretch")

        with tab_passing:
            st.markdown("### Passing Rules")
            cols = ["Asset", "Horizon", "Current MetaDecision", "Calibrated MetaDecision", "PassingRules"]
            available = [col for col in cols if col in audit_report.audit_table.columns]
            passing_table = audit_report.audit_table[available]
            st.dataframe(passing_table, width="stretch")
            st.download_button(
                "📥 Export Passing Rules CSV",
                data=passing_table.to_csv(index=False).encode("utf-8"),
                file_name="meta_audit_passing_rules.csv",
                mime="text/csv",
                key="meta_audit_passing_download",
            )

        with tab_near:
            st.markdown("### Top Near-Miss Candidates")
            st.dataframe(audit_report.near_miss_candidates, width="stretch")
            st.download_button(
                "📥 Export Near Misses CSV",
                data=audit_report.near_miss_candidates.to_csv(index=False).encode("utf-8"),
                file_name="meta_audit_near_misses.csv",
                mime="text/csv",
                key="meta_audit_near_download",
            )

        with tab_rankings:
            rank_a, rank_b, rank_c = st.tabs(["Top Blocked", "Highest Confidence", "Highest Risk"])
            with rank_a:
                st.dataframe(audit_report.top_blocked_candidates, width="stretch")
                st.download_button(
                    "📥 Export Top Blocked CSV",
                    data=audit_report.top_blocked_candidates.to_csv(index=False).encode("utf-8"),
                    file_name="meta_audit_top_blocked.csv",
                    mime="text/csv",
                    key="meta_audit_top_blocked_download",
                )
            with rank_b:
                st.dataframe(audit_report.highest_confidence_candidates, width="stretch")
                st.download_button(
                    "📥 Export Highest Confidence CSV",
                    data=audit_report.highest_confidence_candidates.to_csv(index=False).encode("utf-8"),
                    file_name="meta_audit_highest_confidence.csv",
                    mime="text/csv",
                    key="meta_audit_high_conf_download",
                )
            with rank_c:
                st.dataframe(audit_report.highest_risk_candidates, width="stretch")
                st.download_button(
                    "📥 Export Highest Risk CSV",
                    data=audit_report.highest_risk_candidates.to_csv(index=False).encode("utf-8"),
                    file_name="meta_audit_highest_risk.csv",
                    mime="text/csv",
                    key="meta_audit_high_risk_download",
                )

        with tab_modes:
            st.markdown("### Decision Counts by Mode")
            st.dataframe(audit_report.mode_comparison, width="stretch")
            st.download_button(
                "📥 Export Mode Comparison CSV",
                data=audit_report.mode_comparison.to_csv(index=False).encode("utf-8"),
                file_name="meta_audit_mode_comparison.csv",
                mime="text/csv",
                key="meta_audit_modes_download",
            )

        with tab_thresholds:
            st.markdown("### Threshold Configuration")
            st.dataframe(audit_report.threshold_config, width="stretch")
            st.download_button(
                "📥 Export Threshold Config CSV",
                data=audit_report.threshold_config.to_csv(index=False).encode("utf-8"),
                file_name="meta_audit_threshold_config.csv",
                mime="text/csv",
                key="meta_audit_thresholds_download",
            )


# PAGE: META RELIABILITY GRADING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

elif page == "🏷️ Meta Reliability Grading":
    st.markdown('<p class="main-header">🏷️ Meta Reliability Grading</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 8B grades research reliability only. MetaDecision remains the action gate; "
        "A-grade candidates are still not production-ready."
    )

    latest_audit_report = st.session_state.get("meta_decision_audit_report")
    latest_audit_table = None
    if latest_audit_report is not None and getattr(latest_audit_report, "audit_table", None) is not None:
        latest_audit_table = latest_audit_report.audit_table

    latest_meta_report = st.session_state.get("meta_signal_report")
    latest_meta_table = None
    if latest_meta_report is not None and getattr(latest_meta_report, "decision_table", None) is not None:
        latest_meta_table = latest_meta_report.decision_table

    uploaded_grading_csv = st.file_uploader(
        "Upload Phase 8A audit CSV or Phase 8 meta decision CSV",
        type=["csv"],
        key="meta_grading_upload",
    )

    grading_sources = []
    if latest_audit_table is not None and not latest_audit_table.empty:
        grading_sources.append("Latest Phase 8A audit result")
    if latest_meta_table is not None and not latest_meta_table.empty:
        grading_sources.append("Latest Phase 8 meta result")
    if uploaded_grading_csv is not None:
        grading_sources.append("Uploaded CSV")

    grading_col_a, grading_col_b = st.columns([1, 2])
    with grading_col_a:
        grading_mode = st.selectbox(
            "Grading mode",
            ["Conservative", "Balanced", "Aggressive Research"],
            index=0,
            key="meta_grading_mode",
        )
    with grading_col_b:
        if grading_mode == "Conservative":
            st.caption("Strict grading: prioritizes evidence quality and downgrades borderline candidates.")
        elif grading_mode == "Balanced":
            st.caption("Balanced grading: surfaces research quality without loosening action gates.")
        else:
            st.caption("Aggressive Research: raises research attention only; it never implies production readiness.")

    if grading_sources:
        grading_source = st.radio("Input source", grading_sources, horizontal=True, key="meta_grading_source")
    else:
        grading_source = None
        st.info("Run Phase 8A audit, run Phase 8 meta signal, or upload a CSV.")

    run_grading = st.button("🚀 Run Reliability Grading", type="primary")

    if run_grading:
        if grading_source is None:
            st.error("Provide a Phase 8A audit table or Phase 8 meta decision table.")
            st.stop()
        try:
            if grading_source == "Latest Phase 8A audit result":
                grading_input = latest_audit_table.copy()
            elif grading_source == "Latest Phase 8 meta result":
                grading_input = latest_meta_table.copy()
            else:
                uploaded_grading_csv.seek(0)
                grading_input = pd.read_csv(uploaded_grading_csv)
        except Exception as exc:
            st.error(f"Could not read grading input: {exc}")
            st.stop()

        if grading_input is None or grading_input.empty:
            st.error("Reliability grading input is empty.")
            st.stop()

        with st.spinner("Running meta reliability grading..."):
            try:
                grading_report = run_meta_score_calibration(grading_input, grading_mode=grading_mode)
                st.session_state.meta_reliability_grading_report = grading_report
                st.session_state.meta_reliability_grading_settings = grading_report.settings
            except Exception as exc:
                st.error(f"Meta reliability grading failed: {exc}")
                st.stop()

    grading_report = st.session_state.meta_reliability_grading_report
    if grading_report is None:
        st.info("Run reliability grading to rank research quality without changing MetaDecision gates.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('meta_reliability_grading_settings') or {}}")

        if grading_report.warnings:
            for warning in grading_report.warnings:
                st.warning(warning)

        st.markdown("### Reliability Grading Table")
        st.dataframe(grading_report.grading_table, width="stretch")
        st.download_button(
            "📥 Export Reliability Grading CSV",
            data=grading_report.grading_table.to_csv(index=False).encode("utf-8"),
            file_name="meta_reliability_grading.csv",
            mime="text/csv",
            key="meta_grading_table_download",
        )

        tab_counts, tab_research, tab_defensive, tab_archive, tab_actions, tab_components = st.tabs(
            ["Grade Counts", "Top A/B/C", "Defensive Watch", "Avoid / Archive", "Next Actions", "Score Components"]
        )

        with tab_counts:
            st.markdown("### Grade Counts")
            st.dataframe(grading_report.grade_counts, width="stretch")
            st.download_button(
                "📥 Export Grade Counts CSV",
                data=grading_report.grade_counts.to_csv(index=False).encode("utf-8"),
                file_name="meta_reliability_grade_counts.csv",
                mime="text/csv",
                key="meta_grading_counts_download",
            )

        with tab_research:
            st.markdown("### Top A/B/C Research Candidates")
            st.dataframe(grading_report.top_research_candidates, width="stretch")
            st.download_button(
                "📥 Export Top Research CSV",
                data=grading_report.top_research_candidates.to_csv(index=False).encode("utf-8"),
                file_name="meta_reliability_top_research.csv",
                mime="text/csv",
                key="meta_grading_research_download",
            )

        with tab_defensive:
            st.markdown("### Defensive Watch List")
            st.dataframe(grading_report.defensive_watchlist, width="stretch")
            st.download_button(
                "📥 Export Defensive Watch CSV",
                data=grading_report.defensive_watchlist.to_csv(index=False).encode("utf-8"),
                file_name="meta_reliability_defensive_watch.csv",
                mime="text/csv",
                key="meta_grading_defensive_download",
            )

        with tab_archive:
            st.markdown("### Avoid / Archive List")
            st.dataframe(grading_report.avoid_archive_list, width="stretch")
            st.download_button(
                "📥 Export Avoid Archive CSV",
                data=grading_report.avoid_archive_list.to_csv(index=False).encode("utf-8"),
                file_name="meta_reliability_avoid_archive.csv",
                mime="text/csv",
                key="meta_grading_archive_download",
            )

        with tab_actions:
            st.markdown("### Top Next Actions")
            st.dataframe(grading_report.next_action_summary, width="stretch")
            st.download_button(
                "📥 Export Next Actions CSV",
                data=grading_report.next_action_summary.to_csv(index=False).encode("utf-8"),
                file_name="meta_reliability_next_actions.csv",
                mime="text/csv",
                key="meta_grading_actions_download",
            )

        with tab_components:
            st.markdown("### Score Component Breakdown")
            st.dataframe(grading_report.score_components, width="stretch")
            st.download_button(
                "📥 Export Score Components CSV",
                data=grading_report.score_components.to_csv(index=False).encode("utf-8"),
                file_name="meta_reliability_score_components.csv",
                mime="text/csv",
                key="meta_grading_components_download",
            )


# PAGE: EVIDENCE EXPANSION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

elif page == "🧪 Evidence Expansion":
    st.markdown('<p class="main-header">🧪 Evidence Expansion</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 8C is research-validation only. It stress-tests Phase 8B candidates across predeclared "
        "walk-forward configurations and may conclude that no candidate improved."
    )

    latest_grading_report = st.session_state.get("meta_reliability_grading_report")
    latest_grading_table = None
    if latest_grading_report is not None and getattr(latest_grading_report, "grading_table", None) is not None:
        latest_grading_table = latest_grading_report.grading_table

    uploaded_evidence_csv = st.file_uploader(
        "Upload Phase 8B reliability grading CSV",
        type=["csv"],
        key="evidence_expansion_upload",
    )

    evidence_sources = []
    if latest_grading_table is not None and not latest_grading_table.empty:
        evidence_sources.append("Latest Phase 8B grading result")
    if uploaded_evidence_csv is not None:
        evidence_sources.append("Uploaded Phase 8B CSV")

    if evidence_sources:
        evidence_source = st.radio("Candidate source", evidence_sources, horizontal=True, key="evidence_expansion_source")
    else:
        evidence_source = None
        st.info("Run Phase 8B Meta Reliability Grading first, or upload a Phase 8B CSV.")

    ev_col_a, ev_col_b, ev_col_c, ev_col_d = st.columns(4)
    with ev_col_a:
        evidence_filter = st.selectbox(
            "Candidate selector",
            ["all", "only c/d candidates", "specific asset/horizon"],
            index=0,
            key="evidence_expansion_filter",
        )
    with ev_col_b:
        evidence_depth = st.selectbox("Model depth", ["core", "fast"], index=0, key="evidence_expansion_depth")
    with ev_col_c:
        evidence_phase5 = st.checkbox("Use Phase 5", value=True, key="evidence_expansion_phase5")
    with ev_col_d:
        evidence_signal_mode = st.selectbox("Signal mode", ["long_only", "long_short", "avoid_only"], index=0, key="evidence_expansion_signal_mode")

    selected_ev_assets = None
    selected_ev_horizons = None
    if evidence_filter == "specific asset/horizon":
        spec_a, spec_b = st.columns(2)
        with spec_a:
            selected_ev_assets = st.multiselect(
                "Specific assets",
                get_asset_names(),
                default=get_asset_names(),
                key="evidence_expansion_specific_assets",
            )
        with spec_b:
            horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
            selected_horizon_labels = st.multiselect(
                "Specific horizons",
                horizon_labels,
                default=horizon_labels,
                key="evidence_expansion_specific_horizons",
            )
            selected_ev_horizons = [int(str(label).replace("D", "")) for label in selected_horizon_labels]

    cfg_a, cfg_b, cfg_c = st.columns(3)
    with cfg_a:
        ev_validation_windows = st.multiselect(
            "Validation windows",
            [120, 180, 252],
            default=[120, 180, 252],
            key="evidence_expansion_validation_windows",
        )
        ev_step_sizes = st.multiselect(
            "Step sizes",
            [30, 60],
            default=[30, 60],
            key="evidence_expansion_step_sizes",
        )
    with cfg_b:
        ev_test_windows = st.multiselect(
            "Test windows",
            [60, 90, 126],
            default=[60, 90, 126],
            key="evidence_expansion_test_windows",
        )
        ev_window_modes = st.multiselect(
            "Window modes",
            ["rolling", "expanding"],
            default=["rolling", "expanding"],
            key="evidence_expansion_window_modes",
        )
    with cfg_c:
        ev_cost_labels = st.multiselect(
            "Transaction costs",
            ["0.05%", "0.10%", "0.20%"],
            default=["0.05%", "0.10%", "0.20%"],
            key="evidence_expansion_costs",
        )
        ev_cost_map = {"0.05%": 0.0005, "0.10%": 0.001, "0.20%": 0.002}
        ev_costs = [ev_cost_map[label] for label in ev_cost_labels]
        ev_min_valid = st.number_input(
            "Min valid configurations",
            min_value=1,
            max_value=108,
            value=6,
            step=1,
            key="evidence_expansion_min_valid",
        )

    run_evidence = st.button("🚀 Run Evidence Expansion", type="primary")

    if run_evidence:
        if evidence_source is None:
            st.error("Provide a Phase 8B reliability grading source.")
            st.stop()
        if not ev_validation_windows or not ev_test_windows or not ev_step_sizes or not ev_costs or not ev_window_modes:
            st.error("Select at least one value for every configuration dimension.")
            st.stop()
        try:
            if evidence_source == "Latest Phase 8B grading result":
                evidence_input = latest_grading_table.copy()
            else:
                uploaded_evidence_csv.seek(0)
                evidence_input = pd.read_csv(uploaded_evidence_csv)
        except Exception as exc:
            st.error(f"Could not read Phase 8B grading data: {exc}")
            st.stop()

        if evidence_input is None or evidence_input.empty:
            st.error("Phase 8B grading data is empty.")
            st.stop()

        progress = st.progress(0, text="Preparing evidence expansion...")

        def _evidence_progress(done, total, msg):
            progress.progress(int(done / max(total, 1) * 100), text=msg)

        with st.spinner("Running evidence expansion across walk-forward configurations..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                evidence_report = run_evidence_expansion(
                    grading_table=evidence_input,
                    raw_df=raw_df,
                    candidate_filter=evidence_filter,
                    selected_assets=selected_ev_assets,
                    selected_horizons=selected_ev_horizons,
                    validation_windows=ev_validation_windows,
                    test_windows=ev_test_windows,
                    step_sizes=ev_step_sizes,
                    transaction_costs=ev_costs,
                    window_modes=ev_window_modes,
                    model_depth=evidence_depth,
                    use_phase5_features=evidence_phase5,
                    signal_mode=evidence_signal_mode,
                    min_valid_configurations=int(ev_min_valid),
                    progress_callback=_evidence_progress,
                )
                st.session_state.evidence_expansion_report = evidence_report
                st.session_state.evidence_expansion_settings = evidence_report.settings
                progress.progress(100, text="Evidence expansion complete.")
            except Exception as exc:
                st.error(f"Evidence expansion failed: {exc}")
                st.stop()

    evidence_report = st.session_state.evidence_expansion_report
    if evidence_report is None:
        st.info("Run evidence expansion to stress-test Phase 8B candidates before considering grade changes.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('evidence_expansion_settings') or {}}")

        st.markdown("### Overall Summary")
        st.dataframe(evidence_report.overall_summary, width="stretch")

        tab_promote, tab_full, tab_cost, tab_warn, tab_config, tab_detail = st.tabs(
            [
                "Promotion / Demotion",
                "Full Evidence",
                "Cost Sensitivity",
                "Warnings",
                "Configuration Summary",
                "Candidate Detail",
            ]
        )

        with tab_promote:
            st.markdown("### Promotion / Demotion Recommendations")
            st.dataframe(evidence_report.promotion_recommendations, width="stretch")
            st.download_button(
                "📥 Export Promotion Recommendations CSV",
                data=evidence_report.promotion_recommendations.to_csv(index=False).encode("utf-8"),
                file_name="evidence_expansion_promotion_recommendations.csv",
                mime="text/csv",
                key="evidence_promotion_download",
            )
            st.markdown("### Candidate Robustness Summary")
            st.dataframe(evidence_report.robustness_summary, width="stretch")
            st.download_button(
                "📥 Export Robustness Summary CSV",
                data=evidence_report.robustness_summary.to_csv(index=False).encode("utf-8"),
                file_name="evidence_expansion_robustness_summary.csv",
                mime="text/csv",
                key="evidence_robustness_download",
            )

        with tab_full:
            st.markdown("### Full Expanded Evidence Table")
            st.dataframe(evidence_report.full_evidence_table, width="stretch")
            st.download_button(
                "📥 Export Full Evidence CSV",
                data=evidence_report.full_evidence_table.to_csv(index=False).encode("utf-8"),
                file_name="evidence_expansion_full.csv",
                mime="text/csv",
                key="evidence_full_download",
            )

        with tab_cost:
            st.markdown("### Cost Sensitivity Summary")
            st.dataframe(evidence_report.cost_sensitivity_summary, width="stretch")
            st.download_button(
                "📥 Export Cost Sensitivity CSV",
                data=evidence_report.cost_sensitivity_summary.to_csv(index=False).encode("utf-8"),
                file_name="evidence_expansion_cost_sensitivity.csv",
                mime="text/csv",
                key="evidence_cost_download",
            )

        with tab_warn:
            st.markdown("### Warning Table")
            st.dataframe(evidence_report.warning_table, width="stretch")
            st.download_button(
                "📥 Export Warning Table CSV",
                data=evidence_report.warning_table.to_csv(index=False).encode("utf-8"),
                file_name="evidence_expansion_warnings.csv",
                mime="text/csv",
                key="evidence_warning_download",
            )

        with tab_config:
            st.markdown("### Configuration-Level Failure / Success Table")
            st.dataframe(evidence_report.configuration_summary, width="stretch")
            st.download_button(
                "📥 Export Configuration Summary CSV",
                data=evidence_report.configuration_summary.to_csv(index=False).encode("utf-8"),
                file_name="evidence_expansion_configuration_summary.csv",
                mime="text/csv",
                key="evidence_config_download",
            )

        with tab_detail:
            st.markdown("### Candidate Detail View")
            if not evidence_report.robustness_summary.empty:
                candidate_labels = [
                    f"{row.get('Asset', '')} {int(row.get('Horizon', 0))}D"
                    for _, row in evidence_report.robustness_summary.iterrows()
                ]
                selected_candidate = st.selectbox("Candidate", candidate_labels, key="evidence_detail_candidate")
                selected_asset_label, selected_horizon_label = selected_candidate.rsplit(" ", 1)
                selected_horizon_value = int(selected_horizon_label.replace("D", ""))
                detail = evidence_report.full_evidence_table[
                    evidence_report.full_evidence_table["Asset"].astype(str).eq(selected_asset_label)
                    & evidence_report.full_evidence_table["Horizon"].astype(int).eq(selected_horizon_value)
                ]
                st.dataframe(detail, width="stretch")
            else:
                st.info("No candidate detail rows available.")


# PAGE: EVIDENCE QUALITY DIAGNOSTICS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

elif page == "🔎 Evidence Quality Diagnostics":
    st.markdown('<p class="main-header">🔎 Evidence Quality Diagnostics</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 8D is diagnostics only. It explains weak evidence, signal coverage, benchmark dependence, "
        "cost fragility, window instability, and next research actions without promoting candidates."
    )

    latest_evidence_report = st.session_state.get("evidence_expansion_report")
    latest_full = latest_robustness = latest_promotion = None
    if latest_evidence_report is not None:
        latest_full = getattr(latest_evidence_report, "full_evidence_table", None)
        latest_robustness = getattr(latest_evidence_report, "robustness_summary", None)
        latest_promotion = getattr(latest_evidence_report, "promotion_recommendations", None)

    latest_grading_report = st.session_state.get("meta_reliability_grading_report")
    latest_grading = None
    if latest_grading_report is not None:
        latest_grading = getattr(latest_grading_report, "grading_table", None)

    diag_sources = []
    if latest_full is not None and not latest_full.empty:
        diag_sources.append("Latest Phase 8C session result")
    diag_sources.append("Uploaded Phase 8C CSVs")
    diag_source = st.radio("Diagnostics source", diag_sources, horizontal=True, key="evidence_quality_source")

    uploaded_full = uploaded_robust = uploaded_promo = uploaded_grade = None
    if diag_source == "Uploaded Phase 8C CSVs":
        upload_a, upload_b = st.columns(2)
        with upload_a:
            uploaded_full = st.file_uploader("Upload full expanded evidence CSV", type=["csv"], key="evidence_quality_full_upload")
            uploaded_robust = st.file_uploader("Upload robustness summary CSV", type=["csv"], key="evidence_quality_robust_upload")
        with upload_b:
            uploaded_promo = st.file_uploader("Upload promotion/demotion CSV", type=["csv"], key="evidence_quality_promo_upload")
            uploaded_grade = st.file_uploader("Optional Phase 8B grading CSV", type=["csv"], key="evidence_quality_grade_upload")

    diag_filter_col_a, diag_filter_col_b = st.columns(2)
    with diag_filter_col_a:
        diagnostics_candidate_mode = st.selectbox(
            "Candidate selector",
            ["all", "specific asset/horizon"],
            index=0,
            key="evidence_quality_candidate_selector",
        )
    selected_diag_assets = selected_diag_horizons = None
    with diag_filter_col_b:
        st.caption("All failed and weak candidates remain visible unless a specific asset/horizon filter is selected.")

    if diagnostics_candidate_mode == "specific asset/horizon":
        spec_a, spec_b = st.columns(2)
        with spec_a:
            selected_diag_assets = st.multiselect(
                "Specific assets",
                get_asset_names(),
                default=get_asset_names(),
                key="evidence_quality_assets",
            )
        with spec_b:
            horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
            selected_diag_horizon_labels = st.multiselect(
                "Specific horizons",
                horizon_labels,
                default=horizon_labels,
                key="evidence_quality_horizons",
            )
            selected_diag_horizons = [int(str(label).replace("D", "")) for label in selected_diag_horizon_labels]

    run_quality = st.button("🚀 Run Evidence Quality Diagnostics", type="primary")

    if run_quality:
        try:
            if diag_source == "Latest Phase 8C session result":
                full_for_diag = latest_full.copy()
                robust_for_diag = latest_robustness.copy() if latest_robustness is not None else None
                promo_for_diag = latest_promotion.copy() if latest_promotion is not None else None
                grading_for_diag = latest_grading.copy() if latest_grading is not None else None
            else:
                if uploaded_full is None:
                    st.error("Upload the Phase 8C full expanded evidence CSV.")
                    st.stop()
                uploaded_full.seek(0)
                full_for_diag = pd.read_csv(uploaded_full)
                robust_for_diag = None
                promo_for_diag = None
                grading_for_diag = None
                if uploaded_robust is not None:
                    uploaded_robust.seek(0)
                    robust_for_diag = pd.read_csv(uploaded_robust)
                if uploaded_promo is not None:
                    uploaded_promo.seek(0)
                    promo_for_diag = pd.read_csv(uploaded_promo)
                if uploaded_grade is not None:
                    uploaded_grade.seek(0)
                    grading_for_diag = pd.read_csv(uploaded_grade)
        except Exception as exc:
            st.error(f"Could not read diagnostics input: {exc}")
            st.stop()

        if diagnostics_candidate_mode == "specific asset/horizon":
            if selected_diag_assets:
                full_for_diag = full_for_diag[full_for_diag["Asset"].astype(str).isin(set(selected_diag_assets))]
                if robust_for_diag is not None and "Asset" in robust_for_diag.columns:
                    robust_for_diag = robust_for_diag[robust_for_diag["Asset"].astype(str).isin(set(selected_diag_assets))]
                if promo_for_diag is not None and "Asset" in promo_for_diag.columns:
                    promo_for_diag = promo_for_diag[promo_for_diag["Asset"].astype(str).isin(set(selected_diag_assets))]
                if grading_for_diag is not None and "Asset" in grading_for_diag.columns:
                    grading_for_diag = grading_for_diag[grading_for_diag["Asset"].astype(str).isin(set(selected_diag_assets))]
            if selected_diag_horizons:
                horizons_set = set(int(h) for h in selected_diag_horizons)
                full_for_diag = full_for_diag[pd.to_numeric(full_for_diag["Horizon"], errors="coerce").fillna(0).astype(int).isin(horizons_set)]
                if robust_for_diag is not None and "Horizon" in robust_for_diag.columns:
                    robust_for_diag = robust_for_diag[pd.to_numeric(robust_for_diag["Horizon"], errors="coerce").fillna(0).astype(int).isin(horizons_set)]
                if promo_for_diag is not None and "Horizon" in promo_for_diag.columns:
                    promo_for_diag = promo_for_diag[pd.to_numeric(promo_for_diag["Horizon"], errors="coerce").fillna(0).astype(int).isin(horizons_set)]
                if grading_for_diag is not None and "Horizon" in grading_for_diag.columns:
                    grading_for_diag = grading_for_diag[pd.to_numeric(grading_for_diag["Horizon"], errors="coerce").fillna(0).astype(int).isin(horizons_set)]

        with st.spinner("Running evidence quality diagnostics..."):
            try:
                quality_report = run_evidence_quality_diagnostics(
                    full_evidence_table=full_for_diag,
                    robustness_summary=robust_for_diag,
                    promotion_recommendations=promo_for_diag,
                    grading_table=grading_for_diag,
                )
                st.session_state.evidence_quality_diagnostics_report = quality_report
                st.session_state.evidence_quality_diagnostics_settings = quality_report.settings
            except Exception as exc:
                st.error(f"Evidence quality diagnostics failed: {exc}")
                st.stop()

    quality_report = st.session_state.evidence_quality_diagnostics_report
    if quality_report is None:
        st.info("Run diagnostics to explain why evidence stayed weak or failed.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('evidence_quality_diagnostics_settings') or {}}")
        st.markdown("### Overall Evidence Quality Summary")
        st.dataframe(quality_report.overall_summary, width="stretch")
        st.download_button(
            "📥 Export Overall Summary CSV",
            data=quality_report.overall_summary.to_csv(index=False).encode("utf-8"),
            file_name="evidence_quality_overall_summary.csv",
            mime="text/csv",
            key="evidence_quality_overall_download",
        )

        tabs = st.tabs(
            [
                "Failure Reasons",
                "Signal Coverage",
                "Benchmark Dependency",
                "Threshold / Cooldown",
                "Horizon Quality",
                "Regime Concentration",
                "Probability Warnings",
                "Next Actions",
                "Quality Table",
            ]
        )
        table_specs = [
            ("Candidate Failure Reason Table", quality_report.candidate_failure_reason_table, "evidence_quality_failure_reasons.csv"),
            ("Signal Coverage Table", quality_report.signal_coverage_table, "evidence_quality_signal_coverage.csv"),
            ("Benchmark Dependency Table", quality_report.benchmark_dependency_table, "evidence_quality_benchmark_dependency.csv"),
            ("Threshold/Cooldown Sensitivity Table", quality_report.threshold_cooldown_sensitivity_table, "evidence_quality_threshold_cooldown.csv"),
            ("Horizon Quality Table", quality_report.horizon_quality_table, "evidence_quality_horizon.csv"),
            ("Regime Concentration Table", quality_report.regime_concentration_table, "evidence_quality_regime.csv"),
            ("Probability Quality Warning Table", quality_report.probability_quality_warning_table, "evidence_quality_probability_warnings.csv"),
            ("Next Research Action Table", quality_report.next_research_action_table, "evidence_quality_next_actions.csv"),
            ("Evidence Quality Diagnostics Table", quality_report.evidence_quality_table, "evidence_quality_diagnostics.csv"),
        ]
        for tab, (title, table, filename) in zip(tabs, table_specs):
            with tab:
                st.markdown(f"### {title}")
                st.dataframe(table, width="stretch")
                st.download_button(
                    f"📥 Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )


# PAGE: SIGNAL POLICY SENSITIVITY
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

elif page == "📈 Signal Policy Sensitivity":
    st.markdown('<p class="main-header">📈 Signal Policy Sensitivity</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 8E is policy sensitivity diagnostics only. It asks whether coverage can recover without "
        "destroying benchmark-relative edge, drawdown control, or cost robustness. It does not promote grades."
    )

    latest_quality_report = st.session_state.get("evidence_quality_diagnostics_report")
    latest_evidence_report = st.session_state.get("evidence_expansion_report")
    latest_grading_report = st.session_state.get("meta_reliability_grading_report")
    latest_diag_table = getattr(latest_quality_report, "evidence_quality_table", None) if latest_quality_report is not None else None
    latest_full_table = getattr(latest_evidence_report, "full_evidence_table", None) if latest_evidence_report is not None else None
    latest_grade_table = getattr(latest_grading_report, "grading_table", None) if latest_grading_report is not None else None

    policy_sources = []
    if latest_diag_table is not None and not latest_diag_table.empty and latest_full_table is not None and not latest_full_table.empty:
        policy_sources.append("Latest Phase 8D/8C session result")
    policy_sources.append("Uploaded CSVs")
    policy_source = st.radio("Policy sensitivity source", policy_sources, horizontal=True, key="policy_sensitivity_source")

    uploaded_diag = uploaded_full_policy = uploaded_grade_policy = None
    if policy_source == "Uploaded CSVs":
        upload_a, upload_b, upload_c = st.columns(3)
        with upload_a:
            uploaded_diag = st.file_uploader("Upload Phase 8D diagnostics CSV", type=["csv"], key="policy_diag_upload")
        with upload_b:
            uploaded_full_policy = st.file_uploader("Upload Phase 8C full evidence CSV", type=["csv"], key="policy_full_upload")
        with upload_c:
            uploaded_grade_policy = st.file_uploader("Optional Phase 8B grading CSV", type=["csv"], key="policy_grade_upload")

    pol_col_a, pol_col_b = st.columns(2)
    with pol_col_a:
        policy_candidate_filter = st.selectbox(
            "Candidate selector",
            ["coverage_focus", "all", "specific asset/horizon"],
            index=0,
            key="policy_candidate_filter",
        )
    with pol_col_b:
        st.caption("Default focus targets insufficient coverage, C/D grades, or low evidence quality.")

    selected_policy_assets = selected_policy_horizons = None
    if policy_candidate_filter == "specific asset/horizon":
        spec_a, spec_b = st.columns(2)
        with spec_a:
            selected_policy_assets = st.multiselect("Specific assets", get_asset_names(), default=get_asset_names(), key="policy_specific_assets")
        with spec_b:
            policy_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
            selected_policy_horizon_labels = st.multiselect("Specific horizons", policy_horizon_labels, default=policy_horizon_labels, key="policy_specific_horizons")
            selected_policy_horizons = [int(str(label).replace("D", "")) for label in selected_policy_horizon_labels]

    scan_a, scan_b, scan_c, scan_d = st.columns(4)
    with scan_a:
        policy_thresholds = st.multiselect("Thresholds", [0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65], default=[0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65], format_func=lambda x: f"{x:.3f}", key="policy_thresholds")
    with scan_b:
        policy_cooldowns = st.multiselect("Cooldowns", [0, 1, 2, 3, 5], default=[0, 1, 2, 3, 5], key="policy_cooldowns")
    with scan_c:
        policy_min_probs = st.multiselect("Min probability", [0.50, 0.525, 0.55, 0.575], default=[0.50, 0.525, 0.55, 0.575], format_func=lambda x: f"{x:.3f}", key="policy_min_probs")
    with scan_d:
        policy_max_probs = st.multiselect("Max probability", [0.95, 0.975, 1.00], default=[0.95, 0.975, 1.00], format_func=lambda x: f"{x:.3f}", key="policy_max_probs")
    policy_scan_horizons = st.multiselect("Horizon sensitivity", [1, 5, 10, 20, 30], default=[1, 5, 10, 20, 30], format_func=lambda x: f"{x}D", key="policy_horizon_sensitivity")

    run_policy = st.button("🚀 Run Policy Sensitivity", type="primary")

    if run_policy:
        if not policy_thresholds or not policy_cooldowns or not policy_min_probs or not policy_max_probs or not policy_scan_horizons:
            st.error("Select at least one value for every policy scan dimension.")
            st.stop()
        try:
            if policy_source == "Latest Phase 8D/8C session result":
                diagnostics_input = latest_diag_table.copy()
                full_input = latest_full_table.copy()
                grading_input = latest_grade_table.copy() if latest_grade_table is not None else None
            else:
                if uploaded_diag is None or uploaded_full_policy is None:
                    st.error("Upload both Phase 8D diagnostics and Phase 8C full evidence CSVs.")
                    st.stop()
                uploaded_diag.seek(0)
                uploaded_full_policy.seek(0)
                diagnostics_input = pd.read_csv(uploaded_diag)
                full_input = pd.read_csv(uploaded_full_policy)
                grading_input = None
                if uploaded_grade_policy is not None:
                    uploaded_grade_policy.seek(0)
                    grading_input = pd.read_csv(uploaded_grade_policy)
        except Exception as exc:
            st.error(f"Could not read policy sensitivity inputs: {exc}")
            st.stop()

        with st.spinner("Running signal policy sensitivity diagnostics..."):
            try:
                policy_report = run_signal_policy_sensitivity(
                    diagnostics_table=diagnostics_input,
                    full_evidence_table=full_input,
                    grading_table=grading_input,
                    candidate_filter=policy_candidate_filter,
                    selected_assets=selected_policy_assets,
                    selected_horizons=selected_policy_horizons,
                    thresholds=policy_thresholds,
                    cooldowns=policy_cooldowns,
                    min_probabilities=policy_min_probs,
                    max_probabilities=policy_max_probs,
                    horizons=policy_scan_horizons,
                )
                st.session_state.signal_policy_sensitivity_report = policy_report
                st.session_state.signal_policy_sensitivity_settings = policy_report.settings
            except Exception as exc:
                st.error(f"Signal policy sensitivity failed: {exc}")
                st.stop()

    policy_report = st.session_state.signal_policy_sensitivity_report
    if policy_report is None:
        st.info("Run policy sensitivity to test whether coverage can recover without destroying edge.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('signal_policy_sensitivity_settings') or {}}")
        st.markdown("### Overall Policy Sensitivity Summary")
        st.dataframe(policy_report.overall_summary, width="stretch")
        st.download_button("📥 Export Overall Summary CSV", data=policy_report.overall_summary.to_csv(index=False).encode("utf-8"), file_name="signal_policy_overall_summary.csv", mime="text/csv", key="policy_overall_download")

        tabs = st.tabs(["Recommendations", "Coverage Recovery", "Coverage vs Edge", "Thresholds", "Cooldowns", "Probability Bands", "Horizons", "Warnings", "Next Actions", "Full Table"])
        policy_tables = [
            ("Candidate Recommendation Table", policy_report.candidate_recommendation_table, "signal_policy_recommendations.csv"),
            ("Coverage Recovery Summary", policy_report.coverage_recovery_summary, "signal_policy_coverage_recovery.csv"),
            ("Coverage-vs-Edge Frontier", policy_report.coverage_edge_frontier_table, "signal_policy_frontier.csv"),
            ("Threshold Sensitivity", policy_report.threshold_sensitivity_table, "signal_policy_thresholds.csv"),
            ("Cooldown Sensitivity", policy_report.cooldown_sensitivity_table, "signal_policy_cooldowns.csv"),
            ("Probability Band Sensitivity", policy_report.probability_band_sensitivity_table, "signal_policy_probability_bands.csv"),
            ("Horizon Sensitivity", policy_report.horizon_sensitivity_table, "signal_policy_horizons.csv"),
            ("Warning Table", policy_report.warning_table, "signal_policy_warnings.csv"),
            ("Next Research Action Table", policy_report.next_research_action_table, "signal_policy_next_actions.csv"),
            ("Full Policy Sensitivity Table", policy_report.full_policy_sensitivity_table, "signal_policy_full.csv"),
        ]
        for tab, (title, table, filename) in zip(tabs, policy_tables):
            with tab:
                st.markdown(f"### {title}")
                st.dataframe(table, width="stretch")
                st.download_button(f"📥 Export {title} CSV", data=table.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv", key=f"{filename}_download")


# PAGE: PROBABILITY CALIBRATION
# ═══════════════════════════════════════════════════════════════════════

elif page == "🎯 Probability Calibration":
    st.markdown('<p class="main-header">🎯 Probability Calibration</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 8F is research-validation only. It checks whether probability/confidence values are calibrated "
        "and useful for filtering risk. It does not promote candidates, approve live trading, or tune on locked-test data."
    )

    latest_policy_report = st.session_state.get("signal_policy_sensitivity_report")
    latest_evidence_report = st.session_state.get("evidence_expansion_report")
    latest_quality_report = st.session_state.get("evidence_quality_diagnostics_report")
    latest_policy_rec = getattr(latest_policy_report, "candidate_recommendation_table", None) if latest_policy_report is not None else None
    latest_frontier = getattr(latest_policy_report, "coverage_edge_frontier_table", None) if latest_policy_report is not None else None
    latest_full_evidence = getattr(latest_evidence_report, "full_evidence_table", None) if latest_evidence_report is not None else None
    latest_quality_table = getattr(latest_quality_report, "evidence_quality_table", None) if latest_quality_report is not None else None

    calibration_sources = []
    if latest_frontier is not None and not latest_frontier.empty and latest_full_evidence is not None and not latest_full_evidence.empty:
        calibration_sources.append("Latest Phase 8E/8C session result")
    calibration_sources.append("Uploaded CSVs")
    calibration_source = st.radio("Calibration source", calibration_sources, horizontal=True, key="probability_calibration_source")

    uploaded_raw_trade = uploaded_policy_rec = uploaded_frontier = uploaded_full = uploaded_quality = None
    if calibration_source == "Uploaded CSVs":
        upload_a, upload_b, upload_c = st.columns(3)
        with upload_a:
            uploaded_raw_trade = st.file_uploader("Upload raw trade log CSV", type=["csv"], key="prob_cal_raw_trade_upload")
            uploaded_policy_rec = st.file_uploader("Upload Phase 8E candidate recommendation CSV", type=["csv"], key="prob_cal_rec_upload")
        with upload_b:
            uploaded_frontier = st.file_uploader("Upload Phase 8E coverage-vs-edge frontier CSV", type=["csv"], key="prob_cal_frontier_upload")
            uploaded_quality = st.file_uploader("Optional Phase 8D diagnostics CSV", type=["csv"], key="prob_cal_quality_upload")
        with upload_c:
            uploaded_full = st.file_uploader("Upload Phase 8C full expanded evidence CSV", type=["csv"], key="prob_cal_full_upload")
            st.caption("A raw trade log with ProbabilityUp and ActualDirection is used before aggregate proxy files.")

    cal_col_a, cal_col_b = st.columns(2)
    with cal_col_a:
        calibration_candidate_filter = st.selectbox(
            "Candidate selector",
            ["default_focus", "all", "specific asset/horizon"],
            index=0,
            key="prob_cal_candidate_filter",
            help="Default focus highlights Bitcoin 5D and Crude Oil 5D if present, but the analysis supports all configured assets.",
        )
    with cal_col_b:
        st.caption("Failed candidates and failed probability bins remain visible in the outputs.")

    selected_cal_assets = selected_cal_horizons = None
    if calibration_candidate_filter == "specific asset/horizon":
        spec_a, spec_b = st.columns(2)
        with spec_a:
            selected_cal_assets = st.multiselect("Specific assets", get_asset_names(), default=get_asset_names(), key="prob_cal_assets")
        with spec_b:
            cal_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
            selected_cal_horizon_labels = st.multiselect("Specific horizons", cal_horizon_labels, default=cal_horizon_labels, key="prob_cal_horizons")
            selected_cal_horizons = [int(str(label).replace("D", "")) for label in selected_cal_horizon_labels]

    bin_options = [
        "0.50-0.55",
        "0.55-0.60",
        "0.60-0.65",
        "0.65-0.70",
        "0.70-0.75",
        "0.75-0.80",
        "0.80-0.90",
        "0.90-1.00",
    ]
    cal_scan_a, cal_scan_b, cal_scan_c = st.columns(3)
    with cal_scan_a:
        selected_bin_labels = st.multiselect("Probability bins", bin_options, default=bin_options, key="prob_cal_bins")
    with cal_scan_b:
        cal_min_probs = st.multiselect("Min probability filters", [0.50, 0.55, 0.60, 0.65, 0.70, 0.75], default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75], format_func=lambda x: f"{x:.2f}", key="prob_cal_min_probs")
    with cal_scan_c:
        cal_max_probs = st.multiselect("Max probability caps", [0.90, 0.95, 0.975, 1.00], default=[0.90, 0.95, 0.975, 1.00], format_func=lambda x: f"{x:.3f}", key="prob_cal_max_probs")

    run_calibration = st.button("🚀 Run Probability Calibration", type="primary")

    if run_calibration:
        if not selected_bin_labels or not cal_min_probs or not cal_max_probs:
            st.error("Select at least one probability bin and one min/max filter.")
            st.stop()
        probability_bins = []
        for label in selected_bin_labels:
            lower, upper = str(label).split("-")
            probability_bins.append((float(lower), float(upper)))
        try:
            if calibration_source == "Latest Phase 8E/8C session result":
                raw_trade_input = None
                policy_rec_input = latest_policy_rec.copy() if latest_policy_rec is not None else None
                frontier_input = latest_frontier.copy()
                full_input = latest_full_evidence.copy()
                quality_input = latest_quality_table.copy() if latest_quality_table is not None else None
            else:
                if uploaded_raw_trade is None and (uploaded_frontier is None or uploaded_full is None):
                    st.error("Upload a raw trade log CSV, or upload both the Phase 8E frontier CSV and Phase 8C full evidence CSV.")
                    st.stop()
                raw_trade_input = None
                if uploaded_raw_trade is not None:
                    uploaded_raw_trade.seek(0)
                    raw_trade_input = pd.read_csv(uploaded_raw_trade)
                policy_rec_input = None
                if uploaded_policy_rec is not None and raw_trade_input is None:
                    uploaded_policy_rec.seek(0)
                    policy_rec_input = pd.read_csv(uploaded_policy_rec)
                frontier_input = None
                if uploaded_frontier is not None and raw_trade_input is None:
                    uploaded_frontier.seek(0)
                    frontier_input = pd.read_csv(uploaded_frontier)
                full_input = None
                if uploaded_full is not None and raw_trade_input is None:
                    uploaded_full.seek(0)
                    full_input = pd.read_csv(uploaded_full)
                quality_input = None
                if uploaded_quality is not None:
                    uploaded_quality.seek(0)
                    quality_input = pd.read_csv(uploaded_quality)
        except Exception as exc:
            st.error(f"Could not read probability calibration inputs: {exc}")
            st.stop()

        with st.spinner("Running probability calibration diagnostics..."):
            try:
                calibration_report = run_probability_calibration(
                    candidate_recommendation_table=policy_rec_input,
                    raw_trade_log_table=raw_trade_input,
                    coverage_edge_frontier_table=frontier_input,
                    full_evidence_table=full_input,
                    diagnostics_table=quality_input,
                    candidate_filter=calibration_candidate_filter,
                    selected_assets=selected_cal_assets,
                    selected_horizons=selected_cal_horizons,
                    probability_bins=probability_bins,
                    min_probabilities=cal_min_probs,
                    max_probabilities=cal_max_probs,
                )
                st.session_state.probability_calibration_report = calibration_report
                st.session_state.probability_calibration_settings = calibration_report.settings
                saved_artifacts = save_phase_artifacts(
                    "Phase 8F Probability Calibration",
                    {
                        "probability_calibration_summary": calibration_report.calibration_summary_table,
                        "probability_calibration_warnings": calibration_report.warning_table,
                        "probability_calibration_recommendations": calibration_report.candidate_recommendation_table,
                        "probability_calibration_bins": calibration_report.probability_bin_table,
                        "probability_filter_simulation": calibration_report.probability_filter_simulation_table,
                        "confidence_usefulness": calibration_report.confidence_usefulness_table,
                        "calibration_error": calibration_report.calibration_error_table,
                        "high_confidence_failure": calibration_report.high_confidence_failure_table,
                        "next_research_actions": calibration_report.next_research_action_table,
                        "overall_summary": calibration_report.overall_summary,
                    },
                    config=calibration_report.settings,
                    warnings=calibration_report.warning_table["WarningType"].dropna().astype(str).unique().tolist() if not calibration_report.warning_table.empty and "WarningType" in calibration_report.warning_table.columns else [],
                )
                st.session_state.artifact_store_last_save = saved_artifacts
            except Exception as exc:
                st.error(f"Probability calibration failed: {exc}")
                st.stop()

    calibration_report = st.session_state.probability_calibration_report
    if calibration_report is None:
        st.info("Run calibration diagnostics to test whether high probability actually means better signal quality.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('probability_calibration_settings') or {}}")
        if not calibration_report.warning_table.empty and calibration_report.warning_table["WarningType"].astype(str).eq("ProbabilityUnreliable").any():
            st.warning("At least one candidate has unreliable probability evidence. Keep it in diagnostics.")
        st.markdown("### Overall Calibration Summary")
        st.dataframe(calibration_report.overall_summary, width="stretch")
        st.download_button("📥 Export Overall Summary CSV", data=calibration_report.overall_summary.to_csv(index=False).encode("utf-8"), file_name="probability_calibration_overall_summary.csv", mime="text/csv", key="prob_cal_overall_download")

        tabs = st.tabs([
            "Calibration Summary",
            "Recommendations",
            "Probability Bins",
            "Filter Simulation",
            "Confidence Usefulness",
            "Calibration Error",
            "High-Confidence Failures",
            "Warnings",
            "Next Actions",
        ])
        calibration_tables = [
            ("Calibration Summary Table", calibration_report.calibration_summary_table, "probability_calibration_summary.csv"),
            ("Candidate Recommendation Table", calibration_report.candidate_recommendation_table, "probability_calibration_recommendations.csv"),
            ("Probability Bin Table", calibration_report.probability_bin_table, "probability_calibration_bins.csv"),
            ("Probability Filter Simulation", calibration_report.probability_filter_simulation_table, "probability_calibration_filters.csv"),
            ("Confidence Usefulness Table", calibration_report.confidence_usefulness_table, "probability_calibration_confidence_usefulness.csv"),
            ("Calibration Error Table", calibration_report.calibration_error_table, "probability_calibration_errors.csv"),
            ("High-Confidence Failure Table", calibration_report.high_confidence_failure_table, "probability_calibration_high_confidence_failures.csv"),
            ("Warning Table", calibration_report.warning_table, "probability_calibration_warnings.csv"),
            ("Next Research Action Table", calibration_report.next_research_action_table, "probability_calibration_next_actions.csv"),
        ]
        for tab, (title, table, filename) in zip(tabs, calibration_tables):
            with tab:
                st.markdown(f"### {title}")
                st.dataframe(table, width="stretch")
                st.download_button(f"📥 Export {title} CSV", data=table.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv", key=f"{filename}_download")


# PAGE: FORWARD PAPER EVIDENCE
# ═══════════════════════════════════════════════════════════════════════

elif page == "📈 Forward Paper Evidence":
    st.markdown('<p class="main-header">📈 Forward Paper Evidence</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 9 is forward paper evidence collection only. It records paper signals, waits for outcomes to mature, "
        "and evaluates evidence without live trading, production-ready labels, or candidate promotion."
    )

    latest_forward_report = st.session_state.get("forward_paper_evidence_report")
    latest_forward_log = getattr(latest_forward_report, "forward_signal_log", None) if latest_forward_report is not None else None
    latest_true_raw_report = st.session_state.get("true_raw_trade_log_report")
    latest_true_raw_table = getattr(latest_true_raw_report, "true_raw_trade_log", None) if latest_true_raw_report is not None else None
    latest_probability_report = st.session_state.get("probability_calibration_report")
    latest_probability_summary = getattr(latest_probability_report, "calibration_summary_table", None) if latest_probability_report is not None else None
    latest_probability_warnings = getattr(latest_probability_report, "warning_table", None) if latest_probability_report is not None else None

    upload_a, upload_b, upload_c = st.columns(3)
    with upload_a:
        uploaded_forward_log = st.file_uploader("Upload existing forward signal log CSV", type=["csv"], key="forward_log_upload")
        uploaded_predictions = st.file_uploader("Optional latest model predictions CSV", type=["csv"], key="forward_predictions_upload")
    with upload_b:
        uploaded_true_raw = st.file_uploader("Optional true_raw_trade_log.csv", type=["csv"], key="forward_true_raw_upload")
        uploaded_probability_summary = st.file_uploader("Optional probability_calibration_summary.csv", type=["csv"], key="forward_prob_summary_upload")
    with upload_c:
        uploaded_probability_warnings = st.file_uploader("Optional probability_calibration_warnings.csv", type=["csv"], key="forward_prob_warnings_upload")
        st.caption("Uploaded prediction rows are used first; otherwise the app generates direct-model paper rows for every selected asset/horizon.")

    fwd_col_a, fwd_col_b, fwd_col_c, fwd_col_d = st.columns(4)
    with fwd_col_a:
        forward_assets = st.multiselect("Assets", get_asset_names(), default=get_asset_names(), key="forward_assets")
    with fwd_col_b:
        forward_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        selected_forward_horizon_labels = st.multiselect("Horizons", forward_horizon_labels, default=forward_horizon_labels, key="forward_horizons")
        forward_horizons = [int(str(label).replace("D", "")) for label in selected_forward_horizon_labels]
    with fwd_col_c:
        forward_model_depth = st.selectbox("Model depth", ["fast", "core"], index=0, key="forward_model_depth")
        forward_use_phase5 = st.checkbox("Use Phase 5 features", value=True, key="forward_phase5")
    with fwd_col_d:
        forward_as_of = st.date_input("As-of date", value=pd.Timestamp.today().date(), key="forward_as_of")
        forward_min_evidence = st.number_input("Min matured outcomes", min_value=1, max_value=100, value=10, step=1, key="forward_min_evidence")

    forward_action = st.radio(
        "Action",
        ["Generate today's paper signals and update matured outcomes", "Update matured outcomes only"],
        horizontal=True,
        key="forward_action",
    )
    run_forward = st.button("🚀 Run Forward Paper Evidence", type="primary")

    if run_forward:
        if not forward_assets or not forward_horizons:
            st.error("Select at least one asset and one horizon.")
            st.stop()
        try:
            existing_log_input = latest_forward_log.copy() if latest_forward_log is not None and not latest_forward_log.empty else None
            if uploaded_forward_log is not None:
                uploaded_forward_log.seek(0)
                existing_log_input = pd.read_csv(uploaded_forward_log)
            elif existing_log_input is None:
                existing_log_input = load_latest_artifact("Phase 9 Forward Paper Evidence", "forward_signal_log", required=False)

            prediction_input = None
            if uploaded_predictions is not None:
                uploaded_predictions.seek(0)
                prediction_input = pd.read_csv(uploaded_predictions)

            true_raw_input = latest_true_raw_table.copy() if latest_true_raw_table is not None and not latest_true_raw_table.empty else None
            if uploaded_true_raw is not None:
                uploaded_true_raw.seek(0)
                true_raw_input = pd.read_csv(uploaded_true_raw)
            elif true_raw_input is None:
                true_raw_input = load_latest_artifact("Phase 8I True Raw Trade Logs", "true_raw_trade_log", required=False)

            probability_summary_input = latest_probability_summary.copy() if latest_probability_summary is not None and not latest_probability_summary.empty else None
            if uploaded_probability_summary is not None:
                uploaded_probability_summary.seek(0)
                probability_summary_input = pd.read_csv(uploaded_probability_summary)
            elif probability_summary_input is None:
                probability_summary_input = load_latest_artifact("Phase 8F Probability Calibration", "probability_calibration_summary", required=False)

            probability_warnings_input = latest_probability_warnings.copy() if latest_probability_warnings is not None and not latest_probability_warnings.empty else None
            if uploaded_probability_warnings is not None:
                uploaded_probability_warnings.seek(0)
                probability_warnings_input = pd.read_csv(uploaded_probability_warnings)
            elif probability_warnings_input is None:
                probability_warnings_input = load_latest_artifact("Phase 8F Probability Calibration", "probability_calibration_warnings", required=False)
        except Exception as exc:
            st.error(f"Could not read Phase 9 input files: {exc}")
            st.stop()

        with st.spinner("Running forward paper evidence tracker..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                forward_report = run_forward_paper_evidence_tracker(
                    raw_df=raw_df,
                    existing_forward_signal_log=existing_log_input,
                    prediction_table=prediction_input,
                    true_raw_trade_log_table=true_raw_input,
                    probability_calibration_summary=probability_summary_input,
                    probability_calibration_warnings=probability_warnings_input,
                    assets=forward_assets,
                    horizons=forward_horizons,
                    generate_new_signals=forward_action.startswith("Generate"),
                    update_matured_outcomes=True,
                    as_of_date=forward_as_of,
                    model_depth=forward_model_depth,
                    use_phase5_features=forward_use_phase5,
                    min_forward_evidence=int(forward_min_evidence),
                )
                st.session_state.forward_paper_evidence_report = forward_report
                st.session_state.forward_paper_evidence_settings = forward_report.settings
                saved_artifacts = save_phase_artifacts(
                    "Phase 9 Forward Paper Evidence",
                    {
                        "forward_signal_log": forward_report.forward_signal_log,
                        "pending_outcome_table": forward_report.pending_outcome_table,
                        "matured_outcome_table": forward_report.matured_outcome_table,
                        "forward_accuracy_summary": forward_report.forward_accuracy_summary,
                        "forward_probability_calibration_summary": forward_report.forward_probability_calibration_summary,
                        "asset_horizon_forward_coverage": forward_report.asset_horizon_forward_coverage,
                        "forward_warning_table": forward_report.warning_table,
                        "forward_next_research_actions": forward_report.next_research_action_table,
                    },
                    config=forward_report.settings,
                    warnings=forward_report.warning_table["WarningType"].dropna().astype(str).unique().tolist() if not forward_report.warning_table.empty and "WarningType" in forward_report.warning_table.columns else [],
                )
                st.session_state.artifact_store_last_save = saved_artifacts
            except Exception as exc:
                st.error(f"Forward paper evidence tracker failed: {exc}")
                st.stop()

    forward_report = st.session_state.forward_paper_evidence_report
    if forward_report is None:
        st.info("Upload an existing forward log or generate paper signals to begin collecting forward evidence.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('forward_paper_evidence_settings') or {}}")
        if not forward_report.warning_table.empty and forward_report.warning_table["WarningType"].astype(str).eq("NotEnoughForwardEvidence").any():
            st.warning("Not enough forward evidence yet. Keep collecting paper outcomes before drawing conclusions.")

        st.markdown("### Forward Evidence Coverage")
        st.dataframe(forward_report.asset_horizon_forward_coverage, width="stretch")

        tabs = st.tabs([
            "Signal Log",
            "Pending",
            "Matured",
            "Accuracy",
            "Probability Calibration",
            "Warnings",
            "Next Actions",
        ])
        forward_tables = [
            ("Forward Signal Log", forward_report.forward_signal_log, "forward_signal_log.csv"),
            ("Pending Outcome Table", forward_report.pending_outcome_table, "forward_pending_outcomes.csv"),
            ("Matured Outcome Table", forward_report.matured_outcome_table, "forward_matured_outcomes.csv"),
            ("Forward Accuracy Summary", forward_report.forward_accuracy_summary, "forward_accuracy_summary.csv"),
            ("Forward Probability Calibration Summary", forward_report.forward_probability_calibration_summary, "forward_probability_calibration_summary.csv"),
            ("Warning Table", forward_report.warning_table, "forward_warning_table.csv"),
            ("Next Research Action Table", forward_report.next_research_action_table, "forward_next_research_actions.csv"),
        ]
        for tab, (title, table, filename) in zip(tabs, forward_tables):
            with tab:
                st.markdown(f"### {title}")
                st.dataframe(table, width="stretch")
                st.download_button(
                    f"📥 Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )


# PAGE: ACTIONABLE RESEARCH PLAN
# ═══════════════════════════════════════════════════════════════════════

elif page == "🧭 Actionable Research Plan":
    st.markdown('<p class="main-header">🧭 Actionable Research Plan</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "This page converts research evidence into an honest action plan. It does not execute trades "
        "and does not provide financial advice."
    )
    plan_use_latest_saved = st.checkbox(
        "Use latest saved evidence from artifact store",
        value=True,
        key="plan_use_latest_saved_artifacts",
    )

    latest_probability_report = st.session_state.get("probability_calibration_report")
    latest_true_raw_report = st.session_state.get("true_raw_trade_log_report")
    latest_forward_report = st.session_state.get("forward_paper_evidence_report")

    latest_probability_summary = getattr(latest_probability_report, "calibration_summary_table", None) if latest_probability_report is not None else None
    latest_probability_warnings = getattr(latest_probability_report, "warning_table", None) if latest_probability_report is not None else None
    latest_probability_recommendations = getattr(latest_probability_report, "candidate_recommendation_table", None) if latest_probability_report is not None else None
    latest_probability_bins = getattr(latest_probability_report, "probability_bin_table", None) if latest_probability_report is not None else None
    latest_true_raw = getattr(latest_true_raw_report, "true_raw_trade_log", None) if latest_true_raw_report is not None else None
    latest_forward_log = getattr(latest_forward_report, "forward_signal_log", None) if latest_forward_report is not None else None
    latest_forward_accuracy = getattr(latest_forward_report, "forward_accuracy_summary", None) if latest_forward_report is not None else None
    latest_forward_probability = getattr(latest_forward_report, "forward_probability_calibration_summary", None) if latest_forward_report is not None else None
    latest_forward_warnings = getattr(latest_forward_report, "warning_table", None) if latest_forward_report is not None else None
    latest_forward_next = getattr(latest_forward_report, "next_research_action_table", None) if latest_forward_report is not None else None
    latest_forward_coverage = getattr(latest_forward_report, "asset_horizon_forward_coverage", None) if latest_forward_report is not None else None

    up_a, up_b, up_c, up_d = st.columns(4)
    with up_a:
        uploaded_plan_true_raw = st.file_uploader("Upload true_raw_trade_log.csv", type=["csv"], key="plan_true_raw_upload")
        uploaded_plan_prob_summary = st.file_uploader("Upload probability_calibration_summary.csv", type=["csv"], key="plan_prob_summary_upload")
        uploaded_plan_prob_warnings = st.file_uploader("Upload probability_calibration_warnings.csv", type=["csv"], key="plan_prob_warnings_upload")
    with up_b:
        uploaded_plan_prob_recs = st.file_uploader("Optional probability_calibration_recommendations.csv", type=["csv"], key="plan_prob_recs_upload")
        uploaded_plan_prob_bins = st.file_uploader("Optional probability_calibration_bins.csv", type=["csv"], key="plan_prob_bins_upload")
        uploaded_plan_predictions = st.file_uploader("Optional latest model predictions CSV", type=["csv"], key="plan_predictions_upload")
    with up_c:
        uploaded_plan_forward_log = st.file_uploader("Upload forward_signal_log.csv", type=["csv"], key="plan_forward_log_upload")
        uploaded_plan_forward_accuracy = st.file_uploader("Optional forward_accuracy_summary.csv", type=["csv"], key="plan_forward_accuracy_upload")
        uploaded_plan_forward_prob = st.file_uploader("Optional forward_probability_calibration_summary.csv", type=["csv"], key="plan_forward_prob_upload")
    with up_d:
        uploaded_plan_forward_warnings = st.file_uploader("Optional forward_warning_table.csv", type=["csv"], key="plan_forward_warning_upload")
        uploaded_plan_forward_next = st.file_uploader("Optional forward_next_research_actions.csv", type=["csv"], key="plan_forward_next_upload")
        uploaded_plan_forward_coverage = st.file_uploader("Optional forward evidence coverage CSV", type=["csv"], key="plan_forward_coverage_upload")

    def _resolve_plan_input(phase_name, artifact_name, uploaded_file=None):
        if uploaded_file is not None:
            return resolve_artifact(phase_name, artifact_name, uploaded_file=uploaded_file, prefer_uploaded=True, required=False)
        if plan_use_latest_saved:
            return resolve_artifact(phase_name, artifact_name, required=False)
        return {
            "Artifact": artifact_name,
            "Phase": phase_name,
            "Data": None,
            "Source": "Missing",
            "RunId": "",
            "Rows": 0,
            "CreatedAt": "",
            "Status": "MissingOptional",
            "Path": "",
        }

    plan_resolved_inputs = {
        "true_raw_trade_log": _resolve_plan_input("Phase 8I True Raw Trade Logs", "true_raw_trade_log", uploaded_plan_true_raw),
        "probability_calibration_summary": _resolve_plan_input("Phase 8F Probability Calibration", "probability_calibration_summary", uploaded_plan_prob_summary),
        "probability_calibration_warnings": _resolve_plan_input("Phase 8F Probability Calibration", "probability_calibration_warnings", uploaded_plan_prob_warnings),
        "probability_calibration_recommendations": _resolve_plan_input("Phase 8F Probability Calibration", "probability_calibration_recommendations", uploaded_plan_prob_recs),
        "probability_calibration_bins": _resolve_plan_input("Phase 8F Probability Calibration", "probability_calibration_bins", uploaded_plan_prob_bins),
        "forward_signal_log": _resolve_plan_input("Phase 9 Forward Paper Evidence", "forward_signal_log", uploaded_plan_forward_log),
        "forward_accuracy_summary": _resolve_plan_input("Phase 9 Forward Paper Evidence", "forward_accuracy_summary", uploaded_plan_forward_accuracy),
        "forward_probability_calibration_summary": _resolve_plan_input("Phase 9 Forward Paper Evidence", "forward_probability_calibration_summary", uploaded_plan_forward_prob),
        "forward_warning_table": _resolve_plan_input("Phase 9 Forward Paper Evidence", "forward_warning_table", uploaded_plan_forward_warnings),
        "forward_next_research_actions": _resolve_plan_input("Phase 9 Forward Paper Evidence", "forward_next_research_actions", uploaded_plan_forward_next),
        "forward_evidence_coverage": _resolve_plan_input("Phase 9 Forward Paper Evidence", "asset_horizon_forward_coverage", uploaded_plan_forward_coverage),
        "latest_model_predictions": _resolve_plan_input("Phase 10 External Inputs", "latest_model_predictions", uploaded_plan_predictions),
    }
    plan_input_source_table = build_input_source_table(list(plan_resolved_inputs.values()))
    st.markdown("### Input Sources")
    st.dataframe(plan_input_source_table, width="stretch")

    plan_col_a, plan_col_b, plan_col_c, plan_col_d = st.columns(4)
    with plan_col_a:
        plan_assets = st.multiselect("Assets", get_asset_names(), default=get_asset_names(), key="plan_assets")
    with plan_col_b:
        plan_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        selected_plan_horizon_labels = st.multiselect("Horizons", plan_horizon_labels, default=plan_horizon_labels, key="plan_horizons")
        plan_horizons = [int(str(label).replace("D", "")) for label in selected_plan_horizon_labels]
    with plan_col_c:
        plan_risk_appetite = st.selectbox(
            "Risk appetite",
            ["Conservative", "Balanced Research", "Aggressive Paper Research"],
            index=0,
            key="plan_risk_appetite",
        )
    with plan_col_d:
        plan_min_evidence = st.slider("Minimum evidence score", min_value=0, max_value=100, value=35, step=5, key="plan_min_evidence")
        plan_top_n = st.number_input("Top N plan cards", min_value=1, max_value=30, value=10, step=1, key="plan_top_n")

    plan_include_blocked = st.checkbox("Include blocked candidates in ranked table", value=True, key="plan_include_blocked")
    run_plan = st.button("🚀 Build Actionable Research Plan", type="primary")

    if run_plan:
        if not plan_assets or not plan_horizons:
            st.error("Select at least one asset and one horizon.")
            st.stop()

        try:
            plan_report = run_actionable_research_plan(
                probability_calibration_summary=plan_resolved_inputs["probability_calibration_summary"]["Data"],
                probability_calibration_warnings=plan_resolved_inputs["probability_calibration_warnings"]["Data"],
                probability_calibration_recommendations=plan_resolved_inputs["probability_calibration_recommendations"]["Data"],
                probability_calibration_bins=plan_resolved_inputs["probability_calibration_bins"]["Data"],
                true_raw_trade_log=plan_resolved_inputs["true_raw_trade_log"]["Data"],
                forward_signal_log=plan_resolved_inputs["forward_signal_log"]["Data"],
                forward_accuracy_summary=plan_resolved_inputs["forward_accuracy_summary"]["Data"],
                forward_probability_calibration_summary=plan_resolved_inputs["forward_probability_calibration_summary"]["Data"],
                forward_warning_table=plan_resolved_inputs["forward_warning_table"]["Data"],
                forward_next_research_actions=plan_resolved_inputs["forward_next_research_actions"]["Data"],
                forward_evidence_coverage=plan_resolved_inputs["forward_evidence_coverage"]["Data"],
                latest_model_predictions=plan_resolved_inputs["latest_model_predictions"]["Data"],
                assets=plan_assets,
                horizons=plan_horizons,
                risk_appetite=plan_risk_appetite,
                minimum_evidence_score=float(plan_min_evidence),
                include_blocked_candidates=bool(plan_include_blocked),
                top_n_plan_cards=int(plan_top_n),
            )
            st.session_state.actionable_research_plan_report = plan_report
            st.session_state.actionable_research_plan_settings = plan_report.settings
            saved_artifacts = save_phase_artifacts(
                "Phase 10 Actionable Research Plan",
                {
                    "executive_decision_table": plan_report.executive_decision_table,
                    "ranked_asset_horizon_plan": plan_report.ranked_asset_horizon_plan,
                    "plan_card_table": plan_report.plan_card_table,
                    "entry_trigger_table": plan_report.entry_trigger_table,
                    "invalidation_rule_table": plan_report.invalidation_rule_table,
                    "risk_budget_table": plan_report.risk_budget_table,
                    "evidence_scorecard": plan_report.evidence_scorecard,
                    "blocked_candidates_table": plan_report.blocked_candidates_table,
                    "watchlist_table": plan_report.watchlist_table,
                    "paper_trade_plan_table": plan_report.paper_trade_plan_table,
                    "next_evidence_needed_table": plan_report.next_evidence_needed_table,
                    "warnings_table": plan_report.warnings_table,
                    "input_source_table": plan_input_source_table,
                },
                inputs={k: {kk: vv for kk, vv in v.items() if kk != "Data"} for k, v in plan_resolved_inputs.items()},
                config=plan_report.settings,
                warnings=plan_report.warnings_table["WarningType"].dropna().astype(str).unique().tolist() if not plan_report.warnings_table.empty and "WarningType" in plan_report.warnings_table.columns else [],
            )
            st.session_state.artifact_store_last_save = saved_artifacts
        except Exception as exc:
            st.error(f"Actionable research plan failed: {exc}")
            st.stop()

    plan_report = st.session_state.actionable_research_plan_report
    if plan_report is None:
        st.info("Upload Phase 8/9 evidence or run the prior pages, then build the research action plan.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('actionable_research_plan_settings') or {}}")
        if not plan_report.warnings_table.empty and plan_report.warnings_table["WarningType"].astype(str).eq("NoCandidateDeploymentReady").any():
            st.warning("Real capital remains blocked; paper-only and watchlist research actions are shown when evidence supports tracking.")

        st.markdown("### Executive Summary")
        st.dataframe(plan_report.executive_decision_table, width="stretch")
        st.download_button(
            "📥 Export Executive Summary CSV",
            data=plan_report.executive_decision_table.to_csv(index=False).encode("utf-8"),
            file_name="action_plan_executive_decision.csv",
            mime="text/csv",
            key="plan_executive_download",
        )

        tabs = st.tabs([
            "Top Plan Cards",
            "Ranked Plan",
            "Paper Plan",
            "Watchlist",
            "Capital Blocked",
            "Triggers",
            "Invalidation",
            "Risk Budget",
            "Evidence Scorecard",
            "Evidence Needed",
            "Warnings",
        ])
        plan_tables = [
            ("Plan Card Table", plan_report.plan_card_table, "action_plan_cards.csv"),
            ("Ranked Asset-Horizon Plan", plan_report.ranked_asset_horizon_plan, "action_plan_ranked.csv"),
            ("Paper Trade Plan Table", plan_report.paper_trade_plan_table, "action_plan_paper_only.csv"),
            ("Watchlist Table", plan_report.watchlist_table, "action_plan_watchlist.csv"),
            ("Capital-Blocked Candidates Table", plan_report.blocked_candidates_table, "action_plan_blocked.csv"),
            ("Entry Trigger Table", plan_report.entry_trigger_table, "action_plan_entry_triggers.csv"),
            ("Invalidation Rule Table", plan_report.invalidation_rule_table, "action_plan_invalidation_rules.csv"),
            ("Risk Budget Table", plan_report.risk_budget_table, "action_plan_risk_budget.csv"),
            ("Evidence Scorecard", plan_report.evidence_scorecard, "action_plan_evidence_scorecard.csv"),
            ("Next Evidence Needed Table", plan_report.next_evidence_needed_table, "action_plan_next_evidence_needed.csv"),
            ("Warnings Table", plan_report.warnings_table, "action_plan_warnings.csv"),
        ]
        for tab, (title, table, filename) in zip(tabs, plan_tables):
            with tab:
                st.markdown(f"### {title}")
                st.dataframe(table, width="stretch")
                st.download_button(
                    f"📥 Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )


# PAGE: DAILY RESEARCH CONTROL CENTER
# ═══════════════════════════════════════════════════════════════

elif page == "🧠 Daily Research Control Center":
    st.markdown('<p class="main-header">🧠 Daily Research Control Center</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "This page is a daily research-control cockpit. It tracks paper evidence and capital eligibility gates. "
        "It does not guarantee returns or execute trades."
    )

    daily_use_latest = st.checkbox("Use latest saved artifacts", value=True, key="daily_use_latest_artifacts")
    daily_prefer_uploads = st.checkbox(
        "Prefer manual upload overrides",
        value=False,
        key="daily_prefer_uploads",
        help="When enabled, uploaded CSVs override latest saved artifacts for the matching table.",
    )

    with st.expander("Optional Manual Upload Overrides", expanded=False):
        du_a, du_b, du_c, du_d = st.columns(4)
        with du_a:
            uploaded_daily_true_raw = st.file_uploader("true_raw_trade_log.csv", type=["csv"], key="daily_true_raw_upload")
            uploaded_daily_prob_summary = st.file_uploader("probability_calibration_summary.csv", type=["csv"], key="daily_prob_summary_upload")
            uploaded_daily_prob_warnings = st.file_uploader("probability_calibration_warnings.csv", type=["csv"], key="daily_prob_warnings_upload")
        with du_b:
            uploaded_daily_forward_log = st.file_uploader("forward_signal_log.csv", type=["csv"], key="daily_forward_log_upload")
            uploaded_daily_pending = st.file_uploader("pending_outcome_table.csv", type=["csv"], key="daily_pending_upload")
            uploaded_daily_matured = st.file_uploader("matured_outcome_table.csv", type=["csv"], key="daily_matured_upload")
        with du_c:
            uploaded_daily_forward_accuracy = st.file_uploader("forward_accuracy_summary.csv", type=["csv"], key="daily_forward_accuracy_upload")
            uploaded_daily_forward_prob = st.file_uploader("forward_probability_calibration_summary.csv", type=["csv"], key="daily_forward_prob_upload")
            uploaded_daily_forward_warnings = st.file_uploader("forward warning_table.csv", type=["csv"], key="daily_forward_warnings_upload")
        with du_d:
            uploaded_daily_ranked = st.file_uploader("ranked_asset_horizon_plan.csv", type=["csv"], key="daily_ranked_upload")
            uploaded_daily_paper = st.file_uploader("paper_trade_plan_table.csv", type=["csv"], key="daily_paper_upload")
            uploaded_daily_watch = st.file_uploader("watchlist_table.csv", type=["csv"], key="daily_watch_upload")
            uploaded_daily_risk = st.file_uploader("risk_budget_table.csv", type=["csv"], key="daily_risk_upload")
            uploaded_daily_phase10_warnings = st.file_uploader("Phase 10 warnings_table.csv", type=["csv"], key="daily_phase10_warnings_upload")

    uploaded_daily_overrides = {
        "true_raw_trade_log": uploaded_daily_true_raw,
        "probability_calibration_summary": uploaded_daily_prob_summary,
        "probability_calibration_warnings": uploaded_daily_prob_warnings,
        "forward_signal_log": uploaded_daily_forward_log,
        "pending_outcome_table": uploaded_daily_pending,
        "matured_outcome_table": uploaded_daily_matured,
        "forward_accuracy_summary": uploaded_daily_forward_accuracy,
        "forward_probability_calibration_summary": uploaded_daily_forward_prob,
        "forward_warning_table": uploaded_daily_forward_warnings,
        "ranked_asset_horizon_plan": uploaded_daily_ranked,
        "paper_trade_plan_table": uploaded_daily_paper,
        "watchlist_table": uploaded_daily_watch,
        "risk_budget_table": uploaded_daily_risk,
        "phase10_warnings_table": uploaded_daily_phase10_warnings,
    }

    dc_a, dc_b, dc_c, dc_d = st.columns(4)
    with dc_a:
        daily_report_date = st.date_input("Report date", value=pd.Timestamp.today().date(), key="daily_report_date")
        daily_assets = st.multiselect("Assets", get_asset_names(), default=get_asset_names(), key="daily_assets")
    with dc_b:
        daily_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        selected_daily_horizon_labels = st.multiselect("Horizons", daily_horizon_labels, default=daily_horizon_labels, key="daily_horizons")
        daily_horizons = [int(str(label).replace("D", "")) for label in selected_daily_horizon_labels]
        daily_include_blocked = st.checkbox("Show blocked candidates", value=True, key="daily_include_blocked")
    with dc_c:
        daily_mode = st.selectbox(
            "Capital eligibility mode",
            ["Conservative", "Balanced", "Aggressive Research"],
            index=0,
            key="daily_capital_mode",
        )
        daily_min_matured = st.number_input("Minimum matured forward outcomes", min_value=1, max_value=100, value=10, step=1, key="daily_min_matured")
    with dc_d:
        daily_max_drawdown = st.slider("Maximum drawdown allowed (%)", min_value=1.0, max_value=50.0, value=12.0, step=0.5, key="daily_max_drawdown")
        daily_max_cap = st.slider("Max real capital cap (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="daily_max_cap")

    run_daily_center = st.button("🚀 Run Daily Research Control Center", type="primary")
    if run_daily_center:
        if not daily_assets or not daily_horizons:
            st.error("Select at least one asset and one horizon.")
            st.stop()
        try:
            daily_report = run_daily_research_control_center(
                use_artifact_store=bool(daily_use_latest),
                prefer_uploaded=bool(daily_prefer_uploads),
                uploaded_overrides=uploaded_daily_overrides,
                report_date=daily_report_date,
                assets=daily_assets,
                horizons=daily_horizons,
                include_blocked_candidates=bool(daily_include_blocked),
                capital_eligibility_mode=daily_mode,
                minimum_matured_forward_outcomes=int(daily_min_matured),
                max_drawdown_allowed_pct=float(daily_max_drawdown),
                max_real_capital_pct=float(daily_max_cap),
            )
            st.session_state.daily_research_control_center_report = daily_report
            st.session_state.daily_research_control_center_settings = daily_report.settings
            saved_daily_artifacts = save_phase_artifacts(
                "Phase 11 Daily Research Control Center",
                {
                    "daily_research_summary": daily_report.daily_research_summary,
                    "active_paper_signals_table": daily_report.active_paper_signals_table,
                    "pending_outcomes_table": daily_report.pending_outcomes_table,
                    "matured_today_table": daily_report.matured_today_table,
                    "overdue_outcomes_table": daily_report.overdue_outcomes_table,
                    "top_paper_candidates_today": daily_report.top_paper_candidates_today,
                    "watchlist_review_table": daily_report.watchlist_review_table,
                    "capital_eligibility_table": daily_report.capital_eligibility_table,
                    "structured_capital_plan_table": daily_report.structured_capital_plan_table,
                    "capital_blocker_table": daily_report.capital_blocker_table,
                    "degraded_candidates_table": daily_report.degraded_candidates_table,
                    "improved_candidates_table": daily_report.improved_candidates_table,
                    "blocked_or_avoid_table": daily_report.blocked_or_avoid_table,
                    "evidence_health_table": daily_report.evidence_health_table,
                    "daily_next_actions_table": daily_report.daily_next_actions_table,
                    "warning_table": daily_report.warning_table,
                    "input_source_table": daily_report.input_source_table,
                },
                inputs={},
                config=daily_report.settings,
                warnings=daily_report.warning_table["WarningType"].dropna().astype(str).unique().tolist() if not daily_report.warning_table.empty and "WarningType" in daily_report.warning_table.columns else [],
            )
            st.session_state.artifact_store_last_save = saved_daily_artifacts
        except Exception as exc:
            st.error(f"Daily research control center failed: {exc}")
            st.stop()

    daily_report = st.session_state.daily_research_control_center_report
    if daily_report is None:
        st.info("Run the daily control center to review paper evidence, capital blockers, and strict eligibility gates.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('daily_research_control_center_settings') or {}}")
        if not daily_report.capital_eligibility_table.empty and not daily_report.capital_eligibility_table["RealCapitalAllowed"].astype(bool).any():
            st.warning("No candidate passed every real-capital gate. Paper-only candidates, watchlist reviews, and exact blockers remain visible.")
        if not daily_report.structured_capital_plan_table.empty:
            st.info("Conditional capital plans are capped, monitored, and invalidation-driven. They are not guarantees or trade execution instructions.")

        st.markdown("### Daily Executive Summary")
        st.dataframe(daily_report.daily_research_summary, width="stretch")
        st.markdown("### Input Sources")
        st.dataframe(daily_report.input_source_table, width="stretch")

        tabs = st.tabs(
            [
                "Active Paper Signals",
                "Pending Outcomes",
                "Matured Today",
                "Overdue Outcomes",
                "Best Paper Candidates",
                "Watchlist",
                "Capital Eligibility",
                "Structured Capital Plan",
                "Capital Blockers",
                "Improved",
                "Degraded",
                "Blocked / Avoid",
                "Evidence Health",
                "Daily Next Actions",
                "Warnings",
                "Input Sources",
            ]
        )
        daily_tables = [
            ("Active Paper Signals", daily_report.active_paper_signals_table, "daily_active_paper_signals.csv"),
            ("Pending Outcomes", daily_report.pending_outcomes_table, "daily_pending_outcomes.csv"),
            ("Matured Today", daily_report.matured_today_table, "daily_matured_today.csv"),
            ("Overdue Outcomes", daily_report.overdue_outcomes_table, "daily_overdue_outcomes.csv"),
            ("Best Paper-Only Candidates", daily_report.top_paper_candidates_today, "daily_top_paper_candidates.csv"),
            ("Watchlist Review Table", daily_report.watchlist_review_table, "daily_watchlist_review.csv"),
            ("Capital Eligibility Table", daily_report.capital_eligibility_table, "daily_capital_eligibility.csv"),
            ("Structured Capital Plan Table", daily_report.structured_capital_plan_table, "daily_structured_capital_plan.csv"),
            ("Capital Blocker Table", daily_report.capital_blocker_table, "daily_capital_blockers.csv"),
            ("Improved Candidates Table", daily_report.improved_candidates_table, "daily_improved_candidates.csv"),
            ("Degraded Candidates Table", daily_report.degraded_candidates_table, "daily_degraded_candidates.csv"),
            ("Blocked Or Avoid Table", daily_report.blocked_or_avoid_table, "daily_blocked_or_avoid.csv"),
            ("Evidence Health Table", daily_report.evidence_health_table, "daily_evidence_health.csv"),
            ("Daily Next Actions Table", daily_report.daily_next_actions_table, "daily_next_actions.csv"),
            ("Warning Table", daily_report.warning_table, "daily_warnings.csv"),
            ("Input Source Table", daily_report.input_source_table, "daily_input_sources.csv"),
        ]
        for tab, (title, table, filename) in zip(tabs, daily_tables):
            with tab:
                st.markdown(f"### {title}")
                if title == "Structured Capital Plan Table" and table.empty:
                    st.info("No candidate passed every capital gate, so no structured capital plan is shown.")
                if title == "Capital Blocker Table" and not daily_report.capital_eligibility_table.empty and daily_report.capital_eligibility_table["RealCapitalAllowed"].astype(bool).any():
                    st.caption("Some candidates passed strict capital gates; blockers remain visible for the rest.")
                st.dataframe(table, width="stretch")
                st.download_button(
                    f"📥 Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )


# PAGE: PORTFOLIO & CAPITAL SIMULATOR
# ═══════════════════════════════════════════════════════════════

elif page == "💼 Portfolio & Capital Simulator":
    st.markdown('<p class="main-header">💼 Portfolio & Capital Simulator</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "This page simulates paper and conditional capital allocation under strict risk controls. "
        "It does not execute trades or guarantee returns."
    )

    portfolio_use_latest = st.checkbox("Use latest saved artifacts", value=True, key="portfolio_use_latest_artifacts")
    portfolio_prefer_uploads = st.checkbox(
        "Prefer manual upload overrides",
        value=False,
        key="portfolio_prefer_uploads",
        help="When enabled, uploaded CSVs override latest saved artifacts for the matching table.",
    )

    with st.expander("Optional Manual Upload Overrides", expanded=False):
        pu_a, pu_b, pu_c, pu_d = st.columns(4)
        with pu_a:
            uploaded_portfolio_plan_cards = st.file_uploader("Phase 10 plan_card_table.csv", type=["csv"], key="portfolio_plan_cards_upload")
            uploaded_portfolio_ranked = st.file_uploader("Phase 10 ranked_asset_horizon_plan.csv", type=["csv"], key="portfolio_ranked_upload")
            uploaded_portfolio_paper = st.file_uploader("Phase 10 paper_trade_plan_table.csv", type=["csv"], key="portfolio_paper_upload")
            uploaded_portfolio_watch = st.file_uploader("Phase 10 watchlist_table.csv", type=["csv"], key="portfolio_watch_upload")
        with pu_b:
            uploaded_portfolio_phase10_risk = st.file_uploader("Phase 10 risk_budget_table.csv", type=["csv"], key="portfolio_phase10_risk_upload")
            uploaded_portfolio_capital = st.file_uploader("Phase 11 capital_eligibility_table.csv", type=["csv"], key="portfolio_capital_upload")
            uploaded_portfolio_structured = st.file_uploader("Phase 11 structured_capital_plan_table.csv", type=["csv"], key="portfolio_structured_upload")
            uploaded_portfolio_blockers = st.file_uploader("Phase 11 capital_blocker_table.csv", type=["csv"], key="portfolio_blockers_upload")
        with pu_c:
            uploaded_portfolio_active = st.file_uploader("Phase 11 active_paper_signals_table.csv", type=["csv"], key="portfolio_active_upload")
            uploaded_portfolio_pending = st.file_uploader("Phase 11 pending_outcomes_table.csv", type=["csv"], key="portfolio_pending_upload")
            uploaded_portfolio_top_paper = st.file_uploader("Phase 11 top_paper_candidates_today.csv", type=["csv"], key="portfolio_top_paper_upload")
            uploaded_portfolio_health = st.file_uploader("Phase 11 evidence_health_table.csv", type=["csv"], key="portfolio_health_upload")
        with pu_d:
            uploaded_portfolio_warnings = st.file_uploader("Phase 11 warning_table.csv", type=["csv"], key="portfolio_warnings_upload")
            uploaded_portfolio_raw = st.file_uploader("Phase 8I true_raw_trade_log.csv", type=["csv"], key="portfolio_raw_upload")
            uploaded_portfolio_prob = st.file_uploader("Phase 8F probability_calibration_summary.csv", type=["csv"], key="portfolio_prob_upload")
            uploaded_portfolio_forward = st.file_uploader("Phase 9 forward_signal_log.csv", type=["csv"], key="portfolio_forward_upload")

    uploaded_portfolio_overrides = {
        "plan_card_table": uploaded_portfolio_plan_cards,
        "ranked_asset_horizon_plan": uploaded_portfolio_ranked,
        "paper_trade_plan_table": uploaded_portfolio_paper,
        "watchlist_table": uploaded_portfolio_watch,
        "phase10_risk_budget_table": uploaded_portfolio_phase10_risk,
        "capital_eligibility_table": uploaded_portfolio_capital,
        "structured_capital_plan_table": uploaded_portfolio_structured,
        "capital_blocker_table": uploaded_portfolio_blockers,
        "active_paper_signals_table": uploaded_portfolio_active,
        "pending_outcomes_table": uploaded_portfolio_pending,
        "top_paper_candidates_today": uploaded_portfolio_top_paper,
        "evidence_health_table": uploaded_portfolio_health,
        "phase11_warning_table": uploaded_portfolio_warnings,
        "true_raw_trade_log": uploaded_portfolio_raw,
        "probability_calibration_summary": uploaded_portfolio_prob,
        "forward_signal_log": uploaded_portfolio_forward,
    }

    pc_a, pc_b, pc_c, pc_d = st.columns(4)
    with pc_a:
        portfolio_mode = st.selectbox(
            "Portfolio mode",
            ["Conservative", "Balanced Research", "Aggressive Paper Research"],
            index=0,
            key="portfolio_mode",
        )
        portfolio_assets = st.multiselect("Assets", get_asset_names(), default=get_asset_names(), key="portfolio_assets")
    with pc_b:
        portfolio_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        selected_portfolio_horizon_labels = st.multiselect("Horizons", portfolio_horizon_labels, default=portfolio_horizon_labels, key="portfolio_horizons")
        portfolio_horizons = [int(str(label).replace("D", "")) for label in selected_portfolio_horizon_labels]
        portfolio_paper_capital = st.number_input("Total paper capital", min_value=100.0, max_value=10000000.0, value=100000.0, step=1000.0, key="portfolio_paper_capital")
    with pc_c:
        portfolio_real_cap = st.slider("Max real capital cap (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="portfolio_real_cap")
        portfolio_single_loss = st.slider("Max single idea loss (%)", min_value=0.05, max_value=5.0, value=0.25, step=0.05, key="portfolio_single_loss")
        portfolio_total_loss = st.slider("Max portfolio loss (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="portfolio_total_loss")
    with pc_d:
        portfolio_asset_cap = st.slider("Max single asset exposure (%)", min_value=1.0, max_value=100.0, value=25.0, step=1.0, key="portfolio_asset_cap")
        portfolio_horizon_cap = st.slider("Max single horizon exposure (%)", min_value=1.0, max_value=100.0, value=35.0, step=1.0, key="portfolio_horizon_cap")
        portfolio_include_watch = st.checkbox("Include watchlist candidates", value=True, key="portfolio_include_watch")
        portfolio_include_blocked = st.checkbox("Include blocked candidates", value=True, key="portfolio_include_blocked")

    run_portfolio = st.button("🚀 Run Portfolio & Capital Simulator", type="primary")
    if run_portfolio:
        if not portfolio_assets or not portfolio_horizons:
            st.error("Select at least one asset and one horizon.")
            st.stop()
        try:
            portfolio_report = run_portfolio_capital_simulator(
                use_artifact_store=bool(portfolio_use_latest),
                prefer_uploaded=bool(portfolio_prefer_uploads),
                uploaded_overrides=uploaded_portfolio_overrides,
                assets=portfolio_assets,
                horizons=portfolio_horizons,
                portfolio_mode=portfolio_mode,
                total_paper_capital=float(portfolio_paper_capital),
                max_real_capital_cap_pct=float(portfolio_real_cap),
                max_single_idea_loss_pct=float(portfolio_single_loss),
                max_portfolio_loss_pct=float(portfolio_total_loss),
                max_single_asset_exposure_pct=float(portfolio_asset_cap),
                max_single_horizon_exposure_pct=float(portfolio_horizon_cap),
                include_watchlist_candidates=bool(portfolio_include_watch),
                include_blocked_candidates=bool(portfolio_include_blocked),
            )
            st.session_state.portfolio_capital_simulator_report = portfolio_report
            st.session_state.portfolio_capital_simulator_settings = portfolio_report.settings
            saved_portfolio_artifacts = save_phase_artifacts(
                "Phase 12 Portfolio Capital Simulator",
                {
                    "portfolio_summary_table": portfolio_report.portfolio_summary_table,
                    "allocation_plan_table": portfolio_report.allocation_plan_table,
                    "paper_portfolio_table": portfolio_report.paper_portfolio_table,
                    "conditional_real_capital_table": portfolio_report.conditional_real_capital_table,
                    "position_sizing_table": portfolio_report.position_sizing_table,
                    "risk_budget_table": portfolio_report.risk_budget_table,
                    "portfolio_drawdown_stress_table": portfolio_report.portfolio_drawdown_stress_table,
                    "correlation_concentration_table": portfolio_report.correlation_concentration_table,
                    "cost_slippage_stress_table": portfolio_report.cost_slippage_stress_table,
                    "stop_exit_plan_table": portfolio_report.stop_exit_plan_table,
                    "capital_blocker_table": portfolio_report.capital_blocker_table,
                    "scenario_analysis_table": portfolio_report.scenario_analysis_table,
                    "next_actions_table": portfolio_report.next_actions_table,
                    "warning_table": portfolio_report.warning_table,
                    "input_source_table": portfolio_report.input_source_table,
                },
                inputs={},
                config=portfolio_report.settings,
                warnings=portfolio_report.warning_table["WarningType"].dropna().astype(str).unique().tolist() if not portfolio_report.warning_table.empty and "WarningType" in portfolio_report.warning_table.columns else [],
            )
            st.session_state.artifact_store_last_save = saved_portfolio_artifacts
        except Exception as exc:
            st.error(f"Portfolio simulator failed: {exc}")
            st.stop()

    portfolio_report = st.session_state.portfolio_capital_simulator_report
    if portfolio_report is None:
        st.info("Run the simulator to build a paper portfolio and any Phase 11-gated conditional capital plan.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('portfolio_capital_simulator_settings') or {}}")
        if not portfolio_report.portfolio_summary_table.empty:
            main_warning = str(portfolio_report.portfolio_summary_table.iloc[0].get("MainRiskWarning", ""))
            if main_warning == "RealCapitalBlocked":
                st.warning("No candidate passed Phase 11 real-capital gates. Paper allocation, watchlist rows, and blockers remain visible.")
            elif main_warning:
                st.info("Conditional allocations are capped by risk controls and remain research-only.")

        st.markdown("### Portfolio Executive Summary")
        st.dataframe(portfolio_report.portfolio_summary_table, width="stretch")
        st.markdown("### Input Sources")
        st.dataframe(portfolio_report.input_source_table, width="stretch")

        tabs = st.tabs(
            [
                "Allocation Plan",
                "Paper Portfolio",
                "Conditional Capital",
                "Position Sizing",
                "Risk Budget",
                "Drawdown Stress",
                "Concentration",
                "Cost Stress",
                "Stop / Exit",
                "Capital Blockers",
                "Scenarios",
                "Next Actions",
                "Warnings",
                "Input Sources",
            ]
        )
        portfolio_tables = [
            ("Allocation Plan", portfolio_report.allocation_plan_table, "portfolio_allocation_plan.csv"),
            ("Paper Portfolio", portfolio_report.paper_portfolio_table, "portfolio_paper.csv"),
            ("Conditional Real-Capital Table", portfolio_report.conditional_real_capital_table, "portfolio_conditional_capital.csv"),
            ("Position Sizing", portfolio_report.position_sizing_table, "portfolio_position_sizing.csv"),
            ("Risk Budget", portfolio_report.risk_budget_table, "portfolio_risk_budget.csv"),
            ("Drawdown Stress", portfolio_report.portfolio_drawdown_stress_table, "portfolio_drawdown_stress.csv"),
            ("Correlation / Concentration", portfolio_report.correlation_concentration_table, "portfolio_concentration.csv"),
            ("Cost / Slippage Stress", portfolio_report.cost_slippage_stress_table, "portfolio_cost_stress.csv"),
            ("Stop / Exit Plan", portfolio_report.stop_exit_plan_table, "portfolio_stop_exit.csv"),
            ("Capital Blockers", portfolio_report.capital_blocker_table, "portfolio_capital_blockers.csv"),
            ("Scenario Analysis", portfolio_report.scenario_analysis_table, "portfolio_scenarios.csv"),
            ("Next Actions", portfolio_report.next_actions_table, "portfolio_next_actions.csv"),
            ("Warnings", portfolio_report.warning_table, "portfolio_warnings.csv"),
            ("Input Sources", portfolio_report.input_source_table, "portfolio_input_sources.csv"),
        ]
        for tab, (title, table, filename) in zip(tabs, portfolio_tables):
            with tab:
                st.markdown(f"### {title}")
                if title == "Conditional Real-Capital Table" and table.empty:
                    st.info("No candidate passed Phase 11 real-capital gates, so no conditional capital allocation is shown.")
                st.dataframe(table, width="stretch")
                st.download_button(
                    f"📥 Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )

# PAGE: RISK & WARNING INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════════

elif page == "⚠️ Risk & Warning Intelligence":
    st.markdown('<p class="main-header">⚠️ Risk & Warning Intelligence</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "This dashboard ranks and explains warning evidence from prior research phases. "
        "It does not change signals, tune models, execute trades, or approve live deployment."
    )

    risk_use_latest = st.checkbox("Use latest saved artifacts", value=True, key="risk_warning_use_latest")
    risk_prefer_uploads = st.checkbox(
        "Prefer manual upload overrides",
        value=False,
        key="risk_warning_prefer_uploads",
        help="When enabled, uploaded CSVs override latest saved artifacts for matching inputs.",
    )

    with st.expander("Optional Manual Upload Overrides", expanded=False):
        rw_a, rw_b, rw_c, rw_d = st.columns(4)
        with rw_a:
            uploaded_rw_prob_summary = st.file_uploader("Phase 8F probability_calibration_summary.csv", type=["csv"], key="rw_prob_summary_upload")
            uploaded_rw_prob_warnings = st.file_uploader("Phase 8F probability_calibration_warnings.csv", type=["csv"], key="rw_prob_warnings_upload")
            uploaded_rw_forward_log = st.file_uploader("Phase 9 forward_signal_log.csv", type=["csv"], key="rw_forward_log_upload")
            uploaded_rw_forward_warnings = st.file_uploader("Phase 9 forward_warning_table.csv", type=["csv"], key="rw_forward_warnings_upload")
        with rw_b:
            uploaded_rw_pending = st.file_uploader("Phase 9 pending_outcome_table.csv", type=["csv"], key="rw_pending_upload")
            uploaded_rw_matured = st.file_uploader("Phase 9 matured_outcome_table.csv", type=["csv"], key="rw_matured_upload")
            uploaded_rw_forward_accuracy = st.file_uploader("Phase 9 forward_accuracy_summary.csv", type=["csv"], key="rw_forward_accuracy_upload")
            uploaded_rw_forward_prob = st.file_uploader("Phase 9 forward_probability_calibration_summary.csv", type=["csv"], key="rw_forward_prob_upload")
        with rw_c:
            uploaded_rw_phase10_warnings = st.file_uploader("Phase 10 warnings_table.csv", type=["csv"], key="rw_phase10_warnings_upload")
            uploaded_rw_phase10_blocked = st.file_uploader("Phase 10 blocked_candidates_table.csv", type=["csv"], key="rw_phase10_blocked_upload")
            uploaded_rw_phase10_risk = st.file_uploader("Phase 10 risk_budget_table.csv", type=["csv"], key="rw_phase10_risk_upload")
            uploaded_rw_phase11_capital = st.file_uploader("Phase 11 capital_eligibility_table.csv", type=["csv"], key="rw_phase11_capital_upload")
        with rw_d:
            uploaded_rw_phase11_blockers = st.file_uploader("Phase 11 capital_blocker_table.csv", type=["csv"], key="rw_phase11_blockers_upload")
            uploaded_rw_phase11_warnings = st.file_uploader("Phase 11 warning_table.csv", type=["csv"], key="rw_phase11_warnings_upload")
            uploaded_rw_phase12_allocation = st.file_uploader("Phase 12 allocation_plan_table.csv", type=["csv"], key="rw_phase12_allocation_upload")
            uploaded_rw_phase12_warnings = st.file_uploader("Phase 12 warning_table.csv", type=["csv"], key="rw_phase12_warnings_upload")

        rw_e, rw_f, rw_g = st.columns(3)
        with rw_e:
            uploaded_rw_phase11_pending = st.file_uploader("Phase 11 pending_outcomes_table.csv", type=["csv"], key="rw_phase11_pending_upload")
            uploaded_rw_phase11_matured = st.file_uploader("Phase 11 matured_today_table.csv", type=["csv"], key="rw_phase11_matured_upload")
            uploaded_rw_phase11_health = st.file_uploader("Phase 11 evidence_health_table.csv", type=["csv"], key="rw_phase11_health_upload")
        with rw_f:
            uploaded_rw_phase12_blockers = st.file_uploader("Phase 12 capital_blocker_table.csv", type=["csv"], key="rw_phase12_blockers_upload")
            uploaded_rw_phase12_drawdown = st.file_uploader("Phase 12 portfolio_drawdown_stress_table.csv", type=["csv"], key="rw_phase12_drawdown_upload")
            uploaded_rw_phase12_concentration = st.file_uploader("Phase 12 correlation_concentration_table.csv", type=["csv"], key="rw_phase12_concentration_upload")
        with rw_g:
            uploaded_rw_phase12_cost = st.file_uploader("Phase 12 cost_slippage_stress_table.csv", type=["csv"], key="rw_phase12_cost_upload")
            uploaded_rw_phase12_scenario = st.file_uploader("Phase 12 scenario_analysis_table.csv", type=["csv"], key="rw_phase12_scenario_upload")
            uploaded_rw_phase12_risk = st.file_uploader("Phase 12 risk_budget_table.csv", type=["csv"], key="rw_phase12_risk_upload")

    risk_uploaded_overrides = {
        "probability_calibration_summary": uploaded_rw_prob_summary,
        "probability_calibration_warnings": uploaded_rw_prob_warnings,
        "forward_signal_log": uploaded_rw_forward_log,
        "forward_warning_table": uploaded_rw_forward_warnings,
        "pending_outcome_table": uploaded_rw_pending,
        "matured_outcome_table": uploaded_rw_matured,
        "forward_accuracy_summary": uploaded_rw_forward_accuracy,
        "forward_probability_calibration_summary": uploaded_rw_forward_prob,
        "phase10_warnings_table": uploaded_rw_phase10_warnings,
        "phase10_blocked_candidates_table": uploaded_rw_phase10_blocked,
        "phase10_risk_budget_table": uploaded_rw_phase10_risk,
        "capital_eligibility_table": uploaded_rw_phase11_capital,
        "phase11_capital_blocker_table": uploaded_rw_phase11_blockers,
        "phase11_warning_table": uploaded_rw_phase11_warnings,
        "phase11_pending_outcomes_table": uploaded_rw_phase11_pending,
        "matured_today_table": uploaded_rw_phase11_matured,
        "evidence_health_table": uploaded_rw_phase11_health,
        "allocation_plan_table": uploaded_rw_phase12_allocation,
        "phase12_warning_table": uploaded_rw_phase12_warnings,
        "phase12_capital_blocker_table": uploaded_rw_phase12_blockers,
        "portfolio_drawdown_stress_table": uploaded_rw_phase12_drawdown,
        "correlation_concentration_table": uploaded_rw_phase12_concentration,
        "cost_slippage_stress_table": uploaded_rw_phase12_cost,
        "scenario_analysis_table": uploaded_rw_phase12_scenario,
        "phase12_risk_budget_table": uploaded_rw_phase12_risk,
    }

    rw_c1, rw_c2 = st.columns(2)
    with rw_c1:
        risk_assets = st.multiselect("Assets", get_asset_names(), default=get_asset_names(), key="risk_warning_assets")
    with rw_c2:
        risk_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        selected_risk_horizon_labels = st.multiselect("Horizons", risk_horizon_labels, default=risk_horizon_labels, key="risk_warning_horizons")
        risk_horizons = [int(str(label).replace("D", "")) for label in selected_risk_horizon_labels]

    run_risk_warning = st.button("🚀 Build Risk & Warning Intelligence", type="primary")
    if run_risk_warning:
        if not risk_assets or not risk_horizons:
            st.error("Select at least one asset and one horizon.")
            st.stop()
        try:
            risk_report = run_risk_warning_intelligence(
                use_artifact_store=bool(risk_use_latest),
                prefer_uploaded=bool(risk_prefer_uploads),
                uploaded_overrides=risk_uploaded_overrides,
                assets=risk_assets,
                horizons=risk_horizons,
            )
            st.session_state.risk_warning_intelligence_report = risk_report
            st.session_state.risk_warning_intelligence_settings = risk_report.settings
            saved_risk_artifacts = save_phase_artifacts(
                RISK_INTELLIGENCE_PHASE_NAME,
                {
                    "risk_summary_table": risk_report.risk_summary_table,
                    "top_risks_table": risk_report.top_risks_table,
                    "capital_blocking_risks_table": risk_report.capital_blocking_risks_table,
                    "paper_only_risks_table": risk_report.paper_only_risks_table,
                    "warning_group_table": risk_report.warning_group_table,
                    "asset_horizon_risk_matrix": risk_report.asset_horizon_risk_matrix,
                    "risk_trend_or_status_table": risk_report.risk_trend_or_status_table,
                    "next_risk_actions_table": risk_report.next_risk_actions_table,
                    "raw_warning_evidence": risk_report.raw_warning_evidence,
                    "input_source_table": risk_report.input_source_table,
                },
                inputs={},
                config=risk_report.settings,
                warnings=risk_report.raw_warning_evidence["WarningType"].dropna().astype(str).unique().tolist() if not risk_report.raw_warning_evidence.empty else [],
            )
            st.session_state.artifact_store_last_save = saved_risk_artifacts
        except Exception as exc:
            st.error(f"Risk intelligence failed: {exc}")
            st.stop()

    risk_report = st.session_state.risk_warning_intelligence_report
    if risk_report is None:
        st.info("Run the dashboard using latest saved artifacts or upload warning evidence CSVs.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('risk_warning_intelligence_settings') or {}}")

        max_risk = 0.0
        if not risk_report.risk_summary_table.empty:
            max_risk = float(pd.to_numeric(risk_report.risk_summary_table["RiskScore"], errors="coerce").fillna(0).max())
        verdict = "Critical risk cluster" if max_risk >= 80 else "High risk cluster" if max_risk >= 60 else "Moderate risk cluster" if max_risk >= 35 else "Low current warning load"
        capital_status = "Real capital remains blocked or reduced by current evidence." if not risk_report.capital_blocking_risks_table.empty else "No loaded capital-blocking risk rows."
        paper_status = "Paper tracking can continue with monitoring or reduced size where indicated." if not risk_report.paper_only_risks_table.empty else "No loaded paper-specific risk rows."

        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("Overall Risk Verdict", verdict)
        oc2.metric("Real Capital Status", capital_status)
        oc3.metric("Paper Trading Status", paper_status)

        st.markdown("### Top Risks")
        st.dataframe(risk_report.top_risks_table.head(10), width="stretch")
        st.markdown("### Input Sources")
        st.dataframe(risk_report.input_source_table, width="stretch")

        risk_tabs = st.tabs(
            [
                "Risk Summary",
                "Top Risks",
                "Capital Blocking",
                "Paper-Only",
                "Asset/Horizon Matrix",
                "Warning Groups",
                "Risk Status",
                "Next Actions",
                "Raw Warning Evidence",
                "Input Sources",
            ]
        )
        risk_tables = [
            ("Risk Summary", risk_report.risk_summary_table, "phase13_risk_summary.csv"),
            ("Top Risks", risk_report.top_risks_table, "phase13_top_risks.csv"),
            ("Capital Blocking Risks", risk_report.capital_blocking_risks_table, "phase13_capital_blocking_risks.csv"),
            ("Paper-Only Risks", risk_report.paper_only_risks_table, "phase13_paper_only_risks.csv"),
            ("Asset/Horizon Risk Matrix", risk_report.asset_horizon_risk_matrix, "phase13_asset_horizon_risk_matrix.csv"),
            ("Warning Groups", risk_report.warning_group_table, "phase13_warning_groups.csv"),
            ("Risk Status", risk_report.risk_trend_or_status_table, "phase13_risk_status.csv"),
            ("Next Risk Actions", risk_report.next_risk_actions_table, "phase13_next_risk_actions.csv"),
            ("Raw Warning Evidence", risk_report.raw_warning_evidence, "phase13_raw_warning_evidence.csv"),
            ("Input Sources", risk_report.input_source_table, "phase13_input_sources.csv"),
        ]
        for tab, (title, table, filename) in zip(risk_tabs, risk_tables):
            with tab:
                st.markdown(f"### {title}")
                if table.empty:
                    st.info("No rows for this table with the currently loaded evidence.")
                st.dataframe(table, width="stretch")
                st.download_button(
                    f"📥 Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )


# PAGE: DYNAMIC RISK SIZING
# ════════════════════════════════════════════════════════════════════════════════

elif page == "📐 Dynamic Risk Sizing":
    st.markdown('<p class="main-header">📐 Dynamic Risk Sizing</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "This page dynamically adjusts simulated paper sizing using Phase 12 allocations and Phase 13 risk intelligence. "
        "It does not execute trades or override real-capital gates."
    )

    sizing_use_latest = st.checkbox("Use latest saved artifacts", value=True, key="dynamic_sizing_use_latest")
    sizing_prefer_uploads = st.checkbox(
        "Prefer manual upload overrides",
        value=False,
        key="dynamic_sizing_prefer_uploads",
        help="When enabled, uploaded CSVs override latest saved artifacts for matching inputs.",
    )

    with st.expander("Optional Manual Upload Overrides", expanded=False):
        ds_a, ds_b, ds_c, ds_d = st.columns(4)
        with ds_a:
            uploaded_ds_allocation = st.file_uploader("Phase 12 allocation_plan_table.csv", type=["csv"], key="ds_allocation_upload")
            uploaded_ds_paper = st.file_uploader("Phase 12 paper_portfolio_table.csv", type=["csv"], key="ds_paper_upload")
            uploaded_ds_concentration = st.file_uploader("Phase 12 correlation_concentration_table.csv", type=["csv"], key="ds_concentration_upload")
            uploaded_ds_drawdown = st.file_uploader("Phase 12 portfolio_drawdown_stress_table.csv", type=["csv"], key="ds_drawdown_upload")
        with ds_b:
            uploaded_ds_cost = st.file_uploader("Phase 12 cost_slippage_stress_table.csv", type=["csv"], key="ds_cost_upload")
            uploaded_ds_scenarios = st.file_uploader("Phase 12 scenario_analysis_table.csv", type=["csv"], key="ds_scenarios_upload")
            uploaded_ds_risk_budget = st.file_uploader("Phase 12 risk_budget_table.csv", type=["csv"], key="ds_risk_budget_upload")
            uploaded_ds_position = st.file_uploader("Phase 12 position_sizing_table.csv", type=["csv"], key="ds_position_upload")
        with ds_c:
            uploaded_ds_portfolio_warnings = st.file_uploader("Phase 12 warning_table.csv", type=["csv"], key="ds_portfolio_warnings_upload")
            uploaded_ds_capital_blockers = st.file_uploader("Phase 12 capital_blocker_table.csv", type=["csv"], key="ds_capital_blockers_upload")
            uploaded_ds_risk_summary = st.file_uploader("Phase 13 risk_summary_table.csv", type=["csv"], key="ds_risk_summary_upload")
            uploaded_ds_top_risks = st.file_uploader("Phase 13 top_risks_table.csv", type=["csv"], key="ds_top_risks_upload")
        with ds_d:
            uploaded_ds_capital_risks = st.file_uploader("Phase 13 capital_blocking_risks_table.csv", type=["csv"], key="ds_capital_risks_upload")
            uploaded_ds_paper_risks = st.file_uploader("Phase 13 paper_only_risks_table.csv", type=["csv"], key="ds_paper_risks_upload")
            uploaded_ds_risk_matrix = st.file_uploader("Phase 13 asset_horizon_risk_matrix.csv", type=["csv"], key="ds_risk_matrix_upload")
            uploaded_ds_raw_risk = st.file_uploader("Phase 13 raw_warning_evidence.csv", type=["csv"], key="ds_raw_risk_upload")

        ds_e = st.columns(3)
        with ds_e[0]:
            uploaded_ds_warning_groups = st.file_uploader("Phase 13 warning_group_table.csv", type=["csv"], key="ds_warning_groups_upload")
        with ds_e[1]:
            uploaded_ds_risk_status = st.file_uploader("Phase 13 risk_trend_or_status_table.csv", type=["csv"], key="ds_risk_status_upload")
        with ds_e[2]:
            uploaded_ds_next_risk_actions = st.file_uploader("Phase 13 next_risk_actions_table.csv", type=["csv"], key="ds_next_risk_actions_upload")

    sizing_uploaded_overrides = {
        "allocation_plan_table": uploaded_ds_allocation,
        "paper_portfolio_table": uploaded_ds_paper,
        "correlation_concentration_table": uploaded_ds_concentration,
        "portfolio_drawdown_stress_table": uploaded_ds_drawdown,
        "cost_slippage_stress_table": uploaded_ds_cost,
        "scenario_analysis_table": uploaded_ds_scenarios,
        "phase12_risk_budget_table": uploaded_ds_risk_budget,
        "position_sizing_table": uploaded_ds_position,
        "phase12_warning_table": uploaded_ds_portfolio_warnings,
        "phase12_capital_blocker_table": uploaded_ds_capital_blockers,
        "risk_summary_table": uploaded_ds_risk_summary,
        "top_risks_table": uploaded_ds_top_risks,
        "capital_blocking_risks_table": uploaded_ds_capital_risks,
        "paper_only_risks_table": uploaded_ds_paper_risks,
        "warning_group_table": uploaded_ds_warning_groups,
        "asset_horizon_risk_matrix": uploaded_ds_risk_matrix,
        "risk_trend_or_status_table": uploaded_ds_risk_status,
        "next_risk_actions_table": uploaded_ds_next_risk_actions,
        "raw_warning_evidence": uploaded_ds_raw_risk,
    }

    ds_c1, ds_c2, ds_c3, ds_c4 = st.columns(4)
    with ds_c1:
        sizing_assets = st.multiselect("Assets", get_asset_names(), default=get_asset_names(), key="dynamic_sizing_assets")
        sizing_mode = st.selectbox(
            "Portfolio mode",
            ["Conservative", "Balanced Research", "Aggressive Paper Research"],
            index=1,
            key="dynamic_sizing_mode",
        )
    with ds_c2:
        sizing_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        selected_sizing_horizon_labels = st.multiselect("Horizons", sizing_horizon_labels, default=sizing_horizon_labels, key="dynamic_sizing_horizons")
        sizing_horizons = [int(str(label).replace("D", "")) for label in selected_sizing_horizon_labels]
        sizing_allow_increase = st.checkbox("Allow small low-risk paper increase", value=False, key="dynamic_sizing_allow_increase")
    with ds_c3:
        sizing_asset_cap = st.slider("Max single asset exposure (%)", min_value=1.0, max_value=100.0, value=25.0, step=1.0, key="dynamic_sizing_asset_cap")
        sizing_horizon_cap = st.slider("Max single horizon exposure (%)", min_value=1.0, max_value=100.0, value=35.0, step=1.0, key="dynamic_sizing_horizon_cap")
    with ds_c4:
        sizing_portfolio_cap = st.slider("Max portfolio paper exposure (%)", min_value=1.0, max_value=100.0, value=45.0, step=1.0, key="dynamic_sizing_portfolio_cap")
        sizing_drawdown_cap = st.slider("Max drawdown shock budget (%)", min_value=0.5, max_value=50.0, value=10.0, step=0.5, key="dynamic_sizing_drawdown_cap")

    run_dynamic_sizing = st.button("🚀 Run Dynamic Risk Sizing", type="primary")
    if run_dynamic_sizing:
        if not sizing_assets or not sizing_horizons:
            st.error("Select at least one asset and one horizon.")
            st.stop()
        try:
            sizing_report = run_dynamic_risk_sizing(
                use_artifact_store=bool(sizing_use_latest),
                prefer_uploaded=bool(sizing_prefer_uploads),
                uploaded_overrides=sizing_uploaded_overrides,
                assets=sizing_assets,
                horizons=sizing_horizons,
                portfolio_mode=sizing_mode,
                max_single_asset_exposure_pct=float(sizing_asset_cap),
                max_single_horizon_exposure_pct=float(sizing_horizon_cap),
                max_portfolio_paper_exposure_pct=float(sizing_portfolio_cap),
                max_drawdown_shock_pct=float(sizing_drawdown_cap),
                allow_low_risk_increase=bool(sizing_allow_increase),
            )
            st.session_state.dynamic_risk_sizing_report = sizing_report
            st.session_state.dynamic_risk_sizing_settings = sizing_report.settings
            saved_sizing_artifacts = save_phase_artifacts(
                DYNAMIC_RISK_SIZING_PHASE_NAME,
                {
                    "dynamic_sizing_summary_table": sizing_report.dynamic_sizing_summary_table,
                    "dynamic_position_sizing_table": sizing_report.dynamic_position_sizing_table,
                    "risk_multiplier_summary_table": sizing_report.risk_multiplier_summary_table,
                    "risk_multiplier_table": sizing_report.risk_multiplier_table,
                    "cap_adjustment_table": sizing_report.cap_adjustment_table,
                    "zero_size_table": sizing_report.zero_size_table,
                    "optimized_portfolio_table": sizing_report.optimized_portfolio_table,
                    "drawdown_budget_table": sizing_report.drawdown_budget_table,
                    "risk_adjusted_scenarios_table": sizing_report.risk_adjusted_scenarios_table,
                    "next_sizing_actions_table": sizing_report.next_sizing_actions_table,
                    "input_source_table": sizing_report.input_source_table,
                },
                inputs={},
                config=sizing_report.settings,
                warnings=sizing_report.risk_multiplier_table["RiskCategory"].dropna().astype(str).unique().tolist() if not sizing_report.risk_multiplier_table.empty else [],
            )
            st.session_state.artifact_store_last_save = saved_sizing_artifacts
        except Exception as exc:
            st.error(f"Dynamic risk sizing failed: {exc}")
            st.stop()

    sizing_report = st.session_state.dynamic_risk_sizing_report
    if sizing_report is None:
        st.info("Run the optimizer using latest Phase 12/13 artifacts or upload CSV overrides.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('dynamic_risk_sizing_settings') or {}}")
        if not sizing_report.dynamic_sizing_summary_table.empty:
            summary_row = sizing_report.dynamic_sizing_summary_table.iloc[0]
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Overall sizing verdict", str(summary_row.get("OverallSizingVerdict", "")))
            sm2.metric("Starting paper exposure", f"{float(summary_row.get('StartingPaperExposurePct', 0.0)):.2f}%")
            sm3.metric("Optimized paper exposure", f"{float(summary_row.get('OptimizedPaperExposurePct', 0.0)):.2f}%")
            sm4.metric("Real capital exposure", f"{float(summary_row.get('RealCapitalExposurePct', 0.0)):.2f}%")

        st.markdown("### Dynamic Sizing Table")
        st.dataframe(sizing_report.dynamic_position_sizing_table, width="stretch")
        st.markdown("### Input Sources")
        st.dataframe(sizing_report.input_source_table, width="stretch")

        sizing_tabs = st.tabs(
            [
                "Summary",
                "Dynamic Sizing",
                "Multiplier Summary",
                "Risk Multipliers",
                "Cap Adjustments",
                "Zero Size",
                "Optimized Portfolio",
                "Drawdown Budget",
                "Risk Scenarios",
                "Next Actions",
                "Input Sources",
            ]
        )
        sizing_tables = [
            ("Summary", sizing_report.dynamic_sizing_summary_table, "phase14_dynamic_sizing_summary.csv"),
            ("Dynamic Position Sizing", sizing_report.dynamic_position_sizing_table, "phase14_dynamic_position_sizing.csv"),
            ("Risk Multiplier Summary", sizing_report.risk_multiplier_summary_table, "phase14_risk_multiplier_summary.csv"),
            ("Risk Multipliers", sizing_report.risk_multiplier_table, "phase14_risk_multipliers.csv"),
            ("Cap Adjustments", sizing_report.cap_adjustment_table, "phase14_cap_adjustments.csv"),
            ("Zero Size", sizing_report.zero_size_table, "phase14_zero_size.csv"),
            ("Optimized Portfolio", sizing_report.optimized_portfolio_table, "phase14_optimized_portfolio.csv"),
            ("Drawdown Budget", sizing_report.drawdown_budget_table, "phase14_drawdown_budget.csv"),
            ("Risk-Adjusted Scenarios", sizing_report.risk_adjusted_scenarios_table, "phase14_risk_adjusted_scenarios.csv"),
            ("Next Sizing Actions", sizing_report.next_sizing_actions_table, "phase14_next_sizing_actions.csv"),
            ("Input Sources", sizing_report.input_source_table, "phase14_input_sources.csv"),
        ]
        for tab, (title, table, filename) in zip(sizing_tabs, sizing_tables):
            with tab:
                st.markdown(f"### {title}")
                if table.empty:
                    st.info("No rows for this table with the currently loaded evidence.")
                st.dataframe(table, width="stretch")
                st.download_button(
                    f"📥 Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )


# PAGE: MARKET REGIME INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🌍 Market Regime Intelligence":
    st.markdown('<p class="main-header">🌍 Market Regime Intelligence</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Research-only regime diagnostics. This page adjusts simulated paper-sizing context from Phase 14; "
        "it does not create live execution approval or loosen upstream capital gates."
    )

    regime_use_latest = st.checkbox("Use latest saved Phase 12/13/14 artifacts", value=True, key="regime_use_latest")
    regime_prefer_uploads = st.checkbox(
        "Prefer uploaded CSV overrides",
        value=False,
        key="regime_prefer_uploads",
        help="When enabled, uploaded CSVs override matching latest saved artifacts.",
    )
    regime_use_project_market_data = st.checkbox(
        "Use project master market dataset when no market CSV is uploaded",
        value=True,
        key="regime_use_project_market_data",
    )

    with st.expander("Market Data and Optional Artifact Uploads", expanded=False):
        uploaded_regime_market_data = st.file_uploader(
            "Upload market data CSV",
            type=["csv"],
            key="regime_market_data_upload",
            help="Expected to include Date plus asset price columns such as Gold_Close, Silver_Close, Oil_Close, BTC_Close, SP500_Close, and GLD_Close.",
        )
        rg_a, rg_b, rg_c = st.columns(3)
        with rg_a:
            uploaded_rg_dynamic = st.file_uploader("Phase 14 dynamic_position_sizing_table.csv", type=["csv"], key="rg_dynamic_upload")
            uploaded_rg_optimized = st.file_uploader("Phase 14 optimized_portfolio_table.csv", type=["csv"], key="rg_optimized_upload")
            uploaded_rg_multiplier_summary = st.file_uploader("Phase 14 risk_multiplier_summary_table.csv", type=["csv"], key="rg_multiplier_summary_upload")
            uploaded_rg_scenarios = st.file_uploader("Phase 14 risk_adjusted_scenarios_table.csv", type=["csv"], key="rg_scenarios_upload")
        with rg_b:
            uploaded_rg_risk_matrix = st.file_uploader("Phase 13 asset_horizon_risk_matrix.csv", type=["csv"], key="rg_risk_matrix_upload")
            uploaded_rg_risk_summary = st.file_uploader("Phase 13 risk_summary_table.csv", type=["csv"], key="rg_risk_summary_upload")
            uploaded_rg_top_risks = st.file_uploader("Phase 13 top_risks_table.csv", type=["csv"], key="rg_top_risks_upload")
            uploaded_rg_warning_groups = st.file_uploader("Phase 13 warning_group_table.csv", type=["csv"], key="rg_warning_groups_upload")
        with rg_c:
            uploaded_rg_allocation = st.file_uploader("Phase 12 allocation_plan_table.csv", type=["csv"], key="rg_allocation_upload")
            uploaded_rg_paper = st.file_uploader("Phase 12 paper_portfolio_table.csv", type=["csv"], key="rg_paper_upload")
            uploaded_rg_drawdown = st.file_uploader("Phase 12 portfolio_drawdown_stress_table.csv", type=["csv"], key="rg_drawdown_upload")
            uploaded_rg_cost = st.file_uploader("Phase 12 cost_slippage_stress_table.csv", type=["csv"], key="rg_cost_upload")
            uploaded_rg_concentration = st.file_uploader("Phase 12 correlation_concentration_table.csv", type=["csv"], key="rg_concentration_upload")

    regime_uploaded_overrides = {
        "dynamic_position_sizing_table": uploaded_rg_dynamic,
        "optimized_portfolio_table": uploaded_rg_optimized,
        "risk_multiplier_summary_table": uploaded_rg_multiplier_summary,
        "risk_adjusted_scenarios_table": uploaded_rg_scenarios,
        "asset_horizon_risk_matrix": uploaded_rg_risk_matrix,
        "risk_summary_table": uploaded_rg_risk_summary,
        "top_risks_table": uploaded_rg_top_risks,
        "warning_group_table": uploaded_rg_warning_groups,
        "allocation_plan_table": uploaded_rg_allocation,
        "paper_portfolio_table": uploaded_rg_paper,
        "portfolio_drawdown_stress_table": uploaded_rg_drawdown,
        "cost_slippage_stress_table": uploaded_rg_cost,
        "correlation_concentration_table": uploaded_rg_concentration,
    }

    rg_c1, rg_c2, rg_c3 = st.columns(3)
    with rg_c1:
        regime_assets = st.multiselect("Assets", get_asset_names(), default=get_asset_names(), key="regime_assets")
    with rg_c2:
        regime_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        selected_regime_horizon_labels = st.multiselect(
            "Horizons",
            regime_horizon_labels,
            default=regime_horizon_labels,
            key="regime_horizons",
        )
        regime_horizons = [int(str(label).replace("D", "")) for label in selected_regime_horizon_labels]
    with rg_c3:
        regime_allow_increase = st.checkbox(
            "Allow small favorable-regime paper-size increase",
            value=False,
            key="regime_allow_small_increase",
            help="Default is off, so regime adjustment can keep or reduce Phase 14 paper weights.",
        )

    run_regime = st.button("🚀 Run Market Regime Intelligence", type="primary")
    if run_regime:
        if not regime_assets or not regime_horizons:
            st.error("Select at least one asset and one horizon.")
            st.stop()
        try:
            market_data_input = pd.read_csv(uploaded_regime_market_data) if uploaded_regime_market_data is not None else None
            regime_report = run_market_regime_intelligence(
                market_data=market_data_input,
                use_project_market_data=bool(regime_use_project_market_data),
                use_artifact_store=bool(regime_use_latest),
                prefer_uploaded=bool(regime_prefer_uploads),
                uploaded_overrides=regime_uploaded_overrides,
                assets=regime_assets,
                horizons=regime_horizons,
                allow_small_paper_increase=bool(regime_allow_increase),
                autosave=True,
            )
            st.session_state.market_regime_intelligence_report = regime_report
            st.session_state.market_regime_intelligence_settings = regime_report.settings
            st.session_state.artifact_store_last_save = regime_report.saved_artifacts
        except Exception as exc:
            st.error(f"Market regime intelligence failed: {exc}")
            st.stop()

    regime_report = st.session_state.market_regime_intelligence_report
    if regime_report is None:
        st.info("Run the regime analysis using the project market dataset, uploaded market data, and latest saved research artifacts.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('market_regime_intelligence_settings') or {}}")
        if not regime_report.regime_summary_table.empty:
            regime_summary = regime_report.regime_summary_table.iloc[0]
            rsm1, rsm2, rsm3, rsm4, rsm5 = st.columns(5)
            rsm1.metric("Overall regime", str(regime_summary.get("OverallMarketRegime", "")))
            rsm2.metric("Market stress", str(regime_summary.get("MarketStressLevel", "")))
            rsm3.metric("Asset stress", str(regime_summary.get("AssetRegimeStressLevel", "")))
            rsm4.metric("Risk sentiment", str(regime_summary.get("RiskSentimentRegime", "")))
            rsm5.metric("Regime confidence", f"{float(regime_summary.get('RegimeConfidence', 0.0)):.2f}%")
            st.caption(f"Volatility regime: {regime_summary.get('VolatilityRegime', '')}")
            st.info(str(regime_summary.get("RecommendedResearchPosture", "")))

        st.markdown("### Regime-Adjusted Paper Sizing")
        st.dataframe(regime_report.regime_adjusted_sizing_table, width="stretch")

        regime_tabs = st.tabs(
            [
                "Summary",
                "Asset Regimes",
                "Asset Horizon Regimes",
                "Regime Factors",
                "Transitions",
                "Regime Risks",
                "Adjusted Sizing",
                "Next Actions",
                "Input Sources",
                "Artifact Inputs",
            ]
        )
        regime_tables = [
            ("Summary", regime_report.regime_summary_table, "phase15_regime_summary.csv"),
            ("Asset Regimes", regime_report.asset_regime_table, "phase15_asset_regime.csv"),
            ("Asset Horizon Regimes", regime_report.asset_horizon_regime_table, "phase15_asset_horizon_regime.csv"),
            ("Regime Factors", regime_report.regime_factor_table, "phase15_regime_factors.csv"),
            ("Regime Transitions", regime_report.regime_transition_table, "phase15_regime_transitions.csv"),
            ("Regime Risks", regime_report.regime_risk_table, "phase15_regime_risks.csv"),
            ("Regime-Adjusted Sizing", regime_report.regime_adjusted_sizing_table, "phase15_regime_adjusted_sizing.csv"),
            ("Next Regime Actions", regime_report.next_regime_actions_table, "phase15_next_regime_actions.csv"),
            ("Input Sources", regime_report.regime_input_sources_table, "phase15_input_sources.csv"),
            ("Artifact Inputs", regime_report.input_source_table, "phase15_artifact_input_sources.csv"),
        ]
        for tab, (title, table, filename) in zip(regime_tabs, regime_tables):
            with tab:
                st.markdown(f"### {title}")
                if table.empty:
                    st.info("No rows for this table with the currently loaded evidence.")
                st.dataframe(table, width="stretch")
                st.download_button(
                    f"📥 Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )


# PAGE: STRATEGY BENCHMARK ARENA
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🏁 Strategy Benchmark Arena":
    st.markdown('<p class="main-header">🏁 Strategy Benchmark Arena</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Research benchmarking only. This arena compares model/risk snapshots with simple baselines, "
        "keeps weak results visible, and does not change real-capital gates."
    )

    benchmark_use_latest = st.checkbox("Use latest saved Phase 12/13/14/15 artifacts", value=True, key="benchmark_use_latest")
    benchmark_prefer_uploads = st.checkbox(
        "Prefer uploaded CSV overrides",
        value=False,
        key="benchmark_prefer_uploads",
        help="When enabled, uploaded CSVs override matching latest saved artifacts.",
    )
    benchmark_use_project_market_data = st.checkbox(
        "Use project master market dataset when no market CSV is uploaded",
        value=True,
        key="benchmark_use_project_market_data",
    )

    with st.expander("Market Data and Optional Artifact Uploads", expanded=False):
        uploaded_benchmark_market_data = st.file_uploader(
            "Upload market data CSV",
            type=["csv"],
            key="benchmark_market_data_upload",
            help="Expected to include Date plus configured asset price columns.",
        )
        bm_a, bm_b, bm_c, bm_d = st.columns(4)
        with bm_a:
            uploaded_bm_phase15_sizing = st.file_uploader("Phase 15 regime_adjusted_sizing_table.csv", type=["csv"], key="bm_phase15_sizing_upload")
            uploaded_bm_phase15_horizon = st.file_uploader("Phase 15 asset_horizon_regime_table.csv", type=["csv"], key="bm_phase15_horizon_upload")
            uploaded_bm_phase15_risks = st.file_uploader("Phase 15 regime_risk_table.csv", type=["csv"], key="bm_phase15_risks_upload")
            uploaded_bm_phase15_summary = st.file_uploader("Phase 15 regime_summary_table.csv", type=["csv"], key="bm_phase15_summary_upload")
        with bm_b:
            uploaded_bm_phase14_dynamic = st.file_uploader("Phase 14 dynamic_position_sizing_table.csv", type=["csv"], key="bm_phase14_dynamic_upload")
            uploaded_bm_phase14_portfolio = st.file_uploader("Phase 14 optimized_portfolio_table.csv", type=["csv"], key="bm_phase14_portfolio_upload")
            uploaded_bm_phase14_summary = st.file_uploader("Phase 14 dynamic_sizing_summary_table.csv", type=["csv"], key="bm_phase14_summary_upload")
        with bm_c:
            uploaded_bm_phase13_matrix = st.file_uploader("Phase 13 asset_horizon_risk_matrix.csv", type=["csv"], key="bm_phase13_matrix_upload")
            uploaded_bm_phase13_top = st.file_uploader("Phase 13 top_risks_table.csv", type=["csv"], key="bm_phase13_top_upload")
            uploaded_bm_phase13_groups = st.file_uploader("Phase 13 warning_group_table.csv", type=["csv"], key="bm_phase13_groups_upload")
        with bm_d:
            uploaded_bm_phase12_allocation = st.file_uploader("Phase 12 allocation_plan_table.csv", type=["csv"], key="bm_phase12_allocation_upload")
            uploaded_bm_phase12_paper = st.file_uploader("Phase 12 paper_portfolio_table.csv", type=["csv"], key="bm_phase12_paper_upload")
            uploaded_bm_phase12_drawdown = st.file_uploader("Phase 12 portfolio_drawdown_stress_table.csv", type=["csv"], key="bm_phase12_drawdown_upload")
            uploaded_bm_phase12_cost = st.file_uploader("Phase 12 cost_slippage_stress_table.csv", type=["csv"], key="bm_phase12_cost_upload")
            uploaded_bm_phase10_plan = st.file_uploader("Phase 10 ranked_asset_horizon_plan.csv", type=["csv"], key="bm_phase10_plan_upload")

    benchmark_uploaded_overrides = {
        "regime_adjusted_sizing_table": uploaded_bm_phase15_sizing,
        "asset_horizon_regime_table": uploaded_bm_phase15_horizon,
        "regime_risk_table": uploaded_bm_phase15_risks,
        "regime_summary_table": uploaded_bm_phase15_summary,
        "dynamic_position_sizing_table": uploaded_bm_phase14_dynamic,
        "optimized_portfolio_table": uploaded_bm_phase14_portfolio,
        "dynamic_sizing_summary_table": uploaded_bm_phase14_summary,
        "asset_horizon_risk_matrix": uploaded_bm_phase13_matrix,
        "top_risks_table": uploaded_bm_phase13_top,
        "warning_group_table": uploaded_bm_phase13_groups,
        "allocation_plan_table": uploaded_bm_phase12_allocation,
        "paper_portfolio_table": uploaded_bm_phase12_paper,
        "portfolio_drawdown_stress_table": uploaded_bm_phase12_drawdown,
        "cost_slippage_stress_table": uploaded_bm_phase12_cost,
        "ranked_asset_horizon_plan": uploaded_bm_phase10_plan,
    }

    bmc1, bmc2, bmc3, bmc4 = st.columns(4)
    with bmc1:
        benchmark_assets = st.multiselect("Assets", get_asset_names(), default=get_asset_names(), key="benchmark_assets")
        benchmark_short_ma = st.number_input("Short MA", min_value=5, max_value=100, value=20, step=1, key="benchmark_short_ma")
    with bmc2:
        benchmark_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        selected_benchmark_horizon_labels = st.multiselect("Horizons", benchmark_horizon_labels, default=benchmark_horizon_labels, key="benchmark_horizons")
        benchmark_horizons = [int(str(label).replace("D", "")) for label in selected_benchmark_horizon_labels]
        benchmark_long_ma = st.number_input("Long MA", min_value=10, max_value=250, value=50, step=1, key="benchmark_long_ma")
    with bmc3:
        benchmark_momentum = st.number_input("Momentum lookback", min_value=5, max_value=250, value=20, step=1, key="benchmark_momentum")
        benchmark_mean_window = st.number_input("Mean reversion window", min_value=5, max_value=250, value=20, step=1, key="benchmark_mean_window")
    with bmc4:
        benchmark_cost_bps = st.number_input("Cost bps", min_value=0.0, max_value=200.0, value=10.0, step=1.0, key="benchmark_cost_bps")
        benchmark_slippage_bps = st.number_input("Slippage bps", min_value=0.0, max_value=200.0, value=5.0, step=1.0, key="benchmark_slippage_bps")
        benchmark_random_sims = st.number_input("Random simulations", min_value=10, max_value=500, value=100, step=10, key="benchmark_random_sims")

    benchmark_cost_scenarios = st.multiselect(
        "Cost sensitivity scenarios",
        [0, 5, 10, 25, 50],
        default=[0, 5, 10, 25, 50],
        key="benchmark_cost_scenarios",
    )

    run_benchmark = st.button("🚀 Run Strategy Benchmark Arena", type="primary")
    if run_benchmark:
        if not benchmark_assets or not benchmark_horizons:
            st.error("Select at least one asset and one horizon.")
            st.stop()
        try:
            market_data_input = pd.read_csv(uploaded_benchmark_market_data) if uploaded_benchmark_market_data is not None else None
            benchmark_report = run_strategy_benchmark_arena(
                market_data=market_data_input,
                use_project_market_data=bool(benchmark_use_project_market_data),
                use_artifact_store=bool(benchmark_use_latest),
                prefer_uploaded=bool(benchmark_prefer_uploads),
                uploaded_overrides=benchmark_uploaded_overrides,
                assets=benchmark_assets,
                horizons=benchmark_horizons,
                short_ma=int(benchmark_short_ma),
                long_ma=int(benchmark_long_ma),
                momentum_lookback=int(benchmark_momentum),
                mean_reversion_window=int(benchmark_mean_window),
                cost_bps=float(benchmark_cost_bps),
                slippage_bps=float(benchmark_slippage_bps),
                cost_scenarios_bps=benchmark_cost_scenarios,
                random_simulations=int(benchmark_random_sims),
                autosave=True,
            )
            st.session_state.strategy_benchmark_arena_report = benchmark_report
            st.session_state.strategy_benchmark_arena_settings = benchmark_report.settings
            st.session_state.artifact_store_last_save = benchmark_report.saved_artifacts
        except Exception as exc:
            st.error(f"Strategy benchmark arena failed: {exc}")
            st.stop()

    benchmark_report = st.session_state.strategy_benchmark_arena_report
    if benchmark_report is None:
        st.info("Run the arena to compare model/risk snapshots with simple, time-safe baselines.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('strategy_benchmark_arena_settings') or {}}")
        if not benchmark_report.benchmark_summary_table.empty:
            summary_row = benchmark_report.benchmark_summary_table.iloc[0]
            bs1, bs2, bs3, bs4 = st.columns(4)
            bs1.metric("Benchmark verdict", str(summary_row.get("BenchmarkVerdict", "")))
            bs2.metric("Overall winner", str(summary_row.get("OverallWinner", "")))
            bs3.metric("Model beats hold-only", str(summary_row.get("ModelBeatsHoldOnly", "")))
            bs4.metric("Evidence quality", str(summary_row.get("EvidenceQuality", "")))
            st.info(str(summary_row.get("MainReason", "")))

        benchmark_tabs = st.tabs(
            [
                "Summary",
                "Historical Baselines",
                "Full Leaderboard",
                "Snapshot Impact",
                "Assets",
                "Asset Horizons",
                "Benchmark Dominance",
                "Model Strength",
                "Cost Sensitivity",
                "Random Baseline",
                "Return Sanity",
                "Leakage Checks",
                "Warnings",
                "Next Actions",
                "Input Sources",
                "Artifact Inputs",
            ]
        )
        historical_baseline_table = benchmark_report.strategy_leaderboard_table[
            benchmark_report.strategy_leaderboard_table["ComparableHistorical"].eq(True)
        ].copy() if "ComparableHistorical" in benchmark_report.strategy_leaderboard_table.columns else benchmark_report.strategy_leaderboard_table.copy()
        benchmark_tables = [
            ("Benchmark Summary", benchmark_report.benchmark_summary_table, "phase16_benchmark_summary.csv"),
            ("Historical Baseline Leaderboard", historical_baseline_table, "phase16_historical_baseline_leaderboard.csv"),
            ("Strategy Leaderboard", benchmark_report.strategy_leaderboard_table, "phase16_strategy_leaderboard.csv"),
            ("Snapshot Model/Risk Impact", benchmark_report.snapshot_model_impact_table, "phase16_snapshot_model_impact.csv"),
            ("Asset Benchmark", benchmark_report.asset_benchmark_table, "phase16_asset_benchmark.csv"),
            ("Asset Horizon Benchmark", benchmark_report.asset_horizon_benchmark_table, "phase16_asset_horizon_benchmark.csv"),
            ("Benchmark Dominance", benchmark_report.benchmark_dominance_table, "phase16_benchmark_dominance.csv"),
            ("Model Strength", benchmark_report.model_strength_table, "phase16_model_strength.csv"),
            ("Cost Sensitivity", benchmark_report.cost_sensitivity_table, "phase16_cost_sensitivity.csv"),
            ("Random Baseline", benchmark_report.random_baseline_table, "phase16_random_baseline.csv"),
            ("Return Sanity Checks", benchmark_report.return_sanity_check_table, "phase16_return_sanity_checks.csv"),
            ("Leakage Checks", benchmark_report.leakage_check_table, "phase16_leakage_checks.csv"),
            ("Benchmark Warnings", benchmark_report.benchmark_warning_table, "phase16_benchmark_warnings.csv"),
            ("Next Benchmark Actions", benchmark_report.next_benchmark_actions_table, "phase16_next_benchmark_actions.csv"),
            ("Input Sources", benchmark_report.benchmark_input_sources_table, "phase16_input_sources.csv"),
            ("Artifact Inputs", benchmark_report.artifact_input_source_table, "phase16_artifact_input_sources.csv"),
        ]
        for tab, (title, table, filename) in zip(benchmark_tabs, benchmark_tables):
            with tab:
                st.markdown(f"### {title}")
                if table.empty:
                    st.info("No rows for this table with the currently loaded evidence.")
                st.dataframe(table, width="stretch")
                st.download_button(
                    f"📥 Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )


    st.markdown("---")
    st.markdown("## Historical Replay Edge Audit")
    st.warning(
        "Phase 18 compares Phase 17 historical replay exports with simple baselines. "
        "Proxy replay remains proxy evidence, not true historical trained-model proof, and real capital remains blocked."
    )

    audit_use_latest = st.checkbox("Use latest saved Phase 17 replay artifacts", value=True, key="audit_use_latest")
    audit_prefer_uploads = st.checkbox("Prefer uploaded Phase 17 CSV overrides", value=False, key="audit_prefer_uploads")
    audit_use_project_market_data = st.checkbox("Use project master market dataset for audit baselines", value=True, key="audit_use_project_market_data")

    with st.expander("Historical Replay Audit Inputs", expanded=False):
        uploaded_audit_market_data = st.file_uploader("Audit market data CSV", type=["csv"], key="audit_market_data_upload")
        au_a, au_b, au_c = st.columns(3)
        with au_a:
            uploaded_audit_replay_export = st.file_uploader("phase17_phase16_replay_export.csv", type=["csv"], key="audit_replay_export_upload")
            uploaded_audit_replay_summary = st.file_uploader("phase17_replay_summary.csv", type=["csv"], key="audit_replay_summary_upload")
            uploaded_audit_quality = st.file_uploader("phase17_replay_quality_checks.csv", type=["csv"], key="audit_quality_upload")
        with au_b:
            uploaded_audit_ready = st.file_uploader("phase17_replay_benchmark_ready.csv", type=["csv"], key="audit_ready_upload")
            uploaded_audit_performance = st.file_uploader("phase17_historical_replay_performance.csv", type=["csv"], key="audit_performance_upload")
            uploaded_audit_matrix = st.file_uploader("phase17_replay_asset_horizon_matrix.csv", type=["csv"], key="audit_matrix_upload")
        with au_c:
            uploaded_audit_cap = st.file_uploader("phase17_replay_exposure_cap.csv", type=["csv"], key="audit_cap_upload")
            uploaded_audit_warnings = st.file_uploader("phase17_replay_warnings.csv", type=["csv"], key="audit_warnings_upload")

    audit_uploaded_overrides = {
        "phase16_replay_export_table": uploaded_audit_replay_export,
        "replay_summary_table": uploaded_audit_replay_summary,
        "replay_quality_checks": uploaded_audit_quality,
        "replay_benchmark_ready_table": uploaded_audit_ready,
        "historical_replay_performance": uploaded_audit_performance,
        "replay_asset_horizon_matrix": uploaded_audit_matrix,
        "replay_exposure_cap_table": uploaded_audit_cap,
        "replay_warnings_table": uploaded_audit_warnings,
    }

    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        audit_assets = st.multiselect("Audit assets", get_asset_names(), default=get_asset_names(), key="audit_assets")
    with ac2:
        audit_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        selected_audit_horizon_labels = st.multiselect("Audit horizons", audit_horizon_labels, default=audit_horizon_labels, key="audit_horizons")
        audit_horizons = [int(str(label).replace("D", "")) for label in selected_audit_horizon_labels]
    with ac3:
        audit_cost_bps = st.number_input("Audit cost bps", min_value=0.0, max_value=200.0, value=10.0, step=1.0, key="audit_cost_bps")
        audit_slippage_bps = st.number_input("Audit slippage bps", min_value=0.0, max_value=200.0, value=5.0, step=1.0, key="audit_slippage_bps")
    with ac4:
        audit_random_sims = st.number_input("Audit random simulations", min_value=10, max_value=500, value=100, step=10, key="audit_random_sims")
        audit_min_trades = st.number_input("Min replay trades", min_value=1, max_value=100, value=3, step=1, key="audit_min_trades")

    run_audit = st.button("Run Historical Replay Edge Audit", type="primary")
    if run_audit:
        if not audit_assets or not audit_horizons:
            st.error("Select at least one audit asset and horizon.")
            st.stop()
        try:
            audit_market_data_input = pd.read_csv(uploaded_audit_market_data) if uploaded_audit_market_data is not None else None
            audit_report = run_replay_benchmark_audit(
                market_data=audit_market_data_input,
                use_project_market_data=bool(audit_use_project_market_data),
                use_artifact_store=bool(audit_use_latest),
                prefer_uploaded=bool(audit_prefer_uploads),
                uploaded_overrides=audit_uploaded_overrides,
                assets=audit_assets,
                horizons=audit_horizons,
                cost_bps=float(audit_cost_bps),
                slippage_bps=float(audit_slippage_bps),
                random_simulations=int(audit_random_sims),
                min_trades=int(audit_min_trades),
                autosave=True,
            )
            st.session_state.replay_benchmark_audit_report = audit_report
            st.session_state.replay_benchmark_audit_settings = audit_report.settings
            st.session_state.artifact_store_last_save = audit_report.saved_artifacts
        except Exception as exc:
            st.error(f"Historical replay edge audit failed: {exc}")
            st.stop()

    audit_report = st.session_state.replay_benchmark_audit_report
    if audit_report is None:
        st.info("Run Phase 18 after generating a Phase 17 replay export.")
    else:
        st.caption(f"Artifact phase: {REPLAY_BENCHMARK_AUDIT_PHASE_NAME}")
        st.caption(f"Last run settings: {st.session_state.get('replay_benchmark_audit_settings') or {}}")
        if not audit_report.replay_benchmark_summary_table.empty:
            audit_summary = audit_report.replay_benchmark_summary_table.iloc[0]
            ar1, ar2, ar3, ar4 = st.columns(4)
            ar1.metric("Replay benchmark verdict", str(audit_summary.get("ReplayBenchmarkVerdict", "")))
            ar2.metric("Replay source", str(audit_summary.get("ReplaySource", "")))
            ar3.metric("Proxy beats no exposure", str(audit_summary.get("ProxyBeatsNoExposure", "")))
            ar4.metric("Proxy beats random median", str(audit_summary.get("ProxyBeatsRandomMedian", "")))
            st.info(str(audit_summary.get("MainReason", "")))
            st.warning(str(audit_summary.get("MainLimitation", "")))

        audit_tabs = st.tabs(
            [
                "Summary",
                "Leaderboard",
                "Asset Edge",
                "Asset Horizon Edge",
                "Dominance Failures",
                "Strength",
                "Random",
                "Cost",
                "Drawdown",
                "Quality Gates",
                "Readiness",
                "Next Actions",
                "Input Sources",
            ]
        )
        audit_tables = [
            ("Replay Benchmark Summary", audit_report.replay_benchmark_summary_table, "phase18_replay_benchmark_summary.csv"),
            ("Replay vs Baseline Leaderboard", audit_report.replay_vs_baseline_leaderboard, "phase18_replay_vs_baseline_leaderboard.csv"),
            ("Replay Asset Edge", audit_report.replay_asset_edge_table, "phase18_replay_asset_edge.csv"),
            ("Replay Asset Horizon Edge", audit_report.replay_asset_horizon_edge_table, "phase18_replay_asset_horizon_edge.csv"),
            ("Replay Dominance Failures", audit_report.replay_dominance_failures_table, "phase18_replay_dominance_failures.csv"),
            ("Replay Strength", audit_report.replay_strength_table, "phase18_replay_strength.csv"),
            ("Replay Random Comparison", audit_report.replay_random_comparison_table, "phase18_replay_random_comparison.csv"),
            ("Replay Cost Robustness", audit_report.replay_cost_robustness_table, "phase18_replay_cost_robustness.csv"),
            ("Replay Drawdown Comparison", audit_report.replay_drawdown_comparison_table, "phase18_replay_drawdown_comparison.csv"),
            ("Replay Quality Gates", audit_report.replay_quality_gate_table, "phase18_replay_quality_gates.csv"),
            ("Replay Real-Capital Readiness", audit_report.replay_real_capital_readiness_table, "phase18_replay_real_capital_readiness.csv"),
            ("Replay Next Actions", audit_report.replay_next_actions_table, "phase18_replay_next_actions.csv"),
            ("Replay Benchmark Input Sources", audit_report.replay_benchmark_input_sources_table, "phase18_replay_benchmark_input_sources.csv"),
        ]
        for tab, (title, table, filename) in zip(audit_tabs, audit_tables):
            with tab:
                st.markdown(f"### {title}")
                if table.empty:
                    st.info("No rows for this table with the currently loaded evidence.")
                st.dataframe(table, width="stretch")
                st.download_button(
                    f"Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )


# PAGE: SIGNAL POLICY & EDGE REPAIR LAB
# ----------------------------------------------------------------

elif page == "Phase 19: Signal Policy & Edge Repair Lab":
    st.markdown('<p class="main-header">Signal Policy &amp; Edge Repair Lab</p>', unsafe_allow_html=True)
    st.markdown("---")
    render_research_disclaimer()
    render_blocked_capital_banner()
    st.info("Compare every policy with serious baselines; weak and dominated policies remain visible.")
    render_glossary_expander(glossary_entries(["Baseline", "Cost drag", "Drawdown", "Benchmark dominated"]))

    st.warning(
        "Research and paper-policy diagnostics only. Weak policies remain visible, "
        "and real capital remains blocked."
    )

    policy_lab_use_latest = st.checkbox(
        "Use latest saved Phase 14/15/17/18 artifacts",
        value=True,
        key="policy_lab_use_latest",
    )
    policy_lab_use_project_market_data = st.checkbox(
        "Use project master market dataset when no market CSV is uploaded",
        value=True,
        key="policy_lab_use_project_market_data",
    )
    uploaded_policy_lab_market_data = st.file_uploader(
        "Upload market data CSV",
        type=["csv"],
        key="policy_lab_market_data_upload",
        help="Expected to include Date and the configured asset price columns.",
    )

    pl1, pl2, pl3, pl4 = st.columns(4)
    with pl1:
        policy_lab_assets = st.multiselect(
            "Assets",
            get_asset_names(),
            default=get_asset_names(),
            key="policy_lab_assets",
        )
    with pl2:
        policy_lab_horizons = st.multiselect(
            "Horizons",
            list(POLICY_LAB_HORIZONS),
            default=list(POLICY_LAB_HORIZONS),
            format_func=lambda value: f"{int(value)}D",
            key="policy_lab_horizons",
        )
    with pl3:
        policy_lab_cost_bps = st.number_input(
            "Cost bps",
            min_value=0.0,
            max_value=200.0,
            value=10.0,
            step=1.0,
            key="policy_lab_cost_bps",
        )
        policy_lab_slippage_bps = st.number_input(
            "Slippage bps",
            min_value=0.0,
            max_value=200.0,
            value=5.0,
            step=1.0,
            key="policy_lab_slippage_bps",
        )
    with pl4:
        policy_lab_train_fraction = st.slider(
            "In-sample fraction",
            min_value=0.50,
            max_value=0.80,
            value=0.60,
            step=0.05,
            key="policy_lab_train_fraction",
        )
        policy_lab_min_trades = st.number_input(
            "Minimum trades",
            min_value=1,
            max_value=100,
            value=3,
            step=1,
            key="policy_lab_min_trades",
        )

    plc1, plc2 = st.columns(2)
    with plc1:
        policy_lab_cost_scenarios = st.multiselect(
            "Cost sensitivity scenarios",
            [0, 5, 10, 25, 50],
            default=[0, 5, 10, 25, 50],
            key="policy_lab_cost_scenarios",
        )
    with plc2:
        policy_lab_random_simulations = st.number_input(
            "Random baseline simulations",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            key="policy_lab_random_simulations",
        )

    run_policy_lab = st.button("Run Signal Policy & Edge Repair Lab", type="primary")
    if run_policy_lab:
        if not policy_lab_assets or not policy_lab_horizons:
            st.error("Select at least one asset and one horizon.")
            st.stop()
        try:
            policy_lab_market_data = pd.read_csv(uploaded_policy_lab_market_data) if uploaded_policy_lab_market_data is not None else None
            policy_lab_report = run_signal_policy_edge_lab(
                market_data=policy_lab_market_data,
                use_project_market_data=bool(policy_lab_use_project_market_data),
                use_artifact_store=bool(policy_lab_use_latest),
                assets=policy_lab_assets,
                horizons=policy_lab_horizons,
                cost_bps=float(policy_lab_cost_bps),
                slippage_bps=float(policy_lab_slippage_bps),
                cost_scenarios_bps=policy_lab_cost_scenarios,
                random_simulations=int(policy_lab_random_simulations),
                train_fraction=float(policy_lab_train_fraction),
                min_trades=int(policy_lab_min_trades),
                autosave=True,
            )
            st.session_state.signal_policy_edge_lab_report = policy_lab_report
            st.session_state.signal_policy_edge_lab_settings = policy_lab_report.settings
            st.session_state.artifact_store_last_save = policy_lab_report.saved_artifacts
        except Exception as exc:
            st.error(f"Signal policy edge lab failed: {exc}")
            st.stop()

    policy_lab_report = st.session_state.signal_policy_edge_lab_report
    if policy_lab_report is None:
        st.info("Run this lab to compare time-safe policies with serious baselines.")
    else:
        st.caption("Artifact source: saved signal-policy evidence")
        st.caption(f"Last run settings: {st.session_state.get('signal_policy_edge_lab_settings') or {}}")
        if not policy_lab_report.policy_lab_summary_table.empty:
            policy_summary = policy_lab_report.policy_lab_summary_table.iloc[0]
            ps1, ps2, ps3, ps4 = st.columns(4)
            ps1.metric("Verdict", str(policy_summary.get("PolicyLabVerdict", "")))
            ps2.metric("Best policy", str(policy_summary.get("BestPolicy", "")))
            ps3.metric("Best asset / horizon", f"{policy_summary.get('BestAsset', '')} {policy_summary.get('BestHorizon', '')}D")
            ps4.metric("Broad edge found", str(policy_summary.get("BroadEdgeFound", False)))
            st.info(str(policy_summary.get("MainReason", "")))
            st.warning(str(policy_summary.get("MainLimitation", "")))

        policy_lab_tabs = st.tabs(
            [
                "Summary",
                "Policy Leaderboard",
                "Dominance Failures",
                "Cost Sensitivity",
                "Overfit Audit",
                "Drawdown",
                "Turnover",
                "Quality Gates",
                "Recommendations",
            ]
        )
        policy_lab_tables = [
            ("Policy Lab Summary", policy_lab_report.policy_lab_summary_table, "phase19_policy_lab_summary.csv"),
            ("Policy Leaderboard", policy_lab_report.policy_leaderboard_table, "phase19_policy_leaderboard.csv"),
            ("Policy Dominance Failures", policy_lab_report.policy_dominance_failures_table, "phase19_policy_dominance_failures.csv"),
            ("Policy Cost Sensitivity", policy_lab_report.policy_cost_sensitivity_table, "phase19_policy_cost_sensitivity.csv"),
            ("Policy Overfit Audit", policy_lab_report.policy_overfit_audit_table, "phase19_policy_overfit_audit.csv"),
            ("Policy Drawdown", policy_lab_report.policy_drawdown_table, "phase19_policy_drawdown.csv"),
            ("Policy Turnover", policy_lab_report.policy_turnover_table, "phase19_policy_turnover.csv"),
            ("Policy Quality Gates", policy_lab_report.policy_quality_gates_table, "phase19_policy_quality_gates.csv"),
            ("Policy Recommendations", policy_lab_report.policy_recommendation_table, "phase19_policy_recommendations.csv"),
        ]
        for tab, (title, table, filename) in zip(policy_lab_tabs, policy_lab_tables):
            with tab:
                st.markdown(f"### {title}")
                if table.empty:
                    st.info("No rows are available for this table with the current evidence.")
                st.dataframe(table, width="stretch")
                st.download_button(
                    f"Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )


# PAGE: TRUE HISTORICAL ML REPLAY
# ----------------------------------------------------------------

elif page == "Phase 20: True Historical ML Replay":
    st.markdown('<p class="main-header">Walk-Forward ML Replay</p>', unsafe_allow_html=True)
    st.markdown("---")
    render_research_disclaimer()
    render_blocked_capital_banner()
    st.info("This replay retrains chronologically. A model demonstrates research edge only when it beats the best baseline out of sample.")
    render_glossary_expander(glossary_entries(["Walk-forward validation", "Leakage", "Baseline", "Drawdown", "Cost drag"]))

    st.warning(
        "Research-only historical walk-forward replay. This page retrains inside each historical window, "
        "keeps pending and rejected evidence visible, and leaves real capital blocked."
    )
    st.info(
        "This is separate from the Phase 17 proxy replay. No replay runs automatically; choose bounded settings and start it explicitly."
    )

    true_ml_use_project_data = st.checkbox(
        "Use project master market dataset when no CSV is uploaded",
        value=True,
        key="true_ml_use_project_data",
    )
    uploaded_true_ml_market_data = st.file_uploader(
        "Upload historical market data CSV",
        type=["csv"],
        key="true_ml_market_data_upload",
        help="Expected to include Date and configured asset close-price columns.",
    )

    tm1, tm2, tm3, tm4 = st.columns(4)
    with tm1:
        true_ml_assets = st.multiselect(
            "Assets",
            get_asset_names(),
            default=get_asset_names(),
            key="true_ml_assets",
        )
        true_ml_model = st.selectbox(
            "Model",
            list(TRUE_ML_MODEL_CHOICES),
            index=0,
            key="true_ml_model",
        )
    with tm2:
        true_ml_horizons = st.multiselect(
            "Horizons",
            list(TRUE_ML_HORIZONS),
            default=list(TRUE_ML_HORIZONS),
            format_func=lambda value: f"{int(value)}D",
            key="true_ml_horizons",
        )
        true_ml_max_windows = st.number_input(
            "Maximum windows per asset/horizon",
            min_value=1,
            max_value=100,
            value=8,
            step=1,
            key="true_ml_max_windows",
        )
    with tm3:
        true_ml_min_train_rows = st.number_input(
            "Minimum training rows",
            min_value=20,
            max_value=5000,
            value=120,
            step=10,
            key="true_ml_min_train_rows",
        )
        true_ml_step_size = st.number_input(
            "Walk-forward step size",
            min_value=1,
            max_value=500,
            value=20,
            step=1,
            key="true_ml_step_size",
        )
    with tm4:
        true_ml_cost_bps = st.number_input(
            "Cost bps",
            min_value=0.0,
            max_value=200.0,
            value=10.0,
            step=1.0,
            key="true_ml_cost_bps",
        )
        true_ml_slippage_bps = st.number_input(
            "Slippage bps",
            min_value=0.0,
            max_value=200.0,
            value=5.0,
            step=1.0,
            key="true_ml_slippage_bps",
        )

    true_ml_random_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=1_000_000,
        value=42,
        step=1,
        key="true_ml_random_seed",
    )
    run_true_ml_replay = st.button("Run True Historical ML Replay", type="primary")
    if run_true_ml_replay:
        if not true_ml_assets or not true_ml_horizons:
            st.error("Select at least one asset and one horizon.")
            st.stop()
        try:
            true_ml_market_data = pd.read_csv(uploaded_true_ml_market_data) if uploaded_true_ml_market_data is not None else None
            true_ml_report = run_true_historical_ml_replay(
                market_data=true_ml_market_data,
                use_project_market_data=bool(true_ml_use_project_data),
                assets=true_ml_assets,
                horizons=true_ml_horizons,
                max_windows=int(true_ml_max_windows),
                min_train_rows=int(true_ml_min_train_rows),
                step_size=int(true_ml_step_size),
                model_name=str(true_ml_model),
                cost_bps=float(true_ml_cost_bps),
                slippage_bps=float(true_ml_slippage_bps),
                random_seed=int(true_ml_random_seed),
                autosave=True,
            )
            st.session_state.true_historical_ml_replay_report = true_ml_report
            st.session_state.true_historical_ml_replay_settings = true_ml_report.settings
            st.session_state.artifact_store_last_save = true_ml_report.saved_artifacts
        except Exception as exc:
            st.error(f"True historical ML replay failed: {exc}")
            st.stop()

    true_ml_report = st.session_state.true_historical_ml_replay_report
    if true_ml_report is None:
        st.info("Run the replay to create auditable historical ML predictions and baseline comparisons.")
    else:
        st.caption("Artifact source: saved walk-forward ML replay evidence")
        st.caption(f"Last run settings: {st.session_state.get('true_historical_ml_replay_settings') or {}}")
        if not true_ml_report.true_ml_summary_table.empty:
            true_ml_summary = true_ml_report.true_ml_summary_table.iloc[0]
            ts1, ts2, ts3, ts4 = st.columns(4)
            ts1.metric("Final verdict", str(true_ml_summary.get("FinalVerdict", "")))
            ts2.metric("Predictions", int(true_ml_summary.get("TotalPredictions", 0)))
            ts3.metric("Leakage pass rate", f"{float(true_ml_summary.get('LeakagePassRate', 0.0)):.2f}%")
            ts4.metric("Baseline wins", int(true_ml_summary.get("BeatsBestBaselineCount", 0)))
            st.info(str(true_ml_summary.get("MainReason", "")))
            st.warning(str(true_ml_summary.get("MainLimitation", "")))

        true_ml_tabs = st.tabs(
            [
                "Summary",
                "Leakage Audit",
                "Prediction Log",
                "Performance",
                "Baseline Comparison",
                "Strength / Rejection",
                "Quality Gates",
                "Next Actions",
                "Input Sources",
            ]
        )
        true_ml_tables = [
            ("True ML Summary", true_ml_report.true_ml_summary_table, "phase20_true_ml_summary.csv"),
            ("Leakage Audit", true_ml_report.leakage_audit_table, "phase20_leakage_audit.csv"),
            ("True ML Prediction Log", true_ml_report.true_ml_prediction_log, "phase20_true_ml_prediction_log.csv"),
            ("True ML Performance", true_ml_report.true_ml_performance_table, "phase20_true_ml_performance.csv"),
            ("True ML Baseline Comparison", true_ml_report.true_ml_baseline_comparison_table, "phase20_true_ml_baseline_comparison.csv"),
            ("True ML Strength and Rejection", true_ml_report.true_ml_strength_table, "phase20_true_ml_strength.csv"),
            ("Quality Gates", true_ml_report.quality_gates_table, "phase20_quality_gates.csv"),
            ("Next Actions", true_ml_report.next_actions_table, "phase20_next_actions.csv"),
            ("Input Sources", true_ml_report.input_sources_table, "phase20_input_sources.csv"),
        ]
        for tab, (title, table, filename) in zip(true_ml_tabs, true_ml_tables):
            with tab:
                st.markdown(f"### {title}")
                st.caption(_table_research_explanation(title))
                if table.empty:
                    st.info("No rows are available for this table with the current evidence.")
                st.dataframe(table, width="stretch")
                st.download_button(
                    f"Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )


# PAGE: UNIFIED RISK COMMAND CENTER
# ----------------------------------------------------------------

elif page == "Phase 21: Unified Risk Command Center":
    st.markdown('<p class="main-header">Unified Risk Command Center</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Executive evidence layer across replay, policy, benchmark, and risk results</p>', unsafe_allow_html=True)
    render_research_disclaimer()
    render_blocked_capital_banner()
    st.info("This page combines accepted, rejected, and missing evidence. PaperTrack is not a real-capital approval.")
    render_glossary_expander(glossary_entries(["PaperTrack", "WatchlistOnly", "RealCapitalBlocked", "NoBroadEdgeProven", "Benchmark dominated"]))

    unified_use_latest = st.checkbox(
        "Use latest saved replay-audit, policy-lab, and walk-forward ML artifacts",
        value=True,
        key="unified_use_latest",
    )
    unified_prefer_uploads = st.checkbox(
        "Prefer uploaded CSV overrides",
        value=False,
        key="unified_prefer_uploads",
    )

    with st.expander("Optional Evidence CSV Overrides", expanded=False):
        uc1, uc2, uc3 = st.columns(3)
        with uc1:
            uploaded_u18_summary = st.file_uploader("Replay audit summary", type=["csv"], key="u18_summary")
            uploaded_u18_edge = st.file_uploader("Replay asset-horizon edge", type=["csv"], key="u18_edge")
            uploaded_u18_dominance = st.file_uploader("Replay dominance failures", type=["csv"], key="u18_dominance")
            uploaded_u18_quality = st.file_uploader("Replay quality gates", type=["csv"], key="u18_quality")
        with uc2:
            uploaded_u19_summary = st.file_uploader("Signal policy summary", type=["csv"], key="u19_summary")
            uploaded_u19_edge = st.file_uploader("Signal policy asset-horizon edge", type=["csv"], key="u19_edge")
            uploaded_u19_dominance = st.file_uploader("Signal policy dominance failures", type=["csv"], key="u19_dominance")
            uploaded_u19_overfit = st.file_uploader("Signal policy overfit audit", type=["csv"], key="u19_overfit")
            uploaded_u19_cost = st.file_uploader("Signal policy cost sensitivity", type=["csv"], key="u19_cost")
            uploaded_u19_drawdown = st.file_uploader("Signal policy drawdown", type=["csv"], key="u19_drawdown")
            uploaded_u19_quality = st.file_uploader("Signal policy quality gates", type=["csv"], key="u19_quality")
        with uc3:
            uploaded_u20_summary = st.file_uploader("Walk-forward ML summary", type=["csv"], key="u20_summary")
            uploaded_u20_performance = st.file_uploader("Walk-forward ML performance", type=["csv"], key="u20_performance")
            uploaded_u20_baseline = st.file_uploader("Walk-forward ML baseline comparison", type=["csv"], key="u20_baseline")
            uploaded_u20_strength = st.file_uploader("Walk-forward ML strength", type=["csv"], key="u20_strength")
            uploaded_u20_leakage = st.file_uploader("Walk-forward ML leakage audit", type=["csv"], key="u20_leakage")
            uploaded_u20_quality = st.file_uploader("Walk-forward ML quality gates", type=["csv"], key="u20_quality")

    unified_uploaded_overrides = {
        "phase18_summary": uploaded_u18_summary,
        "phase18_asset_horizon_edge": uploaded_u18_edge,
        "phase18_dominance_failures": uploaded_u18_dominance,
        "phase18_quality_gates": uploaded_u18_quality,
        "phase19_summary": uploaded_u19_summary,
        "phase19_asset_horizon_edge": uploaded_u19_edge,
        "phase19_dominance_failures": uploaded_u19_dominance,
        "phase19_overfit_audit": uploaded_u19_overfit,
        "phase19_cost_sensitivity": uploaded_u19_cost,
        "phase19_drawdown": uploaded_u19_drawdown,
        "phase19_quality_gates": uploaded_u19_quality,
        "phase20_summary": uploaded_u20_summary,
        "phase20_performance": uploaded_u20_performance,
        "phase20_baseline_comparison": uploaded_u20_baseline,
        "phase20_strength": uploaded_u20_strength,
        "phase20_leakage_audit": uploaded_u20_leakage,
        "phase20_quality_gates": uploaded_u20_quality,
    }

    run_unified_center = st.button("Run Unified Risk Command Center", type="primary")
    if run_unified_center:
        try:
            unified_report = run_unified_risk_command_center(
                use_artifact_store=bool(unified_use_latest),
                prefer_uploaded=bool(unified_prefer_uploads),
                uploaded_overrides=unified_uploaded_overrides,
                autosave=True,
            )
            st.session_state.unified_risk_command_center_report = unified_report
            st.session_state.unified_risk_command_center_settings = unified_report.settings
            st.session_state.artifact_store_last_save = unified_report.saved_artifacts
        except Exception as exc:
            st.error(f"Unified risk command center failed: {exc}")
            st.stop()

    unified_report = st.session_state.unified_risk_command_center_report
    if unified_report is None:
        st.info("Run the command center to aggregate replay, policy, and benchmark evidence. Missing artifacts will be reported, not hidden.")
    else:
        st.caption("Artifact source: saved unified risk evidence")
        st.caption(f"Last run settings: {st.session_state.get('unified_risk_command_center_settings') or {}}")
        if not unified_report.unified_summary_table.empty:
            unified_summary = unified_report.unified_summary_table.iloc[0]
            render_metric_grid(
                [
                    {"title": "Command-center verdict", "value": unified_summary.get("CommandCenterVerdict", ""), "subtitle": "Combined replay, policy, and benchmark evidence", "status": "info"},
                    {"title": "Broad edge", "value": unified_summary.get("BroadEdgeStatus", ""), "subtitle": "Breadth across assets and horizons", "status": "warning"},
                    {"title": "Best asset / horizon", "value": f"{unified_summary.get('BestAsset', '')} {unified_summary.get('BestHorizon', '')}D", "subtitle": unified_summary.get("BestModelOrPolicy", ""), "status": "positive"},
                    {"title": "Recommended mode", "value": unified_summary.get("RecommendedMode", ""), "subtitle": "Research workflow only", "status": "neutral"},
                ]
            )
            st.info(str(unified_summary.get("FinalExplanation", "")))

        unified_quality = unified_report.quality_gates if isinstance(unified_report.quality_gates, pd.DataFrame) else pd.DataFrame()
        unified_gate_total = len(unified_quality)
        unified_gate_passed = int(unified_quality["Passed"].astype(bool).sum()) if unified_gate_total and "Passed" in unified_quality.columns else 0
        render_section_header("Evidence Health", "Weak, rejected, and missing evidence remains part of the executive view.")
        render_metric_grid(
            [
                {"title": "Paper-track candidates", "value": len(unified_report.paper_tracking_candidates), "subtitle": "Research tracking only", "status": "positive" if len(unified_report.paper_tracking_candidates) else "neutral"},
                {"title": "Rejected candidates", "value": len(unified_report.rejected_candidates), "subtitle": "Visible and retained for audit", "status": "warning" if len(unified_report.rejected_candidates) else "neutral"},
                {"title": "Risk register rows", "value": len(unified_report.risk_register), "subtitle": "Open warnings and blockers", "status": "critical" if len(unified_report.risk_register) else "neutral"},
                {"title": "Quality gates", "value": f"{unified_gate_passed}/{unified_gate_total}", "subtitle": "Passed evidence requirements", "status": "positive" if unified_gate_total and unified_gate_passed == unified_gate_total else "warning"},
            ]
        )

        render_safe_table(
            unified_report.paper_tracking_candidates.head(10),
            "Top Paper-Tracking Candidates",
            "No conservative paper-tracking candidates passed the current evidence gates; review rejected candidates and next actions.",
        )
        render_safe_table(
            unified_report.rejected_candidates.head(10),
            "Visible Rejections",
            "No rejected candidate rows are available in the current evidence package.",
        )

        unified_tabs = st.tabs(
            [
                "Summary",
                "Asset-Horizon Scorecard",
                "Paper Tracking",
                "Rejected Candidates",
                "Risk Register",
                "Quality Gates",
                "Next Actions",
                "Input Sources",
            ]
        )
        unified_tables = [
            ("Unified Summary", unified_report.unified_summary_table, "phase21_unified_summary.csv"),
            ("Asset-Horizon Scorecard", unified_report.asset_horizon_scorecard, "phase21_asset_horizon_scorecard.csv"),
            ("Paper-Tracking Candidates", unified_report.paper_tracking_candidates, "phase21_paper_tracking_candidates.csv"),
            ("Rejected Candidates", unified_report.rejected_candidates, "phase21_rejected_candidates.csv"),
            ("Risk Register", unified_report.risk_register, "phase21_risk_register.csv"),
            ("Quality Gates", unified_report.quality_gates, "phase21_quality_gates.csv"),
            ("Next Actions", unified_report.next_actions, "phase21_next_actions.csv"),
            ("Input Sources", unified_report.input_sources, "phase21_input_sources.csv"),
        ]
        for tab, (title, table, filename) in zip(unified_tabs, unified_tables):
            with tab:
                st.caption(_table_research_explanation(title))
                render_safe_table(table, title, "No rows are available for this table with the current evidence.")
                st.download_button(
                    f"Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )

        with st.expander("Download Center", expanded=False):
            render_download_buttons({title: (table, filename) for title, table, filename in unified_tables})


# PAGE: PREDICTION EDGE IMPROVEMENT
# ----------------------------------------------------------------

elif page == "Phase 22: Prediction Edge Improvement":
    st.markdown('<p class="main-header">Model Edge Benchmark Lab</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Validation-selected model and feature expansion under true historical replay</p>', unsafe_allow_html=True)
    render_research_disclaimer()
    render_blocked_capital_banner()
    st.info("Computation starts only when Run is pressed. Begin with bounded windows and lightweight candidates.")
    st.caption("If BeatsBestBaseline is false, the model did not demonstrate edge for that asset/horizon.")
    render_glossary_expander(glossary_entries(["Baseline", "Leakage", "Walk-forward validation", "Cost drag", "Benchmark dominated"]))

    edge_use_project_data = st.checkbox(
        "Use project master market dataset when no CSV is uploaded",
        value=True,
        key="edge_use_project_data",
    )
    uploaded_edge_market_data = st.file_uploader(
        "Upload historical market data CSV",
        type=["csv"],
        key="edge_market_data_upload",
        help="Expected to include Date and configured asset close-price columns.",
    )

    pe1, pe2, pe3, pe4 = st.columns(4)
    with pe1:
        edge_assets = st.multiselect(
            "Assets",
            get_asset_names(),
            default=get_asset_names(),
            key="edge_assets",
        )
        edge_horizons = st.multiselect(
            "Horizons",
            list(EDGE_HORIZONS),
            default=list(EDGE_HORIZONS),
            format_func=lambda value: f"{int(value)}D",
            key="edge_horizons",
        )
    with pe2:
        edge_model_options = list(EDGE_MODEL_CHOICES) + list(EDGE_OPTIONAL_MODELS)
        edge_models = st.multiselect(
            "Models",
            edge_model_options,
            default=["Ridge", "ElasticNet", "LinearRegression"],
            key="edge_models",
        )
        edge_groups = st.multiselect(
            "Feature groups",
            list(EDGE_FEATURE_GROUPS),
            default=["PriceReturn", "TechnicalIndicators"],
            key="edge_feature_groups",
        )
    with pe3:
        edge_max_windows = st.number_input(
            "Maximum windows per asset/horizon",
            min_value=1,
            max_value=100,
            value=4,
            step=1,
            key="edge_max_windows",
        )
        edge_min_train_rows = st.number_input(
            "Minimum training rows",
            min_value=20,
            max_value=5000,
            value=120,
            step=10,
            key="edge_min_train_rows",
        )
        edge_step_size = st.number_input(
            "Walk-forward step size",
            min_value=1,
            max_value=500,
            value=20,
            step=1,
            key="edge_step_size",
        )
    with pe4:
        edge_cost_bps = st.number_input(
            "Cost bps",
            min_value=0.0,
            max_value=200.0,
            value=10.0,
            step=1.0,
            key="edge_cost_bps",
        )
        edge_slippage_bps = st.number_input(
            "Slippage bps",
            min_value=0.0,
            max_value=200.0,
            value=5.0,
            step=1.0,
            key="edge_slippage_bps",
        )
        edge_random_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=1_000_000,
            value=42,
            step=1,
            key="edge_random_seed",
        )

    edge_enable_ensemble = st.checkbox(
        "Enable validation-weighted ensemble candidate",
        value=True,
        key="edge_enable_ensemble",
    )
    edge_enable_optional_models = st.checkbox(
        "Enable optional models for this bounded run",
        value=False,
        key="edge_enable_optional_models",
        help="When disabled, optional packages are reported as skipped and are not fitted.",
    )
    edge_cost_scenarios = st.multiselect(
        "Cost sensitivity scenarios",
        [0, 5, 10, 25, 50],
        default=[0, 5, 10, 25, 50],
        key="edge_cost_scenarios",
    )

    run_edge_improvement = st.button("Run Prediction Edge Improvement", type="primary")
    if run_edge_improvement:
        if not edge_assets or not edge_horizons or not edge_models or not edge_groups:
            st.error("Select at least one asset, horizon, model, and feature group.")
            st.stop()
        try:
            edge_market_data = pd.read_csv(uploaded_edge_market_data) if uploaded_edge_market_data is not None else None
            edge_report = run_prediction_edge_improvement(
                market_data=edge_market_data,
                use_project_market_data=bool(edge_use_project_data),
                assets=edge_assets,
                horizons=edge_horizons,
                max_windows=int(edge_max_windows),
                min_train_rows=int(edge_min_train_rows),
                step_size=int(edge_step_size),
                cost_bps=float(edge_cost_bps),
                slippage_bps=float(edge_slippage_bps),
                random_seed=int(edge_random_seed),
                models_to_test=edge_models,
                feature_groups=edge_groups,
                enable_ensemble=bool(edge_enable_ensemble),
                enable_optional_models=bool(edge_enable_optional_models),
                cost_scenarios_bps=edge_cost_scenarios,
                autosave=True,
            )
            st.session_state.prediction_edge_improvement_report = edge_report
            st.session_state.prediction_edge_improvement_settings = edge_report.settings
            st.session_state.artifact_store_last_save = edge_report.saved_artifacts
        except Exception as exc:
            st.error(f"Prediction edge improvement failed: {exc}")
            st.stop()

    edge_report = st.session_state.prediction_edge_improvement_report
    if edge_report is None:
        st.info("Run the benchmark lab to compare validation-selected models and feature groups under true historical replay.")
    else:
        st.caption("Artifact source: saved model-edge benchmark evidence")
        st.caption(f"Last run settings: {st.session_state.get('prediction_edge_improvement_settings') or {}}")
        if not edge_report.prediction_edge_summary.empty:
            edge_summary = edge_report.prediction_edge_summary.iloc[0]
            render_metric_grid(
                [
                    {"title": "Final verdict", "value": edge_summary.get("FinalVerdict", ""), "subtitle": edge_summary.get("BroadEdgeStatus", ""), "status": "info"},
                    {"title": "Best model", "value": edge_summary.get("BestModel", ""), "subtitle": "Validation-selected research candidate", "status": "positive"},
                    {"title": "Best feature group", "value": edge_summary.get("BestFeatureGroup", ""), "subtitle": f"Best gap: {edge_summary.get('BestBaselineGapPct', '')}", "status": "neutral"},
                    {"title": "Baseline wins", "value": int(edge_summary.get("BeatsBestBaselineCount", 0)), "subtitle": "Asset-horizon winners", "status": "warning"},
                ]
            )
            st.caption(f"Optional models available: {edge_summary.get('OptionalModelsAvailable', '') or 'None'}")
            st.caption(f"Optional models tested: {edge_summary.get('OptionalModelsTested', '') or 'None'}")
            st.caption(f"Optional models skipped: {edge_summary.get('OptionalModelsSkipped', '') or 'None'}")

        edge_quality = edge_report.quality_gates if isinstance(edge_report.quality_gates, pd.DataFrame) else pd.DataFrame()
        edge_gate_total = len(edge_quality)
        edge_gate_passed = int(edge_quality["Passed"].astype(bool).sum()) if edge_gate_total and "Passed" in edge_quality.columns else 0
        render_section_header("Benchmark Evidence", "Candidate breadth and rejection visibility matter alongside the best score.")
        render_metric_grid(
            [
                {"title": "Models evaluated", "value": len(edge_report.model_leaderboard), "subtitle": "Leaderboard rows retained", "status": "info"},
                {"title": "Asset-horizon rows", "value": len(edge_report.asset_horizon_model_scorecard), "subtitle": "Cross-market evidence coverage", "status": "neutral"},
                {"title": "Rejected models", "value": len(edge_report.rejected_models), "subtitle": "Weak candidates remain visible", "status": "warning" if len(edge_report.rejected_models) else "neutral"},
                {"title": "Quality gates", "value": f"{edge_gate_passed}/{edge_gate_total}", "subtitle": "Passed model-edge requirements", "status": "positive" if edge_gate_total and edge_gate_passed == edge_gate_total else "warning"},
            ]
        )
        render_safe_table(
            edge_report.rejected_models.head(10),
            "Visible Rejected Models",
            "No rejected model rows are available for the current bounded run.",
        )

        edge_tabs = st.tabs(
            [
                "Summary", "Model Leaderboard", "Asset-Horizon Scorecard", "Prediction Log",
                "Baseline Comparison", "Feature Audit", "Selection Audit", "Leakage Audit",
                "Cost Sensitivity", "Rejected Models", "Quality Gates", "Next Actions", "Input Sources",
            ]
        )
        edge_tables = [
            ("Prediction Edge Summary", edge_report.prediction_edge_summary, "phase22_prediction_edge_summary.csv"),
            ("Model Leaderboard", edge_report.model_leaderboard, "phase22_model_leaderboard.csv"),
            ("Asset-Horizon Model Scorecard", edge_report.asset_horizon_model_scorecard, "phase22_asset_horizon_model_scorecard.csv"),
            ("Prediction Log", edge_report.prediction_log, "phase22_prediction_log.csv"),
            ("Baseline Comparison", edge_report.baseline_comparison, "phase22_baseline_comparison.csv"),
            ("Feature Group Audit", edge_report.feature_group_audit, "phase22_feature_group_audit.csv"),
            ("Model Selection Audit", edge_report.model_selection_audit, "phase22_model_selection_audit.csv"),
            ("Leakage Audit", edge_report.leakage_audit, "phase22_leakage_audit.csv"),
            ("Cost Sensitivity", edge_report.cost_sensitivity, "phase22_cost_sensitivity.csv"),
            ("Rejected Models", edge_report.rejected_models, "phase22_rejected_models.csv"),
            ("Quality Gates", edge_report.quality_gates, "phase22_quality_gates.csv"),
            ("Next Actions", edge_report.next_actions, "phase22_next_actions.csv"),
            ("Input Sources", edge_report.input_sources, "phase22_input_sources.csv"),
        ]
        for tab, (title, table, filename) in zip(edge_tabs, edge_tables):
            with tab:
                st.caption(_table_research_explanation(title))
                render_safe_table(table, title, "No rows are available for this table with the current evidence.")
                st.download_button(
                    f"Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )

        with st.expander("Download Center", expanded=False):
            render_download_buttons({title: (table, filename) for title, table, filename in edge_tables})


# PAGE: HISTORICAL MODEL REPLAY
# ----------------------------------------------------------------

elif page == "🕰️ Historical Model Replay":
    st.markdown('<p class="main-header">🕰️ Historical Model Replay</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Research replay only. This page reconstructs time-safe historical paper signals where possible, "
        "labels proxy evidence clearly, keeps weak rows visible, and does not allow real-capital exposure."
    )

    replay_use_latest = st.checkbox("Use latest saved Phase 9/10/12/13/14/15 artifacts", value=True, key="replay_use_latest")
    replay_prefer_uploads = st.checkbox(
        "Prefer uploaded CSV overrides",
        value=False,
        key="replay_prefer_uploads",
        help="When enabled, uploaded CSVs override matching saved artifacts.",
    )
    replay_use_project_market_data = st.checkbox(
        "Use project master market dataset when no market CSV is uploaded",
        value=True,
        key="replay_use_project_market_data",
    )

    with st.expander("Market Data and Optional Evidence Uploads", expanded=False):
        uploaded_replay_market_data = st.file_uploader(
            "Upload market data CSV",
            type=["csv"],
            key="replay_market_data_upload",
            help="Expected to include Date plus configured asset price columns.",
        )
        rp_a, rp_b, rp_c = st.columns(3)
        with rp_a:
            uploaded_replay_predictions = st.file_uploader("historical_prediction_log.csv", type=["csv"], key="replay_predictions_upload")
            uploaded_replay_forward = st.file_uploader("Phase 9 forward_signal_log.csv", type=["csv"], key="replay_forward_upload")
            uploaded_replay_raw = st.file_uploader("Phase 8I true_raw_trade_log.csv", type=["csv"], key="replay_raw_upload")
        with rp_b:
            uploaded_replay_plan = st.file_uploader("Phase 10 ranked_asset_horizon_plan.csv", type=["csv"], key="replay_plan_upload")
            uploaded_replay_allocation = st.file_uploader("Phase 12 allocation_plan_table.csv", type=["csv"], key="replay_allocation_upload")
            uploaded_replay_risk = st.file_uploader("Phase 13 asset_horizon_risk_matrix.csv", type=["csv"], key="replay_risk_upload")
        with rp_c:
            uploaded_replay_sizing = st.file_uploader("Phase 14 dynamic_position_sizing_table.csv", type=["csv"], key="replay_sizing_upload")
            uploaded_replay_regime = st.file_uploader("Phase 15 asset_horizon_regime_table.csv", type=["csv"], key="replay_regime_upload")

    replay_uploaded_overrides = {
        "historical_prediction_log": uploaded_replay_predictions,
        "forward_signal_log": uploaded_replay_forward,
        "true_raw_trade_log": uploaded_replay_raw,
        "ranked_asset_horizon_plan": uploaded_replay_plan,
        "allocation_plan_table": uploaded_replay_allocation,
        "asset_horizon_risk_matrix": uploaded_replay_risk,
        "dynamic_position_sizing_table": uploaded_replay_sizing,
        "asset_horizon_regime_table": uploaded_replay_regime,
    }

    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        replay_assets = st.multiselect("Assets", get_asset_names(), default=get_asset_names(), key="replay_assets")
    with rc2:
        replay_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
        selected_replay_horizon_labels = st.multiselect("Horizons", replay_horizon_labels, default=replay_horizon_labels, key="replay_horizons")
        replay_horizons = [int(str(label).replace("D", "")) for label in selected_replay_horizon_labels]
    with rc3:
        replay_step = st.number_input("Replay step rows", min_value=1, max_value=60, value=5, step=1, key="replay_step")
        max_replay_paper_weight = st.slider("Max paper weight per row %", min_value=0.0, max_value=30.0, value=20.0, step=1.0, key="replay_max_paper_weight")
    with rc4:
        max_replay_portfolio_exposure = st.slider("Max portfolio paper exposure %", min_value=0.0, max_value=100.0, value=45.0, step=5.0, key="replay_max_portfolio_exposure")
        replay_start_text = st.text_input("Replay start date optional", value="", key="replay_start_date")
        replay_end_text = st.text_input("Replay end date optional", value="", key="replay_end_date")

    run_replay = st.button("Run Historical Model Replay", type="primary")
    if run_replay:
        if not replay_assets or not replay_horizons:
            st.error("Select at least one asset and one horizon.")
            st.stop()
        try:
            replay_market_data_input = pd.read_csv(uploaded_replay_market_data) if uploaded_replay_market_data is not None else None
            replay_report = run_historical_model_replay(
                market_data=replay_market_data_input,
                use_project_market_data=bool(replay_use_project_market_data),
                use_artifact_store=bool(replay_use_latest),
                prefer_uploaded=bool(replay_prefer_uploads),
                uploaded_overrides=replay_uploaded_overrides,
                assets=replay_assets,
                horizons=replay_horizons,
                replay_start_date=replay_start_text.strip() or None,
                replay_end_date=replay_end_text.strip() or None,
                replay_step=int(replay_step),
                max_paper_weight_pct=float(max_replay_paper_weight),
                max_portfolio_paper_exposure_pct=float(max_replay_portfolio_exposure),
                autosave=True,
            )
            st.session_state.historical_model_replay_report = replay_report
            st.session_state.historical_model_replay_settings = replay_report.settings
            st.session_state.artifact_store_last_save = replay_report.saved_artifacts
        except Exception as exc:
            st.error(f"Historical model replay failed: {exc}")
            st.stop()

    replay_report = st.session_state.historical_model_replay_report
    if replay_report is None:
        st.info("Run Phase 17 to build a time-safe replay table and Phase 16-compatible export.")
    else:
        st.caption(f"Artifact phase: {HISTORICAL_REPLAY_PHASE_NAME}")
        st.caption(f"Last run settings: {st.session_state.get('historical_model_replay_settings') or {}}")
        if not replay_report.replay_summary_table.empty:
            replay_summary = replay_report.replay_summary_table.iloc[0]
            rs1, rs2, rs3, rs4, rs5 = st.columns(5)
            rs1.metric("Replay verdict", str(replay_summary.get("ReplayVerdict", "")))
            rs2.metric("Replay source", str(replay_summary.get("ReplaySource", "")))
            rs3.metric("Replay rows", int(replay_summary.get("ReplayRows", 0) or 0))
            rs4.metric("Avg portfolio exposure %", float(replay_summary.get("AveragePortfolioPaperExposurePct", 0.0) or 0.0))
            rs5.metric("Max portfolio exposure %", float(replay_summary.get("MaxPortfolioPaperExposurePct", 0.0) or 0.0))
            st.info(str(replay_summary.get("MainLimitation", "")))
            if str(replay_summary.get("ReplayVerdict", "")) == "ProxyReplayOnly":
                st.warning("Proxy replay is not true historical model prediction evidence. Persist timestamped model predictions for stronger replay.")

        replay_tabs = st.tabs(
            [
                "Summary",
                "Signal Log",
                "Outcomes",
                "Performance",
                "Portfolio Curve",
                "Asset Horizon Matrix",
                "Exposure Cap",
                "Quality Checks",
                "Benchmark Ready",
                "Warnings",
                "Next Actions",
                "Input Sources",
                "Artifact Inputs",
                "Phase 16 Export",
            ]
        )
        replay_tables = [
            ("Replay Summary", replay_report.replay_summary_table, "phase17_replay_summary.csv"),
            ("Historical Replay Signal Log", replay_report.historical_replay_signal_log, "phase17_historical_replay_signal_log.csv"),
            ("Historical Replay Outcomes", replay_report.historical_replay_outcomes, "phase17_historical_replay_outcomes.csv"),
            ("Historical Replay Performance", replay_report.historical_replay_performance, "phase17_historical_replay_performance.csv"),
            ("Historical Replay Portfolio Curve", replay_report.historical_replay_portfolio_curve, "phase17_historical_replay_portfolio_curve.csv"),
            ("Replay Asset Horizon Matrix", replay_report.replay_asset_horizon_matrix, "phase17_replay_asset_horizon_matrix.csv"),
            ("Replay Exposure Cap", replay_report.replay_exposure_cap_table, "phase17_replay_exposure_cap.csv"),
            ("Replay Quality Checks", replay_report.replay_quality_checks, "phase17_replay_quality_checks.csv"),
            ("Replay Benchmark Ready", replay_report.replay_benchmark_ready_table, "phase17_replay_benchmark_ready.csv"),
            ("Replay Warnings", replay_report.replay_warnings_table, "phase17_replay_warnings.csv"),
            ("Next Replay Actions", replay_report.next_replay_actions_table, "phase17_next_replay_actions.csv"),
            ("Replay Input Sources", replay_report.replay_input_sources_table, "phase17_replay_input_sources.csv"),
            ("Artifact Inputs", replay_report.artifact_input_source_table, "phase17_artifact_input_sources.csv"),
            ("Phase 16 Replay Export", replay_report.phase16_replay_export_table, "phase17_phase16_replay_export.csv"),
        ]
        for tab, (title, table, filename) in zip(replay_tabs, replay_tables):
            with tab:
                st.markdown(f"### {title}")
                if table.empty:
                    st.info("No rows for this table with the currently loaded evidence.")
                st.dataframe(table, width="stretch")
                st.download_button(
                    f"Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )


# PAGE: EVIDENCE STORE
# ═══════════════════════════════════════════════════════════════════════

elif page == "🗂️ Evidence Store":
    st.markdown('<p class="main-header">🗂️ Evidence Store</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.info(
        "Phase 10B stores durable research evidence under the project `artifacts/` folder. "
        "These files survive Streamlit restarts, browser refreshes, cache clears, and Python process restarts."
    )

    registry = get_artifact_registry()
    latest_artifacts = list_latest_artifacts()

    st.markdown("### Latest Artifacts")
    if latest_artifacts.empty:
        st.warning("No saved artifacts found yet. Run Phase 8F, Phase 8I, Phase 9, or Phase 10 to populate the store.")
    else:
        phase_filter_options = ["All"] + sorted(latest_artifacts["Phase"].dropna().astype(str).unique().tolist())
        selected_phase_filter = st.selectbox("Phase filter", phase_filter_options, index=0, key="evidence_store_phase_filter")
        display_latest = latest_artifacts if selected_phase_filter == "All" else latest_artifacts[latest_artifacts["Phase"].astype(str).eq(selected_phase_filter)]
        st.dataframe(display_latest, width="stretch")
        st.download_button(
            "📥 Export Latest Artifact Registry CSV",
            data=display_latest.to_csv(index=False).encode("utf-8"),
            file_name="artifact_store_latest.csv",
            mime="text/csv",
            key="artifact_store_latest_download",
        )

        st.markdown("### Download Latest Artifact")
        artifact_labels = [
            f"{row['Phase']} :: {row['ArtifactName']} :: {row['RunId']}"
            for _, row in display_latest.iterrows()
        ]
        if artifact_labels:
            selected_artifact_label = st.selectbox("Latest artifact", artifact_labels, key="artifact_store_download_select")
            selected_idx = artifact_labels.index(selected_artifact_label)
            selected_row = display_latest.iloc[selected_idx]
            try:
                selected_df = load_latest_artifact(selected_row["Phase"], selected_row["ArtifactName"], required=True)
                if isinstance(selected_df, pd.DataFrame):
                    st.download_button(
                        "📥 Download Selected Latest CSV",
                        data=selected_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{selected_row['ArtifactName']}.csv",
                        mime="text/csv",
                        key="artifact_store_selected_download",
                    )
                    st.dataframe(selected_df.head(200), width="stretch")
            except Exception as exc:
                st.error(f"Could not load selected artifact: {exc}")

    st.markdown("### Required Artifact Diagnostics")
    required_specs = [
        {"phase_name": "Phase 8F Probability Calibration", "artifact_name": "probability_calibration_summary", "required": True},
        {"phase_name": "Phase 8F Probability Calibration", "artifact_name": "probability_calibration_warnings", "required": True},
        {"phase_name": "Phase 8I True Raw Trade Logs", "artifact_name": "true_raw_trade_log", "required": True},
        {"phase_name": "Phase 9 Forward Paper Evidence", "artifact_name": "forward_signal_log", "required": True},
        {"phase_name": "Phase 9 Forward Paper Evidence", "artifact_name": "forward_accuracy_summary", "required": False},
        {"phase_name": "Phase 10 Actionable Research Plan", "artifact_name": "ranked_asset_horizon_plan", "required": False},
    ]
    diagnostics = validate_required_artifacts(required_specs)
    st.dataframe(diagnostics, width="stretch")

    st.markdown("### Run History")
    run_rows = []
    for run_id, run_meta in registry.get("Runs", {}).items():
        run_rows.append(
            {
                "RunId": run_id,
                "Phase": run_meta.get("Phase", ""),
                "PhaseSlug": run_meta.get("PhaseSlug", ""),
                "CreatedAt": run_meta.get("CreatedAt", ""),
                "ArtifactCount": len(run_meta.get("Artifacts", {})),
                "ManifestPath": run_meta.get("ManifestPath", ""),
            }
        )
    run_history = pd.DataFrame(run_rows).sort_values("CreatedAt", ascending=False) if run_rows else pd.DataFrame(columns=["RunId", "Phase", "PhaseSlug", "CreatedAt", "ArtifactCount", "ManifestPath"])
    st.dataframe(run_history, width="stretch")
    st.download_button(
        "📥 Export Run History CSV",
        data=run_history.to_csv(index=False).encode("utf-8"),
        file_name="artifact_store_run_history.csv",
        mime="text/csv",
        key="artifact_store_run_history_download",
    )

    st.markdown("### Raw Registry")
    st.json(registry)


# PAGE: TRUE RAW TRADE LOGS
# ═══════════════════════════════════════════════════════════════════════

elif page == "🧾 True Raw Trade Logs":
    st.markdown('<p class="main-header">🧾 True Raw Trade Logs</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "This page generates true raw signal/trade logs for research calibration only. "
        "It is not live trading. The win condition is raw evidence quality, not better returns."
    )

    latest_grading_report = st.session_state.get("meta_reliability_grading_report")
    latest_evidence_report = st.session_state.get("evidence_expansion_report")
    latest_policy_report = st.session_state.get("signal_policy_sensitivity_report")
    latest_grading_table = getattr(latest_grading_report, "grading_table", None) if latest_grading_report is not None else None
    latest_full_evidence = getattr(latest_evidence_report, "full_evidence_table", None) if latest_evidence_report is not None else None
    latest_policy_table = getattr(latest_policy_report, "full_policy_sensitivity_table", None) if latest_policy_report is not None else None
    latest_policy_recs = getattr(latest_policy_report, "candidate_recommendation_table", None) if latest_policy_report is not None else None

    true_raw_source = st.radio(
        "Generation source",
        ["Generate from signal engine", "Uploaded raw/diagnostic CSVs"],
        horizontal=True,
        key="true_raw_source",
    )

    uploaded_true_raw = uploaded_true_grade = uploaded_true_full = uploaded_true_policy = uploaded_true_recs = None
    if true_raw_source == "Uploaded raw/diagnostic CSVs":
        upload_a, upload_b, upload_c = st.columns(3)
        with upload_a:
            uploaded_true_raw = st.file_uploader("Optional raw signal/trade CSV", type=["csv"], key="true_raw_raw_upload")
            uploaded_true_grade = st.file_uploader("Optional Phase 8B grading CSV", type=["csv"], key="true_raw_grade_upload")
        with upload_b:
            uploaded_true_full = st.file_uploader("Optional Phase 8C full evidence CSV", type=["csv"], key="true_raw_full_upload")
            uploaded_true_policy = st.file_uploader("Optional Phase 8E policy CSV", type=["csv"], key="true_raw_policy_upload")
        with upload_c:
            uploaded_true_recs = st.file_uploader("Optional Phase 8E recommendations CSV", type=["csv"], key="true_raw_recs_upload")

    true_col_a, true_col_b, true_col_c, true_col_d = st.columns(4)
    with true_col_a:
        true_candidate_filter = st.selectbox(
            "Candidate selector",
            ["all", "only C/D candidates", "specific asset/horizon"],
            index=0,
            key="true_raw_candidate_filter",
        )
    with true_col_b:
        true_model_depth = st.selectbox("Model depth", ["fast", "core"], index=0, key="true_raw_model_depth")
    with true_col_c:
        true_use_phase5 = st.checkbox("Use Phase 5 features", value=True, key="true_raw_phase5")
    with true_col_d:
        true_signal_mode = st.selectbox("Signal mode", ["long_only", "long_short", "avoid_only"], index=0, key="true_raw_signal_mode")

    if true_candidate_filter == "specific asset/horizon":
        spec_a, spec_b = st.columns(2)
        with spec_a:
            true_assets = st.multiselect("Assets", get_asset_names(), default=get_asset_names(), key="true_raw_assets")
        with spec_b:
            true_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
            selected_true_horizon_labels = st.multiselect("Horizons", true_horizon_labels, default=true_horizon_labels, key="true_raw_horizons")
            true_horizons = [int(str(label).replace("D", "")) for label in selected_true_horizon_labels]
    else:
        controls_a, controls_b = st.columns(2)
        with controls_a:
            true_assets = st.multiselect("Assets", get_asset_names(), default=get_asset_names(), key="true_raw_all_assets")
        with controls_b:
            true_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
            selected_true_horizon_labels = st.multiselect("Horizons", true_horizon_labels, default=true_horizon_labels, key="true_raw_all_horizons")
            true_horizons = [int(str(label).replace("D", "")) for label in selected_true_horizon_labels]

    true_param_a, true_param_b, true_param_c, true_param_d = st.columns(4)
    with true_param_a:
        true_thresholds = st.multiselect(
            "Threshold candidates",
            [0.50, 0.55, 0.60, 0.65, 0.70],
            default=[0.50, 0.55, 0.60, 0.65, 0.70],
            format_func=lambda x: f"{x:.2f}",
            key="true_raw_thresholds",
        )
    with true_param_b:
        true_cooldown = st.number_input("Cooldown rows", min_value=0, max_value=30, value=0, step=1, key="true_raw_cooldown")
    with true_param_c:
        true_cost = st.number_input("Transaction cost", min_value=0.0, max_value=0.02, value=0.001, step=0.0005, format="%.4f", key="true_raw_cost")
    with true_param_d:
        true_validation_fraction = st.slider("Validation segment", min_value=0.30, max_value=0.70, value=0.50, step=0.05, key="true_raw_validation_fraction")

    run_true_raw = st.button("🚀 Generate True Raw Trade Logs", type="primary")

    if run_true_raw:
        if not true_assets or not true_horizons or not true_thresholds:
            st.error("Select at least one asset, horizon, and threshold candidate.")
            st.stop()
        try:
            raw_signal_input = None
            grading_input = latest_grading_table.copy() if latest_grading_table is not None else None
            full_input = latest_full_evidence.copy() if latest_full_evidence is not None else None
            policy_input = latest_policy_table.copy() if latest_policy_table is not None else None
            recs_input = latest_policy_recs.copy() if latest_policy_recs is not None else None
            if true_raw_source == "Uploaded raw/diagnostic CSVs":
                grading_input = full_input = policy_input = recs_input = None
                if uploaded_true_raw is not None:
                    uploaded_true_raw.seek(0)
                    raw_signal_input = pd.read_csv(uploaded_true_raw)
                if uploaded_true_grade is not None:
                    uploaded_true_grade.seek(0)
                    grading_input = pd.read_csv(uploaded_true_grade)
                if uploaded_true_full is not None:
                    uploaded_true_full.seek(0)
                    full_input = pd.read_csv(uploaded_true_full)
                if uploaded_true_policy is not None:
                    uploaded_true_policy.seek(0)
                    policy_input = pd.read_csv(uploaded_true_policy)
                if uploaded_true_recs is not None:
                    uploaded_true_recs.seek(0)
                    recs_input = pd.read_csv(uploaded_true_recs)
                raw_df_input = None
            else:
                raw_df_input = load_raw_data("2015-01-01", use_cache=True)
        except Exception as exc:
            st.error(f"Could not prepare true raw generation inputs: {exc}")
            st.stop()

        with st.spinner("Generating true raw signal/trade logs from the direct forecast and signal engine path..."):
            try:
                true_report = run_true_raw_trade_log_generation(
                    raw_df=raw_df_input,
                    raw_signal_outputs=raw_signal_input,
                    grading_table=grading_input,
                    full_evidence_table=full_input,
                    policy_sensitivity_table=policy_input,
                    candidate_recommendation_table=recs_input,
                    assets=true_assets,
                    horizons=true_horizons,
                    candidate_filter=true_candidate_filter,
                    model_depth=true_model_depth,
                    use_phase5_features=true_use_phase5,
                    signal_mode=true_signal_mode,
                    threshold_candidates=true_thresholds,
                    cooldown=int(true_cooldown),
                    transaction_cost=float(true_cost),
                    validation_fraction=float(true_validation_fraction),
                )
                st.session_state.true_raw_trade_log_report = true_report
                st.session_state.true_raw_trade_log_settings = true_report.settings
                saved_artifacts = save_phase_artifacts(
                    "Phase 8I True Raw Trade Logs",
                    {
                        "true_raw_trade_log": true_report.true_raw_trade_log,
                        "raw_log_quality_summary": true_report.raw_log_quality_summary,
                        "asset_horizon_raw_coverage": true_report.asset_horizon_raw_coverage,
                        "probability_outcome_readiness": true_report.probability_outcome_readiness,
                        "missing_source_diagnostic": true_report.missing_source_diagnostic,
                        "benchmark_comparison": true_report.benchmark_comparison,
                        "drawdown_during_trade": true_report.drawdown_during_trade,
                        "true_raw_warning_table": true_report.warning_table,
                        "next_research_action_table": true_report.next_research_action_table,
                        "phase8_closure_readiness_table": true_report.phase8_closure_readiness_table,
                        "aggregate_fallback_diagnostic": true_report.aggregate_fallback_diagnostic,
                        "no_trade_skipped_signal_table": true_report.no_trade_skipped_signal_table,
                    },
                    config=true_report.settings,
                    warnings=true_report.warning_table["WarningType"].dropna().astype(str).unique().tolist() if not true_report.warning_table.empty and "WarningType" in true_report.warning_table.columns else [],
                )
                st.session_state.artifact_store_last_save = saved_artifacts
            except Exception as exc:
                st.error(f"True raw trade log generation failed: {exc}")
                st.stop()

    true_report = st.session_state.true_raw_trade_log_report
    if true_report is None:
        st.info("Generate true raw logs to close Phase 8 infrastructure or prove the exact missing source limitation.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('true_raw_trade_log_settings') or {}}")
        st.markdown("### Phase 8 Closure Readiness")
        st.dataframe(true_report.phase8_closure_readiness_table, width="stretch")
        st.download_button(
            "📥 Export Phase 8 Closure Readiness CSV",
            data=true_report.phase8_closure_readiness_table.to_csv(index=False).encode("utf-8"),
            file_name="phase8_closure_readiness.csv",
            mime="text/csv",
            key="true_raw_closure_download",
        )

        tabs = st.tabs([
            "Quality Summary",
            "Raw Coverage",
            "Probability Readiness",
            "Missing Sources",
            "True Raw Log",
            "Benchmark",
            "Drawdown",
            "Warnings",
            "Next Actions",
            "Aggregate Diagnostics",
            "No-Trade / Skipped",
        ])
        true_tables = [
            ("Raw Log Quality Summary", true_report.raw_log_quality_summary, "true_raw_quality_summary.csv"),
            ("Asset-Horizon Raw Coverage", true_report.asset_horizon_raw_coverage, "true_raw_asset_horizon_coverage.csv"),
            ("Probability Outcome Readiness", true_report.probability_outcome_readiness, "true_raw_probability_readiness.csv"),
            ("Missing Source Diagnostic", true_report.missing_source_diagnostic, "true_raw_missing_source_diagnostic.csv"),
            ("True Raw Trade Log", true_report.true_raw_trade_log, "true_raw_trade_log.csv"),
            ("Benchmark Comparison", true_report.benchmark_comparison, "true_raw_benchmark_comparison.csv"),
            ("Drawdown During Trade", true_report.drawdown_during_trade, "true_raw_drawdown.csv"),
            ("Warning Table", true_report.warning_table, "true_raw_warnings.csv"),
            ("Next Research Action Table", true_report.next_research_action_table, "true_raw_next_actions.csv"),
            ("Aggregate Fallback Diagnostic", true_report.aggregate_fallback_diagnostic, "true_raw_aggregate_fallback_diagnostic.csv"),
            ("No-Trade / Skipped Signal", true_report.no_trade_skipped_signal_table, "true_raw_no_trade_skipped.csv"),
        ]
        for tab, (title, table, filename) in zip(tabs, true_tables):
            with tab:
                st.markdown(f"### {title}")
                st.dataframe(table, width="stretch")
                st.download_button(
                    f"📥 Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )


# PAGE: RAW TRADE LOG EXPORTER
# ═══════════════════════════════════════════════════════════════════════

elif page == "📜 Raw Trade Log Exporter":
    st.markdown('<p class="main-header">📜 Raw Trade Log Exporter</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 8H is raw evidence infrastructure only. It exports row-level signal/trade evidence when available "
        "and labels reconstructed or aggregate fallback rows honestly. It is not live-trading approval."
    )

    latest_grading_report = st.session_state.get("meta_reliability_grading_report")
    latest_evidence_report = st.session_state.get("evidence_expansion_report")
    latest_policy_report = st.session_state.get("signal_policy_sensitivity_report")
    latest_ledger_report = st.session_state.get("trade_evidence_ledger_report")
    latest_grading_table = getattr(latest_grading_report, "grading_table", None) if latest_grading_report is not None else None
    latest_full_evidence = getattr(latest_evidence_report, "full_evidence_table", None) if latest_evidence_report is not None else None
    latest_policy_table = getattr(latest_policy_report, "full_policy_sensitivity_table", None) if latest_policy_report is not None else None
    latest_policy_recs = getattr(latest_policy_report, "candidate_recommendation_table", None) if latest_policy_report is not None else None
    latest_ledger_table = getattr(latest_ledger_report, "ledger_table", None) if latest_ledger_report is not None else None

    raw_sources = []
    if (
        (latest_grading_table is not None and not latest_grading_table.empty)
        or (latest_full_evidence is not None and not latest_full_evidence.empty)
        or (latest_policy_table is not None and not latest_policy_table.empty)
        or (latest_ledger_table is not None and not latest_ledger_table.empty)
    ):
        raw_sources.append("Latest Phase 8B/8C/8E/8G session result")
    raw_sources.append("Uploaded CSVs")
    raw_source = st.radio("Raw log source", raw_sources, horizontal=True, key="raw_trade_export_source")

    uploaded_grade = uploaded_full = uploaded_policy = uploaded_recs = uploaded_ledger = uploaded_raw = None
    if raw_source == "Uploaded CSVs":
        upload_a, upload_b, upload_c = st.columns(3)
        with upload_a:
            uploaded_grade = st.file_uploader("Upload Phase 8B grading CSV", type=["csv"], key="raw_export_grade_upload")
            uploaded_full = st.file_uploader("Upload Phase 8C full evidence CSV", type=["csv"], key="raw_export_full_upload")
        with upload_b:
            uploaded_policy = st.file_uploader("Upload Phase 8E policy sensitivity CSV", type=["csv"], key="raw_export_policy_upload")
            uploaded_recs = st.file_uploader("Upload Phase 8E candidate recommendation CSV", type=["csv"], key="raw_export_recs_upload")
        with upload_c:
            uploaded_ledger = st.file_uploader("Optional Phase 8G ledger CSV", type=["csv"], key="raw_export_ledger_upload")
            uploaded_raw = st.file_uploader("Optional raw signal/trade CSV", type=["csv"], key="raw_export_raw_upload")
    else:
        uploaded_raw = st.file_uploader("Optional raw signal/trade CSV", type=["csv"], key="raw_export_session_raw_upload")

    raw_col_a, raw_col_b = st.columns(2)
    with raw_col_a:
        raw_candidate_filter = st.selectbox(
            "Candidate selector",
            ["all", "only C/D candidates", "specific asset/horizon"],
            index=0,
            key="raw_export_candidate_filter",
        )
    with raw_col_b:
        st.caption("Raw rows, reconstructed rows, aggregate fallbacks, no-trade rows, and losing trades remain visible.")

    selected_raw_assets = selected_raw_horizons = None
    if raw_candidate_filter == "specific asset/horizon":
        spec_a, spec_b = st.columns(2)
        with spec_a:
            selected_raw_assets = st.multiselect("Specific assets", get_asset_names(), default=get_asset_names(), key="raw_export_assets")
        with spec_b:
            raw_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
            selected_raw_horizon_labels = st.multiselect("Specific horizons", raw_horizon_labels, default=raw_horizon_labels, key="raw_export_horizons")
            selected_raw_horizons = [int(str(label).replace("D", "")) for label in selected_raw_horizon_labels]

    run_raw_export = st.button("🚀 Build Raw Trade Log Export", type="primary")

    if run_raw_export:
        try:
            raw_signal_input = None
            if uploaded_raw is not None:
                uploaded_raw.seek(0)
                raw_signal_input = pd.read_csv(uploaded_raw)
            if raw_source == "Latest Phase 8B/8C/8E/8G session result":
                grading_input = latest_grading_table.copy() if latest_grading_table is not None else None
                full_input = latest_full_evidence.copy() if latest_full_evidence is not None else None
                policy_input = latest_policy_table.copy() if latest_policy_table is not None else None
                recs_input = latest_policy_recs.copy() if latest_policy_recs is not None else None
                ledger_input = latest_ledger_table.copy() if latest_ledger_table is not None else None
            else:
                grading_input = full_input = policy_input = recs_input = ledger_input = None
                if uploaded_grade is not None:
                    uploaded_grade.seek(0)
                    grading_input = pd.read_csv(uploaded_grade)
                if uploaded_full is not None:
                    uploaded_full.seek(0)
                    full_input = pd.read_csv(uploaded_full)
                if uploaded_policy is not None:
                    uploaded_policy.seek(0)
                    policy_input = pd.read_csv(uploaded_policy)
                if uploaded_recs is not None:
                    uploaded_recs.seek(0)
                    recs_input = pd.read_csv(uploaded_recs)
                if uploaded_ledger is not None:
                    uploaded_ledger.seek(0)
                    ledger_input = pd.read_csv(uploaded_ledger)
                if all(obj is None for obj in [grading_input, full_input, policy_input, recs_input, ledger_input, raw_signal_input]):
                    st.error("Upload at least one Phase 8B/8C/8E/8G table or a raw signal/trade CSV.")
                    st.stop()
        except Exception as exc:
            st.error(f"Could not read raw trade log inputs: {exc}")
            st.stop()

        with st.spinner("Building raw signal/trade log export..."):
            try:
                raw_report = run_raw_trade_log_exporter(
                    grading_table=grading_input,
                    full_evidence_table=full_input,
                    policy_sensitivity_table=policy_input,
                    candidate_recommendation_table=recs_input,
                    ledger_table=ledger_input,
                    raw_signal_outputs=raw_signal_input,
                    candidate_filter=raw_candidate_filter,
                    selected_assets=selected_raw_assets,
                    selected_horizons=selected_raw_horizons,
                    configured_assets=get_asset_names(),
                    configured_horizons=DIRECT_FORECAST_HORIZONS,
                )
                st.session_state.raw_trade_log_exporter_report = raw_report
                st.session_state.raw_trade_log_exporter_settings = raw_report.settings
            except Exception as exc:
                st.error(f"Raw trade log exporter failed: {exc}")
                st.stop()

    raw_report = st.session_state.raw_trade_log_exporter_report
    if raw_report is None:
        st.info("Build the raw log export to see whether true row-level signal evidence exists.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('raw_trade_log_exporter_settings') or {}}")
        if not raw_report.raw_log_quality_summary.empty:
            main_warning = str(raw_report.raw_log_quality_summary.iloc[0].get("MainWarning", ""))
            if "NotCalibrationReady" in main_warning or "MissingRawTradeLogs" in main_warning:
                st.warning("Not enough raw trade logs for true calibration yet.")

        st.markdown("### Raw Log Quality Summary")
        st.dataframe(raw_report.raw_log_quality_summary, width="stretch")
        st.download_button(
            "📥 Export Raw Log Quality Summary CSV",
            data=raw_report.raw_log_quality_summary.to_csv(index=False).encode("utf-8"),
            file_name="raw_trade_log_quality_summary.csv",
            mime="text/csv",
            key="raw_export_quality_download",
        )

        tabs = st.tabs([
            "Raw Coverage",
            "Probability Readiness",
            "Raw Log",
            "Trade Outcomes",
            "Benchmark",
            "Drawdown",
            "No-Trade / Skipped",
            "Warnings",
            "Next Actions",
        ])
        raw_tables = [
            ("Asset-Horizon Raw Coverage", raw_report.asset_horizon_raw_coverage_table, "raw_trade_asset_horizon_coverage.csv"),
            ("Probability Outcome Readiness", raw_report.probability_outcome_readiness_table, "raw_trade_probability_readiness.csv"),
            ("Raw Signal/Trade Log", raw_report.raw_signal_trade_log_table, "raw_signal_trade_log.csv"),
            ("Trade Outcome Distribution", raw_report.trade_outcome_distribution_table, "raw_trade_outcome_distribution.csv"),
            ("Benchmark Comparison", raw_report.benchmark_comparison_table, "raw_trade_benchmark_comparison.csv"),
            ("Drawdown During Trade", raw_report.drawdown_during_trade_table, "raw_trade_drawdown.csv"),
            ("No-Trade / Skipped Signal", raw_report.no_trade_skipped_signal_table, "raw_trade_no_trade_skipped.csv"),
            ("Warning Table", raw_report.warning_table, "raw_trade_warnings.csv"),
            ("Next Research Action Table", raw_report.next_research_action_table, "raw_trade_next_actions.csv"),
        ]
        for tab, (title, table, filename) in zip(tabs, raw_tables):
            with tab:
                st.markdown(f"### {title}")
                st.dataframe(table, width="stretch")
                st.download_button(
                    f"📥 Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )


# PAGE: TRADE EVIDENCE LEDGER
# ═══════════════════════════════════════════════════════════════════════

elif page == "📒 Trade Evidence Ledger":
    st.markdown('<p class="main-header">📒 Trade Evidence Ledger</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 8G is research infrastructure only. It builds a multi-asset evidence ledger for future "
        "probability calibration and paper-trading validation. It does not promote candidates or approve live trading."
    )

    latest_calibration_report = st.session_state.get("probability_calibration_report")
    latest_policy_report = st.session_state.get("signal_policy_sensitivity_report")
    latest_evidence_report = st.session_state.get("evidence_expansion_report")
    latest_calibration_summary = getattr(latest_calibration_report, "calibration_summary_table", None) if latest_calibration_report is not None else None
    latest_policy_table = getattr(latest_policy_report, "full_policy_sensitivity_table", None) if latest_policy_report is not None else None
    latest_frontier_table = getattr(latest_policy_report, "coverage_edge_frontier_table", None) if latest_policy_report is not None else None
    latest_full_evidence = getattr(latest_evidence_report, "full_evidence_table", None) if latest_evidence_report is not None else None

    ledger_sources = []
    if (
        (latest_calibration_summary is not None and not latest_calibration_summary.empty)
        or (latest_policy_table is not None and not latest_policy_table.empty)
        or (latest_full_evidence is not None and not latest_full_evidence.empty)
    ):
        ledger_sources.append("Latest Phase 8F/8E/8C session result")
    ledger_sources.append("Uploaded CSVs")
    ledger_source = st.radio("Ledger source", ledger_sources, horizontal=True, key="trade_ledger_source")

    uploaded_calibration = uploaded_policy = uploaded_frontier = uploaded_full = uploaded_raw = None
    if ledger_source == "Uploaded CSVs":
        upload_a, upload_b, upload_c = st.columns(3)
        with upload_a:
            uploaded_calibration = st.file_uploader("Upload Phase 8F calibration summary CSV", type=["csv"], key="ledger_calibration_upload")
            uploaded_policy = st.file_uploader("Upload Phase 8E policy sensitivity CSV", type=["csv"], key="ledger_policy_upload")
        with upload_b:
            uploaded_frontier = st.file_uploader("Upload Phase 8E frontier CSV", type=["csv"], key="ledger_frontier_upload")
            uploaded_full = st.file_uploader("Upload Phase 8C full evidence CSV", type=["csv"], key="ledger_full_upload")
        with upload_c:
            uploaded_raw = st.file_uploader("Optional raw trade/signal log CSV", type=["csv"], key="ledger_raw_upload")
    else:
        uploaded_raw = st.file_uploader("Optional raw trade/signal log CSV", type=["csv"], key="ledger_session_raw_upload")

    ledger_col_a, ledger_col_b = st.columns(2)
    with ledger_col_a:
        ledger_candidate_filter = st.selectbox(
            "Candidate selector",
            ["all", "specific asset/horizon"],
            index=0,
            key="ledger_candidate_filter",
        )
    with ledger_col_b:
        st.caption("Aggregate-derived rows are retained but clearly marked as limited evidence.")

    selected_ledger_assets = selected_ledger_horizons = None
    if ledger_candidate_filter == "specific asset/horizon":
        spec_a, spec_b = st.columns(2)
        with spec_a:
            selected_ledger_assets = st.multiselect("Specific assets", get_asset_names(), default=get_asset_names(), key="ledger_assets")
        with spec_b:
            ledger_horizon_labels = [f"{h}D" for h in DIRECT_FORECAST_HORIZONS]
            selected_ledger_horizon_labels = st.multiselect("Specific horizons", ledger_horizon_labels, default=ledger_horizon_labels, key="ledger_horizons")
            selected_ledger_horizons = [int(str(label).replace("D", "")) for label in selected_ledger_horizon_labels]

    run_ledger = st.button("🚀 Build Trade Evidence Ledger", type="primary")

    if run_ledger:
        try:
            raw_input = None
            if uploaded_raw is not None:
                uploaded_raw.seek(0)
                raw_input = pd.read_csv(uploaded_raw)
            if ledger_source == "Latest Phase 8F/8E/8C session result":
                calibration_input = latest_calibration_summary.copy() if latest_calibration_summary is not None else None
                policy_input = latest_policy_table.copy() if latest_policy_table is not None else None
                frontier_input = latest_frontier_table.copy() if latest_frontier_table is not None else None
                full_input = latest_full_evidence.copy() if latest_full_evidence is not None else None
            else:
                calibration_input = policy_input = frontier_input = full_input = None
                if uploaded_calibration is not None:
                    uploaded_calibration.seek(0)
                    calibration_input = pd.read_csv(uploaded_calibration)
                if uploaded_policy is not None:
                    uploaded_policy.seek(0)
                    policy_input = pd.read_csv(uploaded_policy)
                if uploaded_frontier is not None:
                    uploaded_frontier.seek(0)
                    frontier_input = pd.read_csv(uploaded_frontier)
                if uploaded_full is not None:
                    uploaded_full.seek(0)
                    full_input = pd.read_csv(uploaded_full)
                if calibration_input is None and policy_input is None and frontier_input is None and full_input is None and raw_input is None:
                    st.error("Upload at least one Phase 8F/8E/8C table or a raw trade log CSV.")
                    st.stop()
        except Exception as exc:
            st.error(f"Could not read trade evidence ledger inputs: {exc}")
            st.stop()

        with st.spinner("Building trade evidence ledger..."):
            try:
                ledger_report = run_trade_evidence_ledger(
                    calibration_summary_table=calibration_input,
                    policy_sensitivity_table=policy_input,
                    coverage_edge_frontier_table=frontier_input,
                    full_evidence_table=full_input,
                    raw_trade_logs=raw_input,
                    candidate_filter=ledger_candidate_filter,
                    selected_assets=selected_ledger_assets,
                    selected_horizons=selected_ledger_horizons,
                    configured_assets=get_asset_names(),
                    configured_horizons=DIRECT_FORECAST_HORIZONS,
                )
                st.session_state.trade_evidence_ledger_report = ledger_report
                st.session_state.trade_evidence_ledger_settings = ledger_report.settings
            except Exception as exc:
                st.error(f"Trade evidence ledger failed: {exc}")
                st.stop()

    ledger_report = st.session_state.trade_evidence_ledger_report
    if ledger_report is None:
        st.info("Build the ledger to see whether trade-level probability evidence exists or evidence is still aggregate-derived.")
    else:
        st.caption(f"Last run settings: {st.session_state.get('trade_evidence_ledger_settings') or {}}")
        if not ledger_report.ledger_quality_summary.empty:
            main_warning = str(ledger_report.ledger_quality_summary.iloc[0].get("MainWarning", ""))
            if "NotCalibrationReady" in main_warning or "MissingRawTradeLogs" in main_warning:
                st.warning("Not enough trade-level evidence for true calibration yet.")

        st.markdown("### Ledger Quality Summary")
        st.dataframe(ledger_report.ledger_quality_summary, width="stretch")
        st.download_button(
            "📥 Export Ledger Quality Summary CSV",
            data=ledger_report.ledger_quality_summary.to_csv(index=False).encode("utf-8"),
            file_name="trade_evidence_ledger_quality_summary.csv",
            mime="text/csv",
            key="ledger_quality_download",
        )

        tabs = st.tabs([
            "Coverage",
            "Probability Outcomes",
            "Ledger",
            "Trade Outcomes",
            "Benchmark",
            "Drawdown",
            "Warnings",
            "Next Actions",
        ])
        ledger_tables = [
            ("Asset-Horizon Evidence Coverage", ledger_report.asset_horizon_coverage_table, "trade_ledger_asset_horizon_coverage.csv"),
            ("Probability Outcome Availability", ledger_report.probability_outcome_availability_table, "trade_ledger_probability_outcomes.csv"),
            ("Trade Evidence Ledger", ledger_report.ledger_table, "trade_evidence_ledger.csv"),
            ("Trade Outcome Distribution", ledger_report.trade_outcome_distribution_table, "trade_ledger_outcome_distribution.csv"),
            ("Benchmark Outcome Table", ledger_report.benchmark_outcome_table, "trade_ledger_benchmark_outcomes.csv"),
            ("Drawdown Outcome Table", ledger_report.drawdown_outcome_table, "trade_ledger_drawdown_outcomes.csv"),
            ("Ledger Warning Table", ledger_report.ledger_warning_table, "trade_ledger_warnings.csv"),
            ("Next Research Action Table", ledger_report.next_research_action_table, "trade_ledger_next_actions.csv"),
        ]
        for tab, (title, table, filename) in zip(tabs, ledger_tables):
            with tab:
                st.markdown(f"### {title}")
                st.dataframe(table, width="stretch")
                st.download_button(
                    f"📥 Export {title} CSV",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"{filename}_download",
                )


# PAGE: SIGNAL ENGINE
# ════════════════════════════════════════════════════════════════

elif page == "📡 Signal Engine":
    st.markdown('<p class="main-header">📡 Direction-First Signal Engine</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.warning(
        "Phase 7 uses Phase 6 direct direction probabilities, not exact price forecasts. "
        "Threshold sweeps are research/reporting only and must not be treated as production threshold tuning on test data."
    )

    sig_col_a, sig_col_b, sig_col_c, sig_col_d = st.columns(4)
    with sig_col_a:
        signal_asset = st.selectbox(
            "Asset",
            get_asset_names(),
            index=get_asset_names().index(selected_asset) if selected_asset in get_asset_names() else 0,
            key="signal_engine_asset",
            help="Crude Oil 5D is a smoke path only. The engine supports all configured assets.",
        )
    with sig_col_b:
        signal_horizon_label = st.selectbox(
            "Horizon",
            [f"{h}D" for h in DIRECT_FORECAST_HORIZONS],
            index=1,
            key="signal_engine_horizon",
        )
        signal_horizon = int(str(signal_horizon_label).replace("D", ""))
    with sig_col_c:
        signal_depth = st.selectbox(
            "Model depth",
            ["core", "fast"],
            index=0,
            key="signal_engine_depth",
        )
    with sig_col_d:
        signal_use_phase5 = st.checkbox(
            "Use Phase 5 features",
            value=True,
            key="signal_engine_phase5",
        )

    rule_col_a, rule_col_b, rule_col_c, rule_col_d, rule_col_e = st.columns(5)
    with rule_col_a:
        signal_backtest_style = st.selectbox(
            "Backtest style",
            ["non_overlapping_realistic", "overlapping_research"],
            index=0,
            help="Non-overlapping is the realistic trade mode. Overlapping is an optimistic research comparison.",
            key="signal_engine_backtest_style",
        )
    with rule_col_b:
        signal_mode = st.selectbox(
            "Signal mode",
            ["long_only", "long_short", "avoid_only"],
            index=0,
            key="signal_engine_mode",
        )
    with rule_col_c:
        long_threshold = st.slider(
            "Long threshold",
            min_value=0.50,
            max_value=0.80,
            value=0.55,
            step=0.01,
            key="signal_engine_long_threshold",
        )
    with rule_col_d:
        short_threshold = st.slider(
            "Short threshold",
            min_value=0.20,
            max_value=0.50,
            value=0.45,
            step=0.01,
            disabled=signal_mode != "long_short",
            key="signal_engine_short_threshold",
        )
    with rule_col_e:
        transaction_cost_pct = st.number_input(
            "Transaction cost %",
            min_value=0.0,
            max_value=2.0,
            value=0.10,
            step=0.01,
            key="signal_engine_transaction_cost",
        )

    cooldown_rows = st.number_input(
        "Cooldown rows after trade exit",
        min_value=0,
        max_value=30,
        value=0,
        step=1,
        disabled=signal_backtest_style != "non_overlapping_realistic",
        help="Only used by non-overlapping realistic mode.",
        key="signal_engine_cooldown_rows",
    )

    policy_col_a, policy_col_b = st.columns(2)
    with policy_col_a:
        threshold_policy = st.selectbox(
            "Threshold policy",
            ["validation_locked", "manual_threshold"],
            index=0,
            help="Validation-locked sweeps thresholds only on the first chronological segment and evaluates the selected threshold once on the locked test segment.",
            key="signal_engine_threshold_policy",
        )
    with policy_col_b:
        validation_fraction_pct = st.slider(
            "Validation segment %",
            min_value=30,
            max_value=70,
            value=50,
            step=5,
            disabled=threshold_policy != "validation_locked",
            help="Chronological share of the available Phase 6 out-of-sample signal rows used for threshold selection.",
            key="signal_engine_validation_fraction",
        )

    if threshold_policy == "validation_locked":
        st.info(
            "Validation-locked mode splits the available Phase 6 out-of-sample signal rows into validation and locked-test segments. "
            "Thresholds are selected on validation only, then evaluated once on locked test."
        )

    if signal_backtest_style == "overlapping_research":
        st.warning(
            "Overlapping research mode can be optimistic for multi-day horizons because daily 5D/10D/20D signals overlap. "
            "Use non-overlapping realistic mode before treating a signal as tradable."
        )

    run_signal = st.button("🚀 Run Signal Backtest", type="primary")

    if run_signal:
        if threshold_policy == "manual_threshold" and signal_mode == "long_short" and short_threshold >= long_threshold:
            st.error("Short threshold must be lower than long threshold.")
            st.stop()

        with st.spinner("Training Phase 6 direct model and running signal backtest..."):
            try:
                raw_df = load_raw_data("2015-01-01", use_cache=True)
                signal_output = run_direct_forecast_signal_output(
                    raw_df=raw_df,
                    asset_name=signal_asset,
                    horizon=signal_horizon,
                    model_depth=signal_depth,
                    use_phase5_features=signal_use_phase5,
                )
                if threshold_policy == "validation_locked":
                    signal_result = run_validation_locked_signal_engine(
                        signal_output=signal_output,
                        mode=signal_mode,
                        transaction_cost=float(transaction_cost_pct) / 100.0,
                        backtest_style=signal_backtest_style,
                        cooldown=int(cooldown_rows),
                        validation_fraction=float(validation_fraction_pct) / 100.0,
                    )
                else:
                    signal_result = run_signal_engine(
                        signal_output=signal_output,
                        long_threshold=long_threshold,
                        short_threshold=short_threshold if signal_mode == "long_short" else min(0.45, long_threshold - 0.05),
                        mode=signal_mode,
                        transaction_cost=float(transaction_cost_pct) / 100.0,
                        backtest_style=signal_backtest_style,
                        cooldown=int(cooldown_rows),
                    )
                st.session_state.signal_engine_output = signal_output
                st.session_state.signal_engine_result = signal_result
                st.session_state.signal_engine_settings = {
                    "asset": signal_asset,
                    "horizon": signal_horizon,
                    "model_depth": signal_depth,
                    "use_phase5_features": signal_use_phase5,
                    "threshold_policy": threshold_policy,
                    "validation_fraction_pct": validation_fraction_pct,
                    "backtest_style": signal_backtest_style,
                    "mode": signal_mode,
                    "long_threshold": long_threshold,
                    "short_threshold": short_threshold,
                    "transaction_cost_pct": transaction_cost_pct,
                    "cooldown_rows": int(cooldown_rows),
                }
            except Exception as exc:
                st.error(f"Signal engine failed: {exc}")
                st.stop()

    signal_output = st.session_state.signal_engine_output
    signal_result = st.session_state.signal_engine_result

    if signal_output is None or signal_result is None:
        st.info("Run the signal backtest to evaluate thresholded P(up) signals.")
    else:
        settings = st.session_state.get("signal_engine_settings") or {}
        metrics = signal_result.metrics
        st.caption(f"Last run settings: {settings}")
        st.caption(f"Phase 6 direction model: {signal_output.model_name}")

        if signal_output.feature_leakage_columns:
            st.error(f"Future target columns leaked into features: {signal_output.feature_leakage_columns}")
        else:
            st.success("No future_return_*, future_direction_*, or future_realized_vol_* columns are present in the model features.")

        result_style = str(metrics.get("BacktestStyle", ""))
        if result_style == "overlapping_research":
            st.warning("This result uses overlapping research mode. It is useful for diagnostics but can overstate multi-day signal performance.")

        is_validation_locked = str(metrics.get("ThresholdPolicy", "")) == "validation_locked"
        if is_validation_locked:
            st.info(
                "Validation-locked result: threshold selection used only the validation segment inside the available Phase 6 out-of-sample output. "
                "The metrics below are from the locked test segment."
            )
            lock_a, lock_b, lock_c, lock_d = st.columns(4)
            lock_a.metric("Locked Long", f"{float(metrics.get('SelectedLongThreshold', metrics.get('LongThreshold', 0.0))):.2f}")
            lock_b.metric("Locked Short", f"{float(metrics.get('SelectedShortThreshold', metrics.get('ShortThreshold', 0.0))):.2f}")
            lock_c.metric("Validation Score", f"{float(metrics.get('ValidationSelectionScore', 0.0)):.2f}")
            lock_d.metric("Locked Test Rows", int(metrics.get("LockedTestRows", metrics.get("Rows", 0))))

            locked_warning = str(metrics.get("LockedTestWarning", "") or "")
            validation_warning = str(metrics.get("ValidationWarning", "") or "")
            if validation_warning:
                st.warning(f"Validation warning: {validation_warning}")
            if locked_warning:
                st.warning(f"Locked-test warning: {locked_warning}")

        st.markdown("### Signal Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades / Signals", int(metrics.get("NumberOfTrades", metrics.get("SignalCount", 0))))
        m2.metric("Frequency", f"{float(metrics.get('TradeFrequency_%', metrics.get('SignalFrequency_%', 0.0))):.2f}%")
        m3.metric("Win Rate", f"{float(metrics.get('WinRate_%', metrics.get('WinRateActive_%', 0.0))):.2f}%")
        m4.metric("Verdict", str(metrics.get("ThresholdVerdict", "")))

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Strategy Return", f"{float(metrics.get('TotalCompoundedReturn_%', metrics.get('StrategyTotalReturn_%', 0.0))):+.2f}%")
        r2.metric("Passive Hold", f"{float(metrics.get('BuyHoldReturn_%', 0.0)):+.2f}%")
        r3.metric("Vs Passive Hold", f"{float(metrics.get('StrategyMinusBuyHold_%', 0.0)):+.2f}%")
        r4.metric("Exposure", f"{float(metrics.get('Exposure_%', metrics.get('SignalFrequency_%', 0.0))):.2f}%")

        metric_df = pd.DataFrame([metrics]).T.reset_index()
        metric_df.columns = ["Metric", "Value"]
        st.dataframe(metric_df, width="stretch")

        warnings_text = str(metrics.get("Warnings", "") or "")
        if warnings_text:
            st.error(f"Signal warning: {warnings_text}")
        if float(metrics.get("NumberOfTrades", metrics.get("SignalCount", 0))) < 5:
            st.warning("Very few trades/signals. Treat this as insufficient evidence.")
        if float(metrics.get("StrategyMinusBuyHold_%", 0.0)) <= 0:
            st.error("Strategy failed the passive-hold baseline on this test split.")

        tab_lock, tab_signals, tab_sweep, tab_phase6 = st.tabs([
            "Validation Lock",
            "Trade / Signal Log",
            "Threshold Sweep",
            "Phase 6 Model",
        ])

        with tab_lock:
            if is_validation_locked:
                st.markdown("### Validation-Locked Threshold Selection")
                selected_threshold = getattr(signal_result, "selected_threshold", {}) or {}
                if selected_threshold:
                    selected_df = pd.DataFrame([selected_threshold]).T.reset_index()
                    selected_df.columns = ["Field", "Value"]
                    st.dataframe(selected_df, width="stretch")

                st.markdown("### Validation vs Locked Test")
                comparison = getattr(signal_result, "validation_test_comparison", pd.DataFrame())
                if comparison is not None and not comparison.empty:
                    st.dataframe(comparison, width="stretch")
                    comparison_csv = comparison.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Export Validation vs Test CSV",
                        data=comparison_csv,
                        file_name=f"signal_engine_{signal_output.asset}_{signal_output.horizon}D_validation_locked_comparison.csv".replace(" ", "_"),
                        mime="text/csv",
                    )
                else:
                    st.info("No validation/test comparison table available.")

                st.markdown("### Validation Threshold Table")
                validation_sweep = getattr(signal_result, "validation_sweep", pd.DataFrame())
                if validation_sweep is not None and not validation_sweep.empty:
                    st.dataframe(validation_sweep, width="stretch")
                    validation_csv = validation_sweep.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Export Validation Threshold CSV",
                        data=validation_csv,
                        file_name=f"signal_engine_{signal_output.asset}_{signal_output.horizon}D_validation_thresholds.csv".replace(" ", "_"),
                        mime="text/csv",
                    )
                else:
                    st.info("No validation threshold table available.")
            else:
                st.info("Manual threshold mode did not run validation-locked selection.")

        with tab_signals:
            st.markdown("### Realistic Trade Log" if result_style == "non_overlapping_realistic" else "### Overlapping Research Signal Log")
            st.dataframe(signal_result.signal_frame, width="stretch")
            signal_csv = signal_result.signal_frame.to_csv(index=True).encode("utf-8")
            st.download_button(
                "📥 Export Trade/Signal Log CSV",
                data=signal_csv,
                file_name=f"signal_engine_{signal_output.asset}_{signal_output.horizon}D_signals.csv".replace(" ", "_"),
                mime="text/csv",
            )

        with tab_sweep:
            if is_validation_locked:
                st.markdown("### Validation Threshold Table")
                st.caption("This table was evaluated only on the validation segment and used for locked threshold selection. Locked-test rows were not swept.")
                sweep_table = getattr(signal_result, "validation_sweep", pd.DataFrame())
            else:
                st.markdown("### Research-Only Threshold Sweep")
                st.caption("This sweep is evaluated on the test split for reporting only. Do not select production thresholds from it without separate validation.")
                sweep_table = signal_result.threshold_sweep

            if sweep_table is not None and not sweep_table.empty:
                st.dataframe(sweep_table, width="stretch")
                sweep_csv = sweep_table.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Threshold Sweep CSV",
                    data=sweep_csv,
                    file_name=f"signal_engine_{signal_output.asset}_{signal_output.horizon}D_threshold_sweep.csv".replace(" ", "_"),
                    mime="text/csv",
                )
            else:
                st.info("No threshold sweep table available.")

        with tab_phase6:
            st.markdown("### Phase 6 Direct Model Leaderboard")
            st.dataframe(signal_output.leaderboard, width="stretch")
            st.markdown("### Phase 6 Direction Baselines")
            st.dataframe(signal_output.baseline_board, width="stretch")


# ════════════════════════════════════════════════════════════════
# PAGE: 30-DAY FORECAST
# ════════════════════════════════════════════════════════════════

elif page == "📅 30-Day Forecast":
    st.warning(
        "Legacy diagnostic view - recursive price chaining is not the primary forecasting evidence. "
        "Use Direct Horizon Scanner, Walk-Forward ML Replay, and Model Edge Benchmark Lab for multi-asset validation."
    )
    st.markdown('<p class="main-header">📅 30-Day Price Forecast</p>', unsafe_allow_html=True)
    st.markdown("---")

    if not st.session_state.trained:
        st.warning("⚠️ No models trained yet. Go to **Train Models** first.")
    else:
        _stop_if_asset_mismatch(selected_asset)
        trainer = st.session_state.trainer
        pp = st.session_state.pp
        data = st.session_state.data
        df_features = st.session_state.df_features

        model_name = st.selectbox("Select a model for forecasting", list(trainer.results.keys()))
        n_days = st.slider("Forecast horizon (days)", 5, 60, 30)

        if st.button("🔮 Generate Forecast", type="primary"):
            with st.spinner(f"Generating {n_days}-day forecast..."):
                result = trainer.results[model_name]
                predictor = Predictor(model=result.model, preprocessor=pp, is_sequence_model=False)

                active_target_col = getattr(pp, "target_col", target_col)
                ti = TechnicalIndicators(prefix=_target_prefix(active_target_col))
                fe = FeatureEngineer(target_col=active_target_col)

                try:
                    forecast_df = predictor.forecast(
                        df_features, feature_cols=data.feature_cols, n_days=n_days,
                        indicators_engine=ti, feature_engineer=fe,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    st.info(
                        "The forecast was stopped before model inference. Retrain the selected "
                        "asset model so its saved feature contract matches the current pipeline."
                    )
                    st.stop()
                vol = df_features["Daily_Return"].std()
                forecast_df = predictor.add_confidence_bands(forecast_df, historical_volatility=vol)

            st.success(f"✔ {n_days}-day {selected_asset} forecast generated using {model_name}")

            viz = Visualizer()
            fig = viz.plot_forecast_plotly(
                forecast_df,
                df_features,
                target_col=active_target_col,
                asset_label=selected_asset,
                n_history_days=90,
            )
            st.plotly_chart(fig, width="stretch")

            st.markdown("### Forecast Table")
            st.dataframe(forecast_df, width="stretch")

            csv = forecast_df.to_csv().encode("utf-8")
            st.download_button(
                "📥 Download Forecast (CSV)",
                data=csv,
                file_name=f"{_safe_filename_part(selected_asset)}_{_safe_filename_part(model_name)}_{n_days}day_forecast.csv",
                mime="text/csv",
            )
