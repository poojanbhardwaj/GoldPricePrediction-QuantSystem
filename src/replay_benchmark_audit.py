"""Phase 18 replay-integrated benchmark and edge audit.

This module compares Phase 17 historical replay exports against simple
time-safe baselines. Proxy replay evidence stays labeled as proxy evidence; it
is never treated as proof of true historical trained-model edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.asset_config import get_asset_names, get_target_column
from src.artifact_store import resolve_artifact, save_phase_artifacts


REPLAY_BENCHMARK_AUDIT_PHASE_NAME = "phase18_replay_benchmark_audit"
REPLAY_AUDIT_HORIZONS: Tuple[int, ...] = (1, 5, 10, 20, 30)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKET_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"
DEFAULT_COST_SCENARIOS_BPS: Tuple[float, ...] = (0.0, 5.0, 10.0, 25.0, 50.0)
REPLAY_STRATEGY_PROXY = "HistoricalSignalProxyReplay"
REPLAY_STRATEGY_MODEL = "HistoricalModelPredictionReplay"
RETURN_SANITY_EPSILON = 1e-12
PROXY_SOURCE_MARKERS = (
    "historicalsignalproxyreplay",
    "proxysignalreplay",
    "proxyonly",
    "historical proxy replay",
    "historicalproxyreplay",
)
MODEL_LIKE_STRATEGY_MARKERS = (
    "historicalmodelriskreplay",
    "historicalmodelpredictionreplay",
    "modelriskreplay",
    "mlmodelreplay",
    "truemodelreplay",
)
SIMPLE_BASELINES = {
    "NoExposureBaseline",
    "HoldOnlyBenchmark",
    "MovingAverageCrossover",
    "MomentumBaseline",
    "MeanReversionBaseline",
    "VolatilityScaledBaseline",
    "RandomBaseline",
}

REPLAY_BENCHMARK_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "ReplaySource",
    "ModelReplayQuality",
    "ReplayBenchmarkVerdict",
    "ProxyBeatsHoldOnly",
    "ProxyBeatsMomentum",
    "ProxyBeatsMovingAverage",
    "ProxyBeatsNoExposure",
    "ProxyBeatsRandomMedian",
    "AssetsWithProxyEdge",
    "AssetsBenchmarkDominated",
    "HorizonsWithProxyEdge",
    "HorizonsBenchmarkDominated",
    "MainReason",
    "MainLimitation",
    "RecommendedNextStep",
)

REPLAY_VS_BASELINE_LEADERBOARD_COLUMNS: Tuple[str, ...] = (
    "Rank",
    "StrategyName",
    "Asset",
    "Horizon",
    "ReplaySource",
    "EvaluationMode",
    "TotalReturnPct",
    "NetReturnPct",
    "AnnualizedReturnPct",
    "VolatilityPct",
    "SharpeProxy",
    "MaxDrawdownPct",
    "WinRatePct",
    "TradeCount",
    "ExposurePct",
    "CostImpactPct",
    "ComparableHistorical",
    "BenchmarkRole",
    "Verdict",
    "InvalidReturnRows",
    "DataQualityFlag",
    "ReturnSanityStatus",
)

REPLAY_ASSET_EDGE_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "ProxyReplayReturnPct",
    "BestBaseline",
    "BestBaselineReturnPct",
    "ProxyVsBestBaselinePct",
    "ProxyRank",
    "ProxyBeatsBestBaseline",
    "ProxyBeatsNoExposure",
    "ProxyBeatsRandomMedian",
    "DrawdownComparison",
    "EdgeVerdict",
    "MainReason",
)

REPLAY_ASSET_HORIZON_EDGE_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "ProxyReplayReturnPct",
    "BestBaseline",
    "BestBaselineReturnPct",
    "ProxyVsBestBaselinePct",
    "ProxyBeatsBestBaseline",
    "ProxyBeatsNoExposure",
    "ProxyBeatsRandomMedian",
    "ProxyDrawdownPct",
    "BaselineDrawdownPct",
    "TradeCount",
    "EdgeVerdict",
    "MainReason",
)

REPLAY_DOMINANCE_FAILURE_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "DominatingBaseline",
    "BaselineReturnPct",
    "ProxyReturnPct",
    "GapPct",
    "BaselineDrawdownPct",
    "ProxyDrawdownPct",
    "DominanceReason",
    "RequiredImprovement",
)

REPLAY_STRENGTH_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "ProxyReturnPct",
    "BestBaselineReturnPct",
    "ImprovementPct",
    "DrawdownImprovementPct",
    "RiskAdjustedImprovement",
    "TradeCount",
    "EvidenceStrength",
    "MainStrengthReason",
)

REPLAY_RANDOM_COMPARISON_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "RandomSimulationCount",
    "RandomMedianReturnPct",
    "RandomP25ReturnPct",
    "RandomP75ReturnPct",
    "RandomBestReturnPct",
    "ProxyReturnPct",
    "ProxyBeatsRandomMedian",
    "ProxyPercentileVsRandom",
    "Explanation",
)

REPLAY_COST_ROBUSTNESS_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "CostBps",
    "ProxyNetReturnPct",
    "ReturnLostToCostsPct",
    "CostFragile",
    "BreakEvenCostBps",
    "Explanation",
)

REPLAY_DRAWDOWN_COMPARISON_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "ProxyMaxDrawdownPct",
    "BestBaselineMaxDrawdownPct",
    "DrawdownImprovementPct",
    "ProxyReturnPct",
    "BestBaselineReturnPct",
    "ReturnDrawdownTradeoff",
    "Explanation",
)

REPLAY_QUALITY_GATE_COLUMNS: Tuple[str, ...] = (
    "GateName",
    "Passed",
    "Severity",
    "Explanation",
)

REPLAY_REAL_CAPITAL_READINESS_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "RealCapitalReadiness",
    "BlockingReasons",
    "RequiredEvidenceBeforeEligibility",
    "ProxyEvidenceStatus",
    "ForwardEvidenceStatus",
    "CalibrationStatus",
    "BenchmarkStatus",
    "RecommendedMode",
)

REPLAY_NEXT_ACTION_COLUMNS: Tuple[str, ...] = (
    "Rank",
    "Action",
    "WhyItMatters",
    "AffectedAssets",
    "AffectedHorizons",
    "ExpectedBenefit",
    "Urgency",
    "DependsOn",
)

REPLAY_INPUT_SOURCE_COLUMNS: Tuple[str, ...] = (
    "SourceName",
    "Available",
    "Rows",
    "Columns",
    "LastDate",
    "MissingCriticalColumns",
    "Notes",
)

INPUT_SPECS: Dict[str, Tuple[str, str, bool]] = {
    "phase16_replay_export_table": ("phase17_historical_model_replay", "phase16_replay_export_table", False),
    "replay_summary_table": ("phase17_historical_model_replay", "replay_summary_table", False),
    "replay_quality_checks": ("phase17_historical_model_replay", "replay_quality_checks", False),
    "replay_benchmark_ready_table": ("phase17_historical_model_replay", "replay_benchmark_ready_table", False),
    "historical_replay_performance": ("phase17_historical_model_replay", "historical_replay_performance", False),
    "replay_asset_horizon_matrix": ("phase17_historical_model_replay", "replay_asset_horizon_matrix", False),
    "replay_exposure_cap_table": ("phase17_historical_model_replay", "replay_exposure_cap_table", False),
    "replay_warnings_table": ("phase17_historical_model_replay", "replay_warnings_table", False),
}


@dataclass
class ReplayBenchmarkAuditReport:
    replay_benchmark_summary_table: pd.DataFrame
    replay_vs_baseline_leaderboard: pd.DataFrame
    replay_asset_edge_table: pd.DataFrame
    replay_asset_horizon_edge_table: pd.DataFrame
    replay_dominance_failures_table: pd.DataFrame
    replay_strength_table: pd.DataFrame
    replay_random_comparison_table: pd.DataFrame
    replay_cost_robustness_table: pd.DataFrame
    replay_drawdown_comparison_table: pd.DataFrame
    replay_quality_gate_table: pd.DataFrame
    replay_real_capital_readiness_table: pd.DataFrame
    replay_next_actions_table: pd.DataFrame
    replay_benchmark_input_sources_table: pd.DataFrame
    artifact_input_source_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    settings: Dict[str, Any] = field(default_factory=dict)
    saved_artifacts: Dict[str, Any] = field(default_factory=dict)


def _to_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame(value)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _normalise_horizon(df: Any) -> pd.DataFrame:
    out = _to_frame(df)
    if out.empty:
        return out
    if "Horizon" in out.columns:
        out["Horizon"] = out["Horizon"].astype(str).str.replace("D", "", regex=False)
        out["Horizon"] = pd.to_numeric(out["Horizon"], errors="coerce")
    return out


def _prepare_market_data(market_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = _to_frame(market_data)
    if df.empty:
        return df
    df = df.copy()
    date_col = next((col for col in ["Date", "date", "Datetime", "Timestamp"] if col in df.columns), None)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    else:
        parsed = pd.to_datetime(df.index, errors="coerce")
        if not parsed.isna().all():
            df.index = parsed
            df = df[~df.index.isna()].sort_index()
    return df


def _load_project_market_data() -> Optional[pd.DataFrame]:
    if not DEFAULT_MARKET_DATA_PATH.exists():
        return None
    try:
        return pd.read_csv(DEFAULT_MARKET_DATA_PATH)
    except Exception:
        return None


def _series(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _asset_warning_label(asset: str) -> str:
    clean = "".join(ch for ch in str(asset) if ch.isalnum())
    return f"{clean or 'Asset'}ReturnDataRequiresCare"


def _price_return_sanity(price: pd.Series, asset: str) -> Tuple[pd.Series, int, str, str]:
    """Return clean daily returns and explicit flags for skipped invalid transitions."""
    price = pd.to_numeric(price, errors="coerce").sort_index()
    if len(price) < 2:
        return pd.Series(dtype=float), 0, "InsufficientCleanReturnData", "InsufficientCleanReturnData"

    previous = price.shift(1)
    raw_return = price / previous - 1.0
    transition_mask = previous.notna()
    valid_transition = (
        transition_mask
        & price.notna()
        & previous.notna()
        & np.isfinite(price)
        & np.isfinite(previous)
        & (price > 0)
        & (previous > 0)
        & np.isfinite(raw_return)
        & (raw_return > -1.0 - RETURN_SANITY_EPSILON)
    )
    invalid_rows = int((transition_mask & ~valid_transition).sum())
    clean = raw_return[valid_transition].astype(float)
    if clean.empty:
        flags = ["InvalidReturnData", "InsufficientCleanReturnData"]
        if invalid_rows:
            flags.extend(["NonPositivePriceReturnInvalid", _asset_warning_label(asset)])
        return clean, invalid_rows, "; ".join(dict.fromkeys(flags)), "InsufficientCleanReturnData"
    if invalid_rows:
        flags = [
            "NonPositivePriceReturnInvalid",
            "InvalidReturnRowsSkipped",
            _asset_warning_label(asset),
        ]
        return clean, invalid_rows, "; ".join(dict.fromkeys(flags)), "InvalidReturnRowsSkipped"
    return clean, 0, "", "CleanReturnData"


def _leaderboard_row(
    *,
    strategy: str,
    asset: str,
    horizon: int,
    replay_source: str,
    evaluation_mode: str,
    metrics: Dict[str, float],
    comparable: bool,
    benchmark_role: str,
    verdict: str,
    invalid_rows: int = 0,
    data_quality_flag: str = "",
    return_sanity_status: str = "CleanReturnData",
) -> Dict[str, Any]:
    return {
        "Rank": 0,
        "StrategyName": strategy,
        "Asset": asset,
        "Horizon": int(horizon),
        "ReplaySource": replay_source,
        "EvaluationMode": evaluation_mode,
        **metrics,
        "ComparableHistorical": bool(comparable),
        "BenchmarkRole": benchmark_role,
        "Verdict": verdict,
        "InvalidReturnRows": int(invalid_rows),
        "DataQualityFlag": data_quality_flag,
        "ReturnSanityStatus": return_sanity_status,
    }


def _empty_metrics() -> Dict[str, float]:
    return {
        "TotalReturnPct": 0.0,
        "NetReturnPct": 0.0,
        "AnnualizedReturnPct": 0.0,
        "VolatilityPct": 0.0,
        "SharpeProxy": 0.0,
        "MaxDrawdownPct": 0.0,
        "WinRatePct": 0.0,
        "TradeCount": 0,
        "ExposurePct": 0.0,
        "CostImpactPct": 0.0,
    }


def _resolve_inputs(
    use_artifact_store: bool,
    prefer_uploaded: bool,
    uploaded_overrides: Optional[Dict[str, Any]],
    direct_tables: Dict[str, Any],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    uploaded_overrides = uploaded_overrides or {}
    tables: Dict[str, pd.DataFrame] = {}
    rows: List[Dict[str, Any]] = []
    for key, (phase, artifact, required) in INPUT_SPECS.items():
        aliases = [key, artifact, f"phase17_{artifact}", "phase17_phase16_replay_export" if key == "phase16_replay_export_table" else ""]
        direct = next((direct_tables[name] for name in aliases if name and name in direct_tables and direct_tables[name] is not None), None)
        if direct is not None:
            df = _normalise_horizon(direct)
            tables[key] = df
            rows.append({"Artifact": artifact, "Phase": phase, "Source": "DirectInput", "RunId": "", "Rows": int(len(df)), "CreatedAt": "", "Status": "Loaded", "Path": ""})
            continue
        uploaded = uploaded_overrides.get(key) or uploaded_overrides.get(artifact)
        if use_artifact_store or uploaded is not None:
            resolved = resolve_artifact(phase, artifact, uploaded_file=uploaded, prefer_uploaded=prefer_uploaded, required=required)
            data = resolved.get("Data")
            tables[key] = _normalise_horizon(data) if data is not None else pd.DataFrame()
            rows.append({k: v for k, v in resolved.items() if k != "Data"})
        else:
            tables[key] = pd.DataFrame()
            rows.append({"Artifact": artifact, "Phase": phase, "Source": "Missing", "RunId": "", "Rows": 0, "CreatedAt": "", "Status": "MissingOptional", "Path": ""})
    return tables, pd.DataFrame(rows, columns=["Artifact", "Source", "RunId", "Rows", "CreatedAt", "Status", "Phase", "Path"])


def _input_sources(market_data: pd.DataFrame, replay_export: pd.DataFrame, tables: Dict[str, pd.DataFrame], assets: Iterable[str], project_data_used: bool) -> pd.DataFrame:
    critical = [get_target_column(asset) for asset in assets]
    missing = [col for col in critical if col not in market_data.columns]
    rows = [
        {
            "SourceName": "market_data",
            "Available": bool(not market_data.empty),
            "Rows": int(len(market_data)),
            "Columns": int(len(market_data.columns)) if not market_data.empty else 0,
            "LastDate": str(market_data.index.max().date()) if not market_data.empty and hasattr(market_data.index.max(), "date") else "",
            "MissingCriticalColumns": "; ".join(missing),
            "Notes": "Loaded from project master dataset." if project_data_used else "Loaded from direct input or upload.",
        },
        {
            "SourceName": "phase17_phase16_replay_export",
            "Available": bool(not replay_export.empty),
            "Rows": int(len(replay_export)),
            "Columns": int(len(replay_export.columns)) if not replay_export.empty else 0,
            "LastDate": str(pd.to_datetime(replay_export["Date"], errors="coerce").max().date()) if not replay_export.empty and "Date" in replay_export.columns else "",
            "MissingCriticalColumns": "; ".join(col for col in ["Date", "Asset", "Horizon", "StrategyName", "ExposurePct", "StrategyReturnPct", "ComparableHistorical", "ReplaySource"] if col not in replay_export.columns),
            "Notes": "Phase 17 replay export used for proxy strategy comparison.",
        },
    ]
    for key, df in tables.items():
        rows.append(
            {
                "SourceName": key,
                "Available": bool(not df.empty),
                "Rows": int(len(df)),
                "Columns": int(len(df.columns)) if not df.empty else 0,
                "LastDate": "",
                "MissingCriticalColumns": "",
                "Notes": "Optional Phase 17 supporting artifact.",
            }
        )
    return pd.DataFrame(rows, columns=list(REPLAY_INPUT_SOURCE_COLUMNS))


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    dd = equity / equity.cummax() - 1.0
    return float(dd.min() * 100.0)


def _metrics_from_returns(returns: pd.Series, exposure: pd.Series, cost_bps: float) -> Dict[str, float]:
    returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    exposure = pd.to_numeric(exposure, errors="coerce").fillna(0.0).clip(lower=0.0)
    exposure = exposure.reindex(returns.index).fillna(0.0)
    turnover = exposure.diff().abs().fillna(exposure.abs())
    costs = turnover * (float(cost_bps) / 10000.0)
    net = returns - costs
    gross_equity = (1.0 + returns).cumprod()
    net_equity = (1.0 + net).cumprod()
    total = (gross_equity.iloc[-1] - 1.0) * 100.0 if not gross_equity.empty else 0.0
    net_total = (net_equity.iloc[-1] - 1.0) * 100.0 if not net_equity.empty else 0.0
    annual = ((1.0 + net_total / 100.0) ** (252.0 / max(len(net), 1)) - 1.0) * 100.0 if net_total > -100 and len(net) > 0 else 0.0
    vol = float(net.std(ddof=0) * np.sqrt(252) * 100.0) if len(net) > 1 else 0.0
    sharpe = float((net.mean() / net.std(ddof=0)) * np.sqrt(252)) if len(net) > 1 and net.std(ddof=0) > 0 else 0.0
    active = exposure > 0
    return {
        "TotalReturnPct": round(float(total), 4),
        "NetReturnPct": round(float(net_total), 4),
        "AnnualizedReturnPct": round(float(annual), 4),
        "VolatilityPct": round(vol, 4),
        "SharpeProxy": round(sharpe, 4),
        "MaxDrawdownPct": round(_max_drawdown(net_equity), 4),
        "WinRatePct": round(float((net[active] > 0).mean() * 100.0), 4) if active.any() else 0.0,
        "TradeCount": int(active.sum()),
        "ExposurePct": round(float(exposure.mean() * 100.0), 4),
        "CostImpactPct": round(float(costs.sum() * 100.0), 4),
    }


def _baseline_signals(price: pd.Series, short_ma: int, long_ma: int, momentum_lookback: int, mean_reversion_window: int, mean_reversion_threshold: float, volatility_target_pct: float) -> Dict[str, pd.Series]:
    short = price.rolling(short_ma, min_periods=short_ma).mean()
    long = price.rolling(long_ma, min_periods=long_ma).mean()
    momentum = price / price.shift(momentum_lookback) - 1.0
    rolling_mean = price.rolling(mean_reversion_window, min_periods=mean_reversion_window).mean()
    distance = price / rolling_mean - 1.0
    realized_vol = price.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252) * 100.0
    scaled = (volatility_target_pct / realized_vol.replace(0, np.nan)).clip(lower=0.0, upper=1.0).fillna(0.0)
    return {
        "NoExposureBaseline": pd.Series(0.0, index=price.index),
        "HoldOnlyBenchmark": pd.Series(1.0, index=price.index),
        "MovingAverageCrossover": (short > long).astype(float).fillna(0.0),
        "MomentumBaseline": (momentum > 0).astype(float).fillna(0.0),
        "MeanReversionBaseline": (distance <= -abs(mean_reversion_threshold)).astype(float).fillna(0.0),
        "VolatilityScaledBaseline": scaled,
    }


def _stable_seed(asset: str, horizon: int, seed: int) -> int:
    return int(seed + int(horizon) * 997 + sum(ord(ch) for ch in str(asset)) * 13)


def _is_proxy_text(value: Any) -> bool:
    text = str(value).lower()
    return any(marker in text for marker in PROXY_SOURCE_MARKERS)


def _random_simulation_metrics(price: pd.Series, asset: str, horizon: int, seed: int, simulations: int, cost_bps: float) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(_stable_seed(asset, horizon, seed))
    daily_return, invalid_rows, data_flag, sanity_status = _price_return_sanity(price, asset)
    if daily_return.empty:
        return []
    rows: List[Dict[str, Any]] = []
    for _ in range(int(simulations)):
        signal = pd.Series(rng.binomial(1, 0.5, len(price)).astype(float), index=price.index)
        exposure = signal.shift(1).reindex(daily_return.index).fillna(0.0)
        metrics = _metrics_from_returns(daily_return * exposure, exposure, cost_bps)
        rows.append(
            {
                **metrics,
                "InvalidReturnRows": int(invalid_rows),
                "DataQualityFlag": data_flag,
                "ReturnSanityStatus": sanity_status,
            }
        )
    return rows


def _representative_random_metrics(random_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not random_rows:
        return {
            **_empty_metrics(),
            "InvalidReturnRows": 0,
            "DataQualityFlag": "InsufficientCleanReturnData",
            "ReturnSanityStatus": "InsufficientCleanReturnData",
        }
    values = np.asarray([_safe_float(row.get("NetReturnPct"), np.nan) for row in random_rows], dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return {
            **_empty_metrics(),
            "InvalidReturnRows": int(max((row.get("InvalidReturnRows", 0) for row in random_rows), default=0)),
            "DataQualityFlag": "InsufficientCleanReturnData",
            "ReturnSanityStatus": "InsufficientCleanReturnData",
        }
    median_value = float(np.median(values[finite]))
    finite_indices = np.where(finite)[0]
    representative_index = int(finite_indices[np.argmin(np.abs(values[finite] - median_value))])
    return dict(random_rows[representative_index])


def _replay_metrics(replay_export: pd.DataFrame, total_cost_bps: float) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], pd.DataFrame]]:
    rows: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, int], pd.DataFrame] = {}
    if replay_export.empty:
        return pd.DataFrame(columns=list(REPLAY_VS_BASELINE_LEADERBOARD_COLUMNS)), groups
    df = replay_export.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Asset", "Horizon"])
    df["Horizon"] = pd.to_numeric(df["Horizon"], errors="coerce")
    for (asset, horizon), group in df.groupby(["Asset", "Horizon"], dropna=False):
        group = group.sort_values("Date").copy()
        group = group[group.get("ComparableHistorical", True).astype(bool)] if "ComparableHistorical" in group.columns else group
        groups[(str(asset), int(horizon))] = group
        if group.empty:
            continue
        strategy_name = str(group["StrategyName"].dropna().iloc[0]) if "StrategyName" in group.columns and not group["StrategyName"].dropna().empty else REPLAY_STRATEGY_PROXY
        replay_source = str(group["ReplaySource"].dropna().iloc[0]) if "ReplaySource" in group.columns and not group["ReplaySource"].dropna().empty else ""
        returns = pd.to_numeric(group.get("StrategyReturnPct", pd.Series(0.0, index=group.index)), errors="coerce").fillna(0.0) / 100.0
        exposure = pd.to_numeric(group.get("ExposurePct", pd.Series(0.0, index=group.index)), errors="coerce").fillna(0.0) / 100.0
        metrics = _metrics_from_returns(pd.Series(returns.values, index=group["Date"]), pd.Series(exposure.values, index=group["Date"]), total_cost_bps)
        evaluation_mode = "HistoricalDailyExposure"
        if "EvaluationMode" in group.columns and not group["EvaluationMode"].astype(str).eq("HistoricalDailyExposure").all():
            evaluation_mode = "InsufficientData"
        rows.append(
            _leaderboard_row(
                strategy=strategy_name,
                asset=str(asset),
                horizon=int(horizon),
                replay_source=replay_source,
                evaluation_mode=evaluation_mode,
                metrics=metrics,
                comparable=bool(not group.empty),
                benchmark_role="ReplayProxy" if _is_proxy_text(replay_source) else "HistoricalModelReplay",
                verdict="ReplayStrategy",
                invalid_rows=0,
                data_quality_flag="NotPriceDerived",
                return_sanity_status="ReplayReturnStream",
            )
        )
    return pd.DataFrame(rows, columns=list(REPLAY_VS_BASELINE_LEADERBOARD_COLUMNS)), groups


def _baseline_leaderboard(market_data: pd.DataFrame, assets: Iterable[str], horizons: Iterable[int], total_cost_bps: float, short_ma: int, long_ma: int, momentum_lookback: int, mean_reversion_window: int, mean_reversion_threshold: float, volatility_target_pct: float, random_seed: int, random_simulations: int) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], List[float]]]:
    rows: List[Dict[str, Any]] = []
    random_map: Dict[Tuple[str, int], List[float]] = {}
    for asset in assets:
        price = _series(market_data, get_target_column(asset))
        for horizon in horizons:
            if price.empty:
                for strategy in SIMPLE_BASELINES:
                    rows.append(
                        _leaderboard_row(
                            strategy=strategy,
                            asset=str(asset),
                            horizon=int(horizon),
                            replay_source="",
                            evaluation_mode="InsufficientData",
                            metrics=_empty_metrics(),
                            comparable=False,
                            benchmark_role="RandomBaseline" if strategy == "RandomBaseline" else "SimpleBaseline",
                            verdict="InsufficientData",
                            invalid_rows=0,
                            data_quality_flag="MissingPriceData",
                            return_sanity_status="InsufficientCleanReturnData",
                        )
                    )
                random_map[(asset, int(horizon))] = []
                continue
            daily_return, invalid_rows, data_flag, sanity_status = _price_return_sanity(price, str(asset))
            signals = _baseline_signals(price, short_ma, long_ma, momentum_lookback, mean_reversion_window, mean_reversion_threshold, volatility_target_pct)
            for strategy, signal in signals.items():
                if daily_return.empty:
                    rows.append(
                        _leaderboard_row(
                            strategy=strategy,
                            asset=str(asset),
                            horizon=int(horizon),
                            replay_source="",
                            evaluation_mode="InsufficientCleanReturnData",
                            metrics=_empty_metrics(),
                            comparable=False,
                            benchmark_role="SimpleBaseline",
                            verdict="InvalidReturnData",
                            invalid_rows=invalid_rows,
                            data_quality_flag=data_flag,
                            return_sanity_status=sanity_status,
                        )
                    )
                    continue
                exposure = signal.reindex(price.index).fillna(0.0).clip(lower=0.0, upper=1.0).shift(1).reindex(daily_return.index).fillna(0.0)
                metrics = _metrics_from_returns(daily_return * exposure, exposure, total_cost_bps)
                rows.append(
                    _leaderboard_row(
                        strategy=strategy,
                        asset=str(asset),
                        horizon=int(horizon),
                        replay_source="",
                        evaluation_mode="HistoricalDailyExposure",
                        metrics=metrics,
                        comparable=True,
                        benchmark_role="SimpleBaseline",
                        verdict="Baseline" if sanity_status != "InsufficientCleanReturnData" else "InvalidReturnData",
                        invalid_rows=invalid_rows,
                        data_quality_flag=data_flag,
                        return_sanity_status=sanity_status,
                    )
                )
            random_metric_rows = _random_simulation_metrics(price, str(asset), int(horizon), random_seed, random_simulations, total_cost_bps)
            random_values = [_safe_float(row.get("NetReturnPct"), np.nan) for row in random_metric_rows]
            random_values = [value for value in random_values if np.isfinite(value)]
            random_map[(asset, int(horizon))] = random_values
            if random_metric_rows:
                representative = _representative_random_metrics(random_metric_rows)
                rows.append(
                    _leaderboard_row(
                        strategy="RandomBaseline",
                        asset=str(asset),
                        horizon=int(horizon),
                        replay_source="",
                        evaluation_mode="HistoricalDailyExposure",
                        metrics={key: representative.get(key, 0.0) for key in _empty_metrics()},
                        comparable=True,
                        benchmark_role="RandomBaseline",
                        verdict="Baseline",
                        invalid_rows=int(representative.get("InvalidReturnRows", invalid_rows)),
                        data_quality_flag=str(representative.get("DataQualityFlag", data_flag)),
                        return_sanity_status=str(representative.get("ReturnSanityStatus", sanity_status)),
                    )
                )
            else:
                rows.append(
                    _leaderboard_row(
                        strategy="RandomBaseline",
                        asset=str(asset),
                        horizon=int(horizon),
                        replay_source="",
                        evaluation_mode="InsufficientCleanReturnData",
                        metrics=_empty_metrics(),
                        comparable=False,
                        benchmark_role="RandomBaseline",
                        verdict="InvalidReturnData",
                        invalid_rows=invalid_rows,
                        data_quality_flag=data_flag or "InsufficientCleanReturnData",
                        return_sanity_status=sanity_status if sanity_status != "CleanReturnData" else "InsufficientCleanReturnData",
                    )
                )
    return pd.DataFrame(rows, columns=list(REPLAY_VS_BASELINE_LEADERBOARD_COLUMNS)), random_map


def _rank_leaderboard(leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty:
        return pd.DataFrame(columns=list(REPLAY_VS_BASELINE_LEADERBOARD_COLUMNS))
    out = leaderboard.copy()
    out = out.sort_values(["NetReturnPct", "SharpeProxy"], ascending=[False, False]).reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1)
    return out[list(REPLAY_VS_BASELINE_LEADERBOARD_COLUMNS)]


def _row_for(group: pd.DataFrame, strategy: str) -> pd.Series:
    row = group[group["StrategyName"].astype(str).eq(strategy)]
    return row.iloc[0] if not row.empty else pd.Series(dtype=object)


def _proxy_row(group: pd.DataFrame) -> pd.Series:
    rows = group[group["BenchmarkRole"].astype(str).isin(["ReplayProxy", "HistoricalModelReplay"])]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def _best_baseline(group: pd.DataFrame) -> pd.Series:
    baselines = group[group["BenchmarkRole"].astype(str).isin(["SimpleBaseline", "RandomBaseline"])]
    baselines = baselines[baselines["ComparableHistorical"].eq(True)]
    if baselines.empty:
        return pd.Series(dtype=object)
    priority = {
        "NoExposureBaseline": 0,
        "HoldOnlyBenchmark": 1,
        "MomentumBaseline": 2,
        "MovingAverageCrossover": 3,
        "VolatilityScaledBaseline": 4,
        "MeanReversionBaseline": 5,
        "RandomBaseline": 6,
    }
    ranked = baselines.copy()
    ranked["_TiePriority"] = ranked["StrategyName"].map(priority).fillna(99)
    return ranked.sort_values(["NetReturnPct", "SharpeProxy", "_TiePriority"], ascending=[False, False, True]).iloc[0]


def _random_comparison(random_map: Dict[Tuple[str, int], List[float]], leaderboard: pd.DataFrame, assets: Iterable[str], horizons: Iterable[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for asset in assets:
        for horizon in horizons:
            values = np.asarray(random_map.get((str(asset), int(horizon)), []), dtype=float)
            group = leaderboard[leaderboard["Asset"].astype(str).eq(str(asset)) & leaderboard["Horizon"].astype(int).eq(int(horizon))]
            proxy = _proxy_row(group)
            proxy_return = _safe_float(proxy.get("NetReturnPct", np.nan), np.nan)
            if len(values) == 0 or not np.isfinite(proxy_return):
                rows.append({"Asset": asset, "Horizon": int(horizon), "RandomSimulationCount": int(len(values)), "RandomMedianReturnPct": 0.0, "RandomP25ReturnPct": 0.0, "RandomP75ReturnPct": 0.0, "RandomBestReturnPct": 0.0, "ProxyReturnPct": proxy_return if np.isfinite(proxy_return) else np.nan, "ProxyBeatsRandomMedian": False, "ProxyPercentileVsRandom": 0.0, "Explanation": "Random baseline or proxy replay return is unavailable."})
                continue
            percentile = float((values <= proxy_return).mean() * 100.0)
            rows.append({"Asset": asset, "Horizon": int(horizon), "RandomSimulationCount": int(len(values)), "RandomMedianReturnPct": round(float(np.median(values)), 4), "RandomP25ReturnPct": round(float(np.percentile(values, 25)), 4), "RandomP75ReturnPct": round(float(np.percentile(values, 75)), 4), "RandomBestReturnPct": round(float(np.max(values)), 4), "ProxyReturnPct": round(proxy_return, 4), "ProxyBeatsRandomMedian": bool(proxy_return >= np.median(values)), "ProxyPercentileVsRandom": round(percentile, 4), "Explanation": "Compares capped replay strategy return with reproducible random exposure simulations."})
    return pd.DataFrame(rows, columns=list(REPLAY_RANDOM_COMPARISON_COLUMNS))


def _edge_tables(leaderboard: pd.DataFrame, random_table: pd.DataFrame, assets: Iterable[str], horizons: Iterable[int], min_trades: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ah_rows: List[Dict[str, Any]] = []
    dominance: List[Dict[str, Any]] = []
    strengths: List[Dict[str, Any]] = []
    drawdown_rows: List[Dict[str, Any]] = []
    for asset in assets:
        for horizon in horizons:
            group = leaderboard[leaderboard["Asset"].astype(str).eq(str(asset)) & leaderboard["Horizon"].astype(int).eq(int(horizon))]
            proxy = _proxy_row(group)
            baseline = _best_baseline(group)
            no_exp = _row_for(group, "NoExposureBaseline")
            random_row = random_table[random_table["Asset"].astype(str).eq(str(asset)) & random_table["Horizon"].astype(int).eq(int(horizon))]
            proxy_return = _safe_float(proxy.get("NetReturnPct", np.nan), np.nan)
            baseline_return = _safe_float(baseline.get("NetReturnPct", np.nan), np.nan)
            proxy_dd = _safe_float(proxy.get("MaxDrawdownPct", 0.0), 0.0)
            baseline_dd = _safe_float(baseline.get("MaxDrawdownPct", 0.0), 0.0)
            trade_count = int(_safe_float(proxy.get("TradeCount", 0), 0))
            random_median = _safe_float(random_row["RandomMedianReturnPct"].iloc[0] if not random_row.empty else np.nan, np.nan)
            beats_random = bool(np.isfinite(proxy_return) and np.isfinite(random_median) and proxy_return >= random_median)
            no_exp_return = _safe_float(no_exp.get("NetReturnPct", 0.0), 0.0)
            if not np.isfinite(proxy_return) or proxy.empty:
                verdict = "InsufficientData"
                reason = "No comparable replay rows were available for this asset and horizon."
            elif trade_count < int(min_trades):
                verdict = "InsufficientTrades"
                reason = "Replay has too few active rows for robust benchmark evidence."
            elif proxy_return >= baseline_return and proxy_return >= no_exp_return and beats_random:
                verdict = "ProxyEdge" if proxy_dd >= baseline_dd - 10.0 else "ProxyMixed"
                reason = "Historical proxy replay beats the strongest simple baseline and random median." if verdict == "ProxyEdge" else "Proxy return is competitive but drawdown is weaker."
            elif proxy_return >= baseline_return:
                verdict = "ProxyCompetitive"
                reason = "Historical proxy replay is competitive with the strongest simple baseline but not all secondary checks pass."
            elif proxy_return >= no_exp_return or beats_random:
                verdict = "ProxyMixed"
                reason = "Historical proxy replay has partial evidence but does not beat the strongest baseline."
            else:
                verdict = "BenchmarkDominated"
                reason = "A simple baseline or no-exposure comparison dominates the proxy replay."
            gap = proxy_return - baseline_return if np.isfinite(proxy_return) and np.isfinite(baseline_return) else np.nan
            ah_rows.append({"Asset": asset, "Horizon": int(horizon), "ProxyReplayReturnPct": round(proxy_return, 4) if np.isfinite(proxy_return) else np.nan, "BestBaseline": baseline.get("StrategyName", ""), "BestBaselineReturnPct": round(baseline_return, 4) if np.isfinite(baseline_return) else np.nan, "ProxyVsBestBaselinePct": round(gap, 4) if np.isfinite(gap) else np.nan, "ProxyBeatsBestBaseline": bool(np.isfinite(gap) and gap >= 0), "ProxyBeatsNoExposure": bool(np.isfinite(proxy_return) and proxy_return >= no_exp_return), "ProxyBeatsRandomMedian": beats_random, "ProxyDrawdownPct": round(proxy_dd, 4), "BaselineDrawdownPct": round(baseline_dd, 4), "TradeCount": trade_count, "EdgeVerdict": verdict, "MainReason": reason})
            drawdown_improvement = baseline_dd - proxy_dd
            tradeoff = "ProxyLowerDrawdown" if drawdown_improvement < 0 else "ProxyHigherDrawdown" if drawdown_improvement > 0 else "SimilarDrawdown"
            drawdown_rows.append({"Asset": asset, "Horizon": int(horizon), "ProxyMaxDrawdownPct": round(proxy_dd, 4), "BestBaselineMaxDrawdownPct": round(baseline_dd, 4), "DrawdownImprovementPct": round(drawdown_improvement, 4), "ProxyReturnPct": round(proxy_return, 4) if np.isfinite(proxy_return) else np.nan, "BestBaselineReturnPct": round(baseline_return, 4) if np.isfinite(baseline_return) else np.nan, "ReturnDrawdownTradeoff": tradeoff, "Explanation": "Compares capped proxy replay drawdown with the strongest simple baseline."})
            if verdict == "BenchmarkDominated" or (np.isfinite(gap) and gap < 0):
                dominance.append({"Asset": asset, "Horizon": int(horizon), "DominatingBaseline": baseline.get("StrategyName", ""), "BaselineReturnPct": round(baseline_return, 4), "ProxyReturnPct": round(proxy_return, 4) if np.isfinite(proxy_return) else np.nan, "GapPct": round(abs(gap), 4) if np.isfinite(gap) else np.nan, "BaselineDrawdownPct": round(baseline_dd, 4), "ProxyDrawdownPct": round(proxy_dd, 4), "DominanceReason": "The strongest simple baseline has higher net return than the capped proxy replay.", "RequiredImprovement": "Improve replay edge, reduce costs, or keep this row research-only."})
            if verdict in {"ProxyEdge", "ProxyCompetitive"} and np.isfinite(gap) and gap >= 0:
                strengths.append({"Asset": asset, "Horizon": int(horizon), "ProxyReturnPct": round(proxy_return, 4), "BestBaselineReturnPct": round(baseline_return, 4), "ImprovementPct": round(gap, 4), "DrawdownImprovementPct": round(drawdown_improvement, 4), "RiskAdjustedImprovement": round(gap + drawdown_improvement * 0.25, 4), "TradeCount": trade_count, "EvidenceStrength": "ProxyOnlyStrong" if verdict == "ProxyEdge" else "ProxyOnlyCompetitive", "MainStrengthReason": "Historical proxy replay is ahead of the strongest simple baseline; still not true trained ML replay evidence."})
    ah = pd.DataFrame(ah_rows, columns=list(REPLAY_ASSET_HORIZON_EDGE_COLUMNS))
    asset_rows: List[Dict[str, Any]] = []
    for asset, group in ah.groupby("Asset", dropna=False):
        proxy_avg = float(pd.to_numeric(group["ProxyReplayReturnPct"], errors="coerce").mean()) if not group.empty else np.nan
        baseline_avg = float(pd.to_numeric(group["BestBaselineReturnPct"], errors="coerce").mean()) if not group.empty else np.nan
        gap = proxy_avg - baseline_avg if np.isfinite(proxy_avg) and np.isfinite(baseline_avg) else np.nan
        edge_rows = group[group["EdgeVerdict"].isin(["ProxyEdge", "ProxyCompetitive"])]
        dominated_rows = group[group["EdgeVerdict"].eq("BenchmarkDominated")]
        verdict = "ProxyEdge" if not edge_rows.empty and dominated_rows.empty else "ProxyMixed" if not edge_rows.empty else "BenchmarkDominated" if not dominated_rows.empty else "InsufficientData"
        ranks = leaderboard[leaderboard["Asset"].astype(str).eq(str(asset))].sort_values(["NetReturnPct", "SharpeProxy"], ascending=[False, False])
        proxy_rank = int(ranks[ranks["BenchmarkRole"].astype(str).isin(["ReplayProxy", "HistoricalModelReplay"])]["Rank"].min()) if not ranks.empty and not ranks[ranks["BenchmarkRole"].astype(str).isin(["ReplayProxy", "HistoricalModelReplay"])].empty else 0
        asset_rows.append({"Asset": asset, "ProxyReplayReturnPct": round(proxy_avg, 4) if np.isfinite(proxy_avg) else np.nan, "BestBaseline": "; ".join(sorted(group["BestBaseline"].dropna().astype(str).unique())), "BestBaselineReturnPct": round(baseline_avg, 4) if np.isfinite(baseline_avg) else np.nan, "ProxyVsBestBaselinePct": round(gap, 4) if np.isfinite(gap) else np.nan, "ProxyRank": proxy_rank, "ProxyBeatsBestBaseline": bool(np.isfinite(gap) and gap >= 0), "ProxyBeatsNoExposure": bool(group["ProxyBeatsNoExposure"].mean() >= 0.5) if not group.empty else False, "ProxyBeatsRandomMedian": bool(group["ProxyBeatsRandomMedian"].mean() >= 0.5) if not group.empty else False, "DrawdownComparison": "Mixed by horizon", "EdgeVerdict": verdict, "MainReason": "Asset has at least one proxy edge row." if not edge_rows.empty else "Asset is dominated or lacks enough replay evidence."})
    return (
        pd.DataFrame(asset_rows, columns=list(REPLAY_ASSET_EDGE_COLUMNS)),
        ah,
        pd.DataFrame(dominance, columns=list(REPLAY_DOMINANCE_FAILURE_COLUMNS)),
        pd.DataFrame(strengths, columns=list(REPLAY_STRENGTH_COLUMNS)),
        pd.DataFrame(drawdown_rows, columns=list(REPLAY_DRAWDOWN_COMPARISON_COLUMNS)),
    )


def _cost_robustness(replay_groups: Dict[Tuple[str, int], pd.DataFrame], cost_scenarios_bps: Iterable[float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (asset, horizon), group in replay_groups.items():
        if group.empty:
            continue
        group = group.sort_values("Date")
        gross = pd.to_numeric(group.get("StrategyReturnPct", pd.Series(0.0, index=group.index)), errors="coerce").fillna(0.0) / 100.0
        exposure = pd.to_numeric(group.get("ExposurePct", pd.Series(0.0, index=group.index)), errors="coerce").fillna(0.0) / 100.0
        index = pd.to_datetime(group["Date"], errors="coerce")
        zero = _metrics_from_returns(pd.Series(gross.values, index=index), pd.Series(exposure.values, index=index), 0.0)["NetReturnPct"]
        breakeven = np.nan
        for probe in range(0, 501, 5):
            net = _metrics_from_returns(pd.Series(gross.values, index=index), pd.Series(exposure.values, index=index), float(probe))["NetReturnPct"]
            if net <= 0:
                breakeven = float(probe)
                break
        for cost in cost_scenarios_bps:
            net = _metrics_from_returns(pd.Series(gross.values, index=index), pd.Series(exposure.values, index=index), float(cost))["NetReturnPct"]
            lost = zero - net
            fragile = bool((zero >= 0 and net < 0) or lost > max(2.0, abs(zero) * 0.5))
            rows.append({"Asset": asset, "Horizon": int(horizon), "CostBps": float(cost), "ProxyNetReturnPct": round(float(net), 4), "ReturnLostToCostsPct": round(float(lost), 4), "CostFragile": fragile, "BreakEvenCostBps": round(breakeven, 4) if np.isfinite(breakeven) else np.nan, "Explanation": "Costs destroy or materially reduce proxy replay return." if fragile else "Cost impact is visible and retained for review."})
    return pd.DataFrame(rows, columns=list(REPLAY_COST_ROBUSTNESS_COLUMNS))


def _quality_gates(tables: Dict[str, pd.DataFrame], replay_export: pd.DataFrame, leaderboard: pd.DataFrame, random_table: pd.DataFrame, cost_table: pd.DataFrame, min_matured_rows: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    phase17_quality = tables.get("replay_quality_checks", pd.DataFrame())
    cap = tables.get("replay_exposure_cap_table", pd.DataFrame())
    summary = tables.get("replay_summary_table", pd.DataFrame())
    quality_pass = bool(not phase17_quality.empty and phase17_quality["Passed"].astype(bool).all())
    cap_pass = True
    if not cap.empty and {"ExposureAfterCapPct", "MaxPortfolioPaperExposurePct"}.issubset(cap.columns):
        cap_pass = bool((pd.to_numeric(cap["ExposureAfterCapPct"], errors="coerce").fillna(0.0) <= pd.to_numeric(cap["MaxPortfolioPaperExposurePct"], errors="coerce").fillna(0.0) + 1e-6).all())
    replay_sources = set(replay_export.get("ReplaySource", pd.Series(dtype=str)).dropna().astype(str))
    strategy_names = set(replay_export.get("StrategyName", pd.Series(dtype=str)).dropna().astype(str))
    source_labeled = bool(replay_sources and not {""}.intersection(replay_sources))
    summary_quality = str(summary.iloc[0].get("ModelReplayQuality", "")) if not summary.empty else ""
    summary_source = str(summary.iloc[0].get("ReplaySource", "")) if not summary.empty else ""
    proxy_summary = any(marker in f"{summary_quality} {summary_source}".lower() for marker in PROXY_SOURCE_MARKERS)
    proxy_rows = pd.DataFrame()
    if not replay_export.empty:
        source_text = replay_export.get("ReplaySource", pd.Series("", index=replay_export.index)).astype(str).str.lower()
        quality_text = replay_export.get("ModelReplayQuality", pd.Series("", index=replay_export.index)).astype(str).str.lower()
        strategy_text = replay_export.get("StrategyName", pd.Series("", index=replay_export.index)).astype(str).str.lower()
        proxy_mask = source_text.apply(lambda value: any(marker in value for marker in PROXY_SOURCE_MARKERS)) | quality_text.apply(lambda value: any(marker in value for marker in PROXY_SOURCE_MARKERS))
        proxy_rows = replay_export[proxy_mask].copy()
        model_like_proxy_mask = proxy_mask & strategy_text.apply(lambda value: any(marker in value for marker in MODEL_LIKE_STRATEGY_MARKERS))
        model_like_proxy_count = int(model_like_proxy_mask.sum())
        clearly_proxy_named = bool((~proxy_mask | strategy_text.eq("historicalsignalproxyreplay")).all())
    else:
        model_like_proxy_count = 0
        clearly_proxy_named = not proxy_summary
    is_proxy = bool(proxy_summary or not proxy_rows.empty)
    proxy_not_ml = bool(not is_proxy or (model_like_proxy_count == 0 and clearly_proxy_named))
    proxy_gate_explanation = (
        "Proxy replay is clearly labeled as proxy evidence."
        if proxy_not_ml
        else "Proxy replay is being mislabeled as historical model evidence by a model-like StrategyName."
    )
    comparable = replay_export[replay_export.get("ComparableHistorical", pd.Series(False, index=replay_export.index)).astype(bool)] if not replay_export.empty else pd.DataFrame()
    comparable_valid = bool(not comparable.empty and comparable.get("EvaluationMode", pd.Series("", index=comparable.index)).astype(str).eq("HistoricalDailyExposure").all())
    q_lookup = phase17_quality.set_index("CheckName")["Passed"].to_dict() if not phase17_quality.empty and "CheckName" in phase17_quality.columns else {}
    no_future = bool(q_lookup.get("NoFutureDataUsed", True) and q_lookup.get("OutcomesAfterReplayDate", True))
    no_same_day = bool(q_lookup.get("NoSameDayCloseLeakage", True))
    leaderboard_returns = pd.to_numeric(leaderboard.get("NetReturnPct", pd.Series(dtype=float)), errors="coerce") if not leaderboard.empty else pd.Series(dtype=float)
    return_sanity = bool(not leaderboard_returns.empty and leaderboard_returns.abs().max() <= 100000)
    baseline_rows = leaderboard[leaderboard["BenchmarkRole"].astype(str).isin(["SimpleBaseline", "RandomBaseline"])] if not leaderboard.empty else pd.DataFrame()
    comparable_baselines = baseline_rows[baseline_rows.get("ComparableHistorical", pd.Series(False, index=baseline_rows.index)).astype(bool)] if not baseline_rows.empty else pd.DataFrame()
    if comparable_baselines.empty:
        baseline_return_sanity = False
        if baseline_rows.empty:
            baseline_sanity_explanation = "No comparable baseline rows were available for return sanity checks."
        else:
            status = baseline_rows.get("ReturnSanityStatus", pd.Series("", index=baseline_rows.index)).astype(str)
            flags = baseline_rows.get("DataQualityFlag", pd.Series("", index=baseline_rows.index)).astype(str)
            invalid_rows = pd.to_numeric(baseline_rows.get("InvalidReturnRows", pd.Series(0, index=baseline_rows.index)), errors="coerce").fillna(0)
            insufficient_clean = bool(
                status.eq("InsufficientCleanReturnData").any()
                or flags.str.contains("InsufficientCleanReturnData", case=False, regex=False).any()
                or ((invalid_rows > 0) & status.ne("CleanReturnData")).any()
            )
            baseline_sanity_explanation = (
                "InsufficientCleanReturnData: no valid positive price transitions were available for comparable baseline return sanity checks."
                if insufficient_clean
                else "No comparable baseline rows were available for return sanity checks."
            )
        nonpositive_handling = bool({"InvalidReturnRows", "DataQualityFlag", "ReturnSanityStatus"}.issubset(leaderboard.columns))
    else:
        net = pd.to_numeric(comparable_baselines["NetReturnPct"], errors="coerce")
        status = comparable_baselines.get("ReturnSanityStatus", pd.Series("", index=comparable_baselines.index)).astype(str)
        flags = comparable_baselines.get("DataQualityFlag", pd.Series("", index=comparable_baselines.index)).astype(str)
        invalid_rows = pd.to_numeric(comparable_baselines.get("InvalidReturnRows", pd.Series(0, index=comparable_baselines.index)), errors="coerce").fillna(0)
        explained_invalid = (
            status.ne("CleanReturnData")
            | flags.str.contains("InvalidReturn|NonPositivePriceReturnInvalid|InsufficientCleanReturnData|ReturnDataRequiresCare", case=False, regex=True)
        )
        unexplained_below_minus_100 = (net < -100.0 - RETURN_SANITY_EPSILON) & ~explained_invalid
        insufficient_clean = status.eq("InsufficientCleanReturnData")
        baseline_return_sanity = bool(not unexplained_below_minus_100.any() and not insufficient_clean.any())
        skipped_invalid = int((invalid_rows > 0).sum())
        if unexplained_below_minus_100.any():
            baseline_sanity_explanation = "BaselineReturnBelowMinus100 found without an InvalidReturnData explanation."
        elif insufficient_clean.any():
            baseline_sanity_explanation = "One or more baselines have InsufficientCleanReturnData after invalid transitions were detected."
        elif skipped_invalid:
            baseline_sanity_explanation = f"InvalidReturnRowsSkipped for {skipped_invalid} comparable baseline rows; invalid transitions were excluded from compounding."
        else:
            baseline_sanity_explanation = "Baseline return streams passed strict price-transition sanity checks."
        nonpositive_handling = bool(
            {"InvalidReturnRows", "DataQualityFlag", "ReturnSanityStatus"}.issubset(leaderboard.columns)
            and not ((invalid_rows > 0) & flags.eq("")).any()
        )
    random_rows = baseline_rows[baseline_rows["StrategyName"].astype(str).eq("RandomBaseline")] if not baseline_rows.empty else pd.DataFrame()
    if random_rows.empty:
        random_metrics_valid = False
        random_metrics_explanation = "RandomBaselineMetricsIncomplete: no random baseline leaderboard rows were produced."
    else:
        random_comparable = random_rows[random_rows.get("ComparableHistorical", pd.Series(False, index=random_rows.index)).astype(bool)]
        numeric_cols = ["NetReturnPct", "MaxDrawdownPct", "WinRatePct", "TradeCount", "VolatilityPct", "SharpeProxy", "ExposurePct"]
        finite_metrics = bool(not random_comparable.empty)
        for col in numeric_cols:
            if col not in random_comparable.columns:
                finite_metrics = False
                break
            finite_metrics = finite_metrics and bool(np.isfinite(pd.to_numeric(random_comparable[col], errors="coerce")).all())
        random_net = pd.to_numeric(random_comparable.get("NetReturnPct", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        random_dd = pd.to_numeric(random_comparable.get("MaxDrawdownPct", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        random_win = pd.to_numeric(random_comparable.get("WinRatePct", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        placeholder_like = bool(((random_net < -RETURN_SANITY_EPSILON) & (random_dd == 0.0) & (random_win == 0.0)).any())
        random_metrics_valid = bool(finite_metrics and not placeholder_like)
        random_metrics_explanation = (
            "RandomBaseline metrics are computed from a representative seeded simulation."
            if random_metrics_valid
            else "RandomBaselineMetricsIncomplete: leaderboard rows have missing or placeholder equity-curve metrics."
        )
    gates = [
        ("Phase17QualityChecksPassed", quality_pass, "Critical", "Phase 17 replay quality checks must pass."),
        ("Phase17ExposureCapRespected", cap_pass, "Critical", "Phase 17 capped portfolio exposure must remain within the configured cap."),
        ("ReplaySourceClearlyLabeled", source_labeled, "High", "ReplaySource must clearly distinguish proxy replay from true historical model prediction replay."),
        ("ProxyNotMisrepresentedAsML", proxy_not_ml, "Critical", proxy_gate_explanation),
        ("ComparableHistoricalRowsValid", comparable_valid, "Critical", "Comparable rows must be matured historical daily-exposure rows."),
        ("NoFutureLeakage", no_future, "Critical", "Phase 17 must not use future data for replay signals."),
        ("NoSameDayLeakage", no_same_day, "Critical", "Replay outcomes must not rely on same-day close leakage."),
        ("SufficientMaturedRows", int(len(comparable)) >= int(min_matured_rows), "Medium", "Enough comparable replay rows are needed for an edge audit."),
        ("ReturnSanityChecksPassed", return_sanity, "Critical", "Replay and baseline returns must remain within sane bounds."),
        ("BaselineReturnSanityPassed", baseline_return_sanity, "Critical", baseline_sanity_explanation),
        ("RandomBaselineMetricsValid", random_metrics_valid, "High", random_metrics_explanation),
        ("NonPositivePriceHandlingApplied", nonpositive_handling, "High", "Non-positive/missing price transitions are counted and excluded from price-derived baseline compounding."),
        ("RandomBaselineAvailable", bool(not random_table.empty and random_table["RandomSimulationCount"].max() > 0), "Medium", "Random baseline comparison must be available."),
        ("CostSensitivityAvailable", bool(not cost_table.empty), "Medium", "Cost sensitivity table must be available."),
    ]
    for name, passed, severity, explanation in gates:
        rows.append({"GateName": name, "Passed": bool(passed), "Severity": severity, "Explanation": explanation})
    return pd.DataFrame(rows, columns=list(REPLAY_QUALITY_GATE_COLUMNS))


def _summary(tables: Dict[str, pd.DataFrame], quality: pd.DataFrame, asset_edge: pd.DataFrame, ah_edge: pd.DataFrame, random_table: pd.DataFrame) -> pd.DataFrame:
    replay_summary = tables.get("replay_summary_table", pd.DataFrame())
    replay_export_quality = "ProxyOnly"
    replay_source = "HistoricalSignalProxyReplay"
    if not replay_summary.empty:
        replay_source = str(replay_summary.iloc[0].get("ReplaySource", replay_source))
        replay_export_quality = str(replay_summary.iloc[0].get("ModelReplayQuality", replay_export_quality))
    critical_fail = bool(not quality.empty and (~quality["Passed"].astype(bool) & quality["Severity"].astype(str).eq("Critical")).any())
    if critical_fail:
        verdict = "ReplayQualityFailed"
        reason = "One or more critical replay quality gates failed."
    elif ah_edge.empty or ah_edge["EdgeVerdict"].isin(["InsufficientData", "InsufficientTrades"]).mean() > 0.8:
        verdict = "InsufficientReplayEvidence"
        reason = "Replay evidence is too sparse for a benchmark edge claim."
    else:
        edge_count = int(ah_edge["EdgeVerdict"].isin(["ProxyEdge", "ProxyCompetitive"]).sum())
        dominated_count = int(ah_edge["EdgeVerdict"].eq("BenchmarkDominated").sum())
        if edge_count > 0 and dominated_count == 0:
            verdict = "ProxyEdgePresent"
            reason = "Historical proxy replay has rows that beat simple baselines; this is still proxy-only evidence."
        elif edge_count > 0:
            verdict = "ProxyMixed"
            reason = "Historical proxy replay works in some rows but is benchmark-dominated elsewhere."
        elif dominated_count > 0:
            verdict = "ProxyBenchmarkDominated"
            reason = "Simple baselines dominate the historical proxy replay."
        else:
            verdict = "ProxyMixed"
            reason = "Replay evidence is mixed and does not support a broad edge claim."
    def beats(strategy: str) -> bool:
        # Asset/horizon table stores the strongest baseline, so global checks use rows where that named baseline is present.
        return bool(False) if ah_edge.empty else bool(ah_edge[ah_edge["BestBaseline"].astype(str).eq(strategy)]["ProxyBeatsBestBaseline"].mean() >= 0.5)
    random_beats = bool(random_table["ProxyBeatsRandomMedian"].mean() >= 0.5) if not random_table.empty else False
    edge_rows = ah_edge[ah_edge["EdgeVerdict"].isin(["ProxyEdge", "ProxyCompetitive"])] if not ah_edge.empty else pd.DataFrame()
    dominated = ah_edge[ah_edge["EdgeVerdict"].eq("BenchmarkDominated")] if not ah_edge.empty else pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "ReplaySource": replay_source,
                "ModelReplayQuality": replay_export_quality,
                "ReplayBenchmarkVerdict": verdict,
                "ProxyBeatsHoldOnly": beats("HoldOnlyBenchmark"),
                "ProxyBeatsMomentum": beats("MomentumBaseline"),
                "ProxyBeatsMovingAverage": beats("MovingAverageCrossover"),
                "ProxyBeatsNoExposure": bool(ah_edge["ProxyBeatsNoExposure"].mean() >= 0.5) if not ah_edge.empty else False,
                "ProxyBeatsRandomMedian": random_beats,
                "AssetsWithProxyEdge": "; ".join(sorted(edge_rows["Asset"].dropna().astype(str).unique())) if not edge_rows.empty else "",
                "AssetsBenchmarkDominated": "; ".join(sorted(dominated["Asset"].dropna().astype(str).unique())) if not dominated.empty else "",
                "HorizonsWithProxyEdge": "; ".join(f"{int(h)}D" for h in sorted(pd.to_numeric(edge_rows["Horizon"], errors="coerce").dropna().unique())) if not edge_rows.empty else "",
                "HorizonsBenchmarkDominated": "; ".join(f"{int(h)}D" for h in sorted(pd.to_numeric(dominated["Horizon"], errors="coerce").dropna().unique())) if not dominated.empty else "",
                "MainReason": reason,
                "MainLimitation": "Proxy-only evidence; not true historical trained-model prediction replay." if replay_export_quality == "ProxyOnly" else "Historical prediction replay still requires forward confirmation.",
                "RecommendedNextStep": "Persist timestamped model predictions and rerun replay-integrated benchmark audit.",
            }
        ],
        columns=list(REPLAY_BENCHMARK_SUMMARY_COLUMNS),
    )


def _real_capital_readiness(ah_edge: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    gates_pass = bool(not quality.empty and quality["Passed"].astype(bool).all())
    for _, row in ah_edge.iterrows():
        verdict = str(row["EdgeVerdict"])
        if not gates_pass:
            readiness = "Blocked"
            blocking = "Quality gates failed; real capital remains blocked."
        elif verdict in {"ProxyEdge", "ProxyCompetitive"}:
            readiness = "ResearchOnly"
            blocking = "Proxy-only benchmark evidence is not enough for real-capital eligibility."
        elif verdict == "ProxyMixed":
            readiness = "PaperOnly"
            blocking = "Mixed proxy benchmark evidence; restrict to paper research."
        elif verdict in {"InsufficientData", "InsufficientTrades"}:
            readiness = "InsufficientEvidence"
            blocking = "Replay evidence is insufficient."
        else:
            readiness = "Blocked"
            blocking = "Benchmark dominated."
        rows.append({"Asset": row["Asset"], "Horizon": int(row["Horizon"]), "RealCapitalReadiness": readiness, "BlockingReasons": blocking, "RequiredEvidenceBeforeEligibility": "True timestamped model replay, calibrated probabilities, mature forward evidence, and benchmark robustness.", "ProxyEvidenceStatus": verdict, "ForwardEvidenceStatus": "Not evaluated in Phase 18", "CalibrationStatus": "Not evaluated in Phase 18", "BenchmarkStatus": verdict, "RecommendedMode": "Paper research only" if readiness in {"ResearchOnly", "PaperOnly"} else "Research review only"})
    return pd.DataFrame(rows, columns=list(REPLAY_REAL_CAPITAL_READINESS_COLUMNS))


def _next_actions(summary: pd.DataFrame, dominance: pd.DataFrame, strengths: pd.DataFrame, ah_edge: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    verdict = str(summary.iloc[0]["ReplayBenchmarkVerdict"]) if not summary.empty else "InsufficientReplayEvidence"
    rows.append({"Rank": 0, "Action": "Persist timestamped model predictions for true historical replay.", "WhyItMatters": "Proxy replay cannot prove trained model history.", "AffectedAssets": "ALL", "AffectedHorizons": "ALL", "ExpectedBenefit": "Separates proxy evidence from true historical prediction evidence.", "Urgency": "High", "DependsOn": "Prediction logging pipeline."})
    if not dominance.empty:
        rows.append({"Rank": 0, "Action": "Investigate benchmark-dominated replay rows.", "WhyItMatters": "Simple baselines beating proxy replay is a central research finding.", "AffectedAssets": "; ".join(sorted(dominance["Asset"].astype(str).unique())), "AffectedHorizons": "; ".join(f"{int(h)}D" for h in sorted(pd.to_numeric(dominance["Horizon"], errors="coerce").dropna().unique())), "ExpectedBenefit": "Avoids overreading weak proxy behavior.", "Urgency": "High", "DependsOn": "Dominance failure table."})
    if not strengths.empty:
        rows.append({"Rank": 0, "Action": "Focus research on competitive proxy rows.", "WhyItMatters": "Rows that beat baselines are the only candidates worth deeper replay logging.", "AffectedAssets": "; ".join(sorted(strengths["Asset"].astype(str).unique())), "AffectedHorizons": "; ".join(f"{int(h)}D" for h in sorted(pd.to_numeric(strengths["Horizon"], errors="coerce").dropna().unique())), "ExpectedBenefit": "Concentrates future validation on the strongest honest evidence.", "Urgency": "Medium", "DependsOn": "Strength table and forward evidence."})
    if verdict in {"ProxyBenchmarkDominated", "InsufficientReplayEvidence"}:
        rows.append({"Rank": 0, "Action": "Do not upgrade research posture from Phase 18 alone.", "WhyItMatters": "Benchmark audit did not produce enough robust proxy edge.", "AffectedAssets": "ALL", "AffectedHorizons": "ALL", "ExpectedBenefit": "Keeps weak evidence visible without overstating it.", "Urgency": "High", "DependsOn": "More replay and forward evidence."})
    actions = pd.DataFrame(rows, columns=list(REPLAY_NEXT_ACTION_COLUMNS))
    actions["Rank"] = np.arange(1, len(actions) + 1)
    return actions


def run_replay_benchmark_audit(
    *,
    market_data: Optional[pd.DataFrame] = None,
    use_project_market_data: bool = True,
    use_artifact_store: bool = False,
    prefer_uploaded: bool = False,
    uploaded_overrides: Optional[Dict[str, Any]] = None,
    assets: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    short_ma: int = 20,
    long_ma: int = 50,
    momentum_lookback: int = 20,
    mean_reversion_window: int = 20,
    mean_reversion_threshold: float = 0.03,
    volatility_target_pct: float = 12.0,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    cost_scenarios_bps: Iterable[float] = DEFAULT_COST_SCENARIOS_BPS,
    random_seed: int = 42,
    random_simulations: int = 100,
    min_trades: int = 3,
    min_matured_rows: int = 10,
    autosave: bool = False,
    **direct_tables: Any,
) -> ReplayBenchmarkAuditReport:
    asset_list = list(assets or get_asset_names())
    horizon_list = [int(h) for h in (horizons or REPLAY_AUDIT_HORIZONS)]
    project_used = False
    if market_data is None and use_project_market_data:
        market_data = _load_project_market_data()
        project_used = market_data is not None
    market = _prepare_market_data(market_data)
    tables, artifact_sources = _resolve_inputs(bool(use_artifact_store), bool(prefer_uploaded), uploaded_overrides, direct_tables)
    replay_export = tables.get("phase16_replay_export_table", pd.DataFrame()).copy()
    if not replay_export.empty:
        replay_export = _normalise_horizon(replay_export)
    total_cost_bps = float(cost_bps) + float(slippage_bps)
    replay_rows, replay_groups = _replay_metrics(replay_export, total_cost_bps)
    baseline_rows, random_map = _baseline_leaderboard(market, asset_list, horizon_list, total_cost_bps, int(short_ma), int(long_ma), int(momentum_lookback), int(mean_reversion_window), float(mean_reversion_threshold), float(volatility_target_pct), int(random_seed), int(random_simulations))
    leaderboard = _rank_leaderboard(pd.concat([replay_rows, baseline_rows], ignore_index=True))
    random_table = _random_comparison(random_map, leaderboard, asset_list, horizon_list)
    asset_edge, ah_edge, dominance, strengths, drawdown = _edge_tables(leaderboard, random_table, asset_list, horizon_list, int(min_trades))
    cost_table = _cost_robustness(replay_groups, cost_scenarios_bps)
    quality = _quality_gates(tables, replay_export, leaderboard, random_table, cost_table, int(min_matured_rows))
    if not quality.empty and not quality["Passed"].astype(bool).all():
        # Keep rows visible, but mark replay strategy rows as not valid benchmark winners if critical gates fail.
        critical_failed = bool((~quality["Passed"].astype(bool) & quality["Severity"].astype(str).eq("Critical")).any())
        if critical_failed and not leaderboard.empty:
            replay_mask = leaderboard["BenchmarkRole"].astype(str).isin(["ReplayProxy", "HistoricalModelReplay"])
            leaderboard.loc[replay_mask, "Verdict"] = "ReplayQualityFailed"
    summary = _summary(tables, quality, asset_edge, ah_edge, random_table)
    readiness = _real_capital_readiness(ah_edge, quality)
    actions = _next_actions(summary, dominance, strengths, ah_edge)
    input_sources = _input_sources(market, replay_export, tables, asset_list, project_used)
    settings = {
        "phase": "18",
        "purpose": "replay_integrated_benchmark_edge_audit",
        "assets": asset_list,
        "horizons": horizon_list,
        "cost_bps": float(cost_bps),
        "slippage_bps": float(slippage_bps),
        "random_seed": int(random_seed),
        "random_simulations": int(random_simulations),
        "min_trades": int(min_trades),
        "min_matured_rows": int(min_matured_rows),
    }
    report = ReplayBenchmarkAuditReport(
        replay_benchmark_summary_table=summary.reset_index(drop=True),
        replay_vs_baseline_leaderboard=leaderboard.reset_index(drop=True),
        replay_asset_edge_table=asset_edge.reset_index(drop=True),
        replay_asset_horizon_edge_table=ah_edge.reset_index(drop=True),
        replay_dominance_failures_table=dominance.reset_index(drop=True),
        replay_strength_table=strengths.reset_index(drop=True),
        replay_random_comparison_table=random_table.reset_index(drop=True),
        replay_cost_robustness_table=cost_table.reset_index(drop=True),
        replay_drawdown_comparison_table=drawdown.reset_index(drop=True),
        replay_quality_gate_table=quality.reset_index(drop=True),
        replay_real_capital_readiness_table=readiness.reset_index(drop=True),
        replay_next_actions_table=actions.reset_index(drop=True),
        replay_benchmark_input_sources_table=input_sources.reset_index(drop=True),
        artifact_input_source_table=artifact_sources.reset_index(drop=True),
        settings=settings,
    )
    if autosave:
        report.saved_artifacts = save_phase_artifacts(
            REPLAY_BENCHMARK_AUDIT_PHASE_NAME,
            {
                "replay_benchmark_summary_table": report.replay_benchmark_summary_table,
                "replay_vs_baseline_leaderboard": report.replay_vs_baseline_leaderboard,
                "replay_asset_edge_table": report.replay_asset_edge_table,
                "replay_asset_horizon_edge_table": report.replay_asset_horizon_edge_table,
                "replay_dominance_failures_table": report.replay_dominance_failures_table,
                "replay_strength_table": report.replay_strength_table,
                "replay_random_comparison_table": report.replay_random_comparison_table,
                "replay_cost_robustness_table": report.replay_cost_robustness_table,
                "replay_drawdown_comparison_table": report.replay_drawdown_comparison_table,
                "replay_quality_gate_table": report.replay_quality_gate_table,
                "replay_real_capital_readiness_table": report.replay_real_capital_readiness_table,
                "replay_next_actions_table": report.replay_next_actions_table,
                "replay_benchmark_input_sources_table": report.replay_benchmark_input_sources_table,
                "artifact_input_source_table": report.artifact_input_source_table,
            },
            inputs={},
            config=report.settings,
            warnings=[],
        )
    return report


__all__ = [
    "REPLAY_ASSET_EDGE_COLUMNS",
    "REPLAY_ASSET_HORIZON_EDGE_COLUMNS",
    "REPLAY_BENCHMARK_AUDIT_PHASE_NAME",
    "REPLAY_BENCHMARK_SUMMARY_COLUMNS",
    "REPLAY_COST_ROBUSTNESS_COLUMNS",
    "REPLAY_DOMINANCE_FAILURE_COLUMNS",
    "REPLAY_DRAWDOWN_COMPARISON_COLUMNS",
    "REPLAY_INPUT_SOURCE_COLUMNS",
    "REPLAY_NEXT_ACTION_COLUMNS",
    "REPLAY_QUALITY_GATE_COLUMNS",
    "REPLAY_RANDOM_COMPARISON_COLUMNS",
    "REPLAY_REAL_CAPITAL_READINESS_COLUMNS",
    "REPLAY_STRENGTH_COLUMNS",
    "REPLAY_VS_BASELINE_LEADERBOARD_COLUMNS",
    "ReplayBenchmarkAuditReport",
    "run_replay_benchmark_audit",
]
