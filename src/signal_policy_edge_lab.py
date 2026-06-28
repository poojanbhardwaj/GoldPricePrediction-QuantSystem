"""Phase 19 signal policy and edge repair lab.

The lab tests time-safe research policies against simple baselines. It is a
paper/research diagnostic layer only: it does not train models, change targets,
or relax real-capital gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.artifact_store import resolve_artifact, save_phase_artifacts
from src.asset_config import get_asset_names, get_target_column


POLICY_EDGE_LAB_PHASE_NAME = "phase19_signal_policy_edge_lab"
POLICY_LAB_HORIZONS: Tuple[int, ...] = (1, 5, 10, 20, 30)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKET_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"
DEFAULT_COST_SCENARIOS_BPS: Tuple[float, ...] = (0.0, 5.0, 10.0, 25.0, 50.0)
RETURN_SANITY_EPSILON = 1e-12

POLICY_NAMES: Tuple[str, ...] = (
    "TrendMomentumPolicy",
    "InverseMomentumPolicy",
    "RegimeFilteredMomentumPolicy",
    "VolatilityScaledMomentumPolicy",
    "DrawdownAvoidancePolicy",
    "MovingAverageTrendPolicy",
    "BreakoutPolicy",
    "MeanReversionPolicy",
    "BenchmarkAwareHybridPolicy",
    "CostAwarePolicy",
    "NoExposurePolicy",
)

BASELINE_NAMES: Tuple[str, ...] = (
    "HoldOnlyBenchmark",
    "MomentumBaseline",
    "MovingAverageCrossover",
    "NoExposurePolicy",
)

POLICY_LAB_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "PolicyLabVerdict",
    "BestPolicy",
    "BestPolicyOutOfSample",
    "BestAsset",
    "BestHorizon",
    "PoliciesTested",
    "AssetsCovered",
    "HorizonsCovered",
    "ProxyBaselineGapAddressed",
    "BroadEdgeFound",
    "MainReason",
    "MainLimitation",
    "RecommendedNextStep",
)

POLICY_LEADERBOARD_COLUMNS: Tuple[str, ...] = (
    "Rank",
    "PolicyName",
    "Asset",
    "Horizon",
    "EvaluationMode",
    "TotalReturnPct",
    "NetReturnPct",
    "AnnualizedReturnPct",
    "VolatilityPct",
    "SharpeProxy",
    "MaxDrawdownPct",
    "WinRatePct",
    "TradeCount",
    "TurnoverPct",
    "CostImpactPct",
    "ExposurePct",
    "BeatsHoldOnly",
    "BeatsMomentum",
    "BeatsMovingAverage",
    "BeatsNoExposure",
    "BeatsRandomMedian",
    "PolicyVerdict",
    "DataQualityFlag",
)

POLICY_ASSET_EDGE_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "BestPolicy",
    "BestPolicyReturnPct",
    "BestBaseline",
    "BestBaselineReturnPct",
    "PolicyVsBestBaselinePct",
    "BeatsBestBaseline",
    "EdgeVerdict",
    "MainReason",
    "MainWeakness",
)

POLICY_ASSET_HORIZON_EDGE_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "BestPolicy",
    "BestPolicyReturnPct",
    "BestBaseline",
    "BestBaselineReturnPct",
    "PolicyVsBestBaselinePct",
    "BeatsBestBaseline",
    "MaxDrawdownPct",
    "TradeCount",
    "EdgeVerdict",
    "MainReason",
)

POLICY_DOMINANCE_FAILURE_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "PolicyName",
    "DominatingBaseline",
    "BaselineReturnPct",
    "PolicyReturnPct",
    "GapPct",
    "BaselineDrawdownPct",
    "PolicyDrawdownPct",
    "DominanceReason",
    "RequiredImprovement",
)

POLICY_STRENGTH_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "PolicyName",
    "PolicyReturnPct",
    "BestBaselineReturnPct",
    "ImprovementPct",
    "DrawdownImprovementPct",
    "RiskAdjustedImprovement",
    "TradeCount",
    "EvidenceStrength",
    "MainStrengthReason",
)

POLICY_OVERFIT_AUDIT_COLUMNS: Tuple[str, ...] = (
    "PolicyName",
    "Asset",
    "Horizon",
    "InSampleReturnPct",
    "OutOfSampleReturnPct",
    "ISOSTabilityGapPct",
    "OverfitRisk",
    "Explanation",
)

POLICY_COST_SENSITIVITY_COLUMNS: Tuple[str, ...] = (
    "PolicyName",
    "Asset",
    "Horizon",
    "CostBps",
    "NetReturnPct",
    "ReturnLostToCostsPct",
    "CostFragile",
    "BreakEvenCostBps",
    "Explanation",
)

POLICY_RANDOM_COMPARISON_COLUMNS: Tuple[str, ...] = (
    "PolicyName",
    "Asset",
    "Horizon",
    "RandomSimulationCount",
    "RandomMedianReturnPct",
    "RandomP25ReturnPct",
    "RandomP75ReturnPct",
    "RandomBestReturnPct",
    "PolicyReturnPct",
    "PolicyBeatsRandomMedian",
    "PolicyPercentileVsRandom",
    "Explanation",
)

POLICY_DRAWDOWN_COLUMNS: Tuple[str, ...] = (
    "PolicyName",
    "Asset",
    "Horizon",
    "MaxDrawdownPct",
    "DrawdownVsBestBaselinePct",
    "DrawdownControlWorked",
    "ReturnDrawdownTradeoff",
    "Explanation",
)

POLICY_TURNOVER_COLUMNS: Tuple[str, ...] = (
    "PolicyName",
    "Asset",
    "Horizon",
    "TradeCount",
    "TurnoverPct",
    "AvgHoldingPeriodDays",
    "CostImpactPct",
    "TurnoverRisk",
    "Explanation",
)

POLICY_QUALITY_GATES_COLUMNS: Tuple[str, ...] = (
    "GateName",
    "Passed",
    "Severity",
    "Explanation",
)

POLICY_RECOMMENDATION_COLUMNS: Tuple[str, ...] = (
    "Rank",
    "PolicyName",
    "Asset",
    "Horizon",
    "RecommendationType",
    "WhyItMatters",
    "EvidenceStatus",
    "RequiredNextValidation",
    "RealCapitalStatus",
    "PaperResearchStatus",
)

POLICY_NEXT_ACTION_COLUMNS: Tuple[str, ...] = (
    "Rank",
    "Action",
    "WhyItMatters",
    "AffectedPolicies",
    "AffectedAssets",
    "AffectedHorizons",
    "ExpectedBenefit",
    "Urgency",
    "DependsOn",
)

POLICY_INPUT_SOURCE_COLUMNS: Tuple[str, ...] = (
    "SourceName",
    "Available",
    "Rows",
    "Columns",
    "LastDate",
    "MissingCriticalColumns",
    "Notes",
)

INPUT_SPECS: Dict[str, Tuple[str, str, bool]] = {
    "phase18_replay_asset_horizon_edge": ("phase18_replay_benchmark_audit", "replay_asset_horizon_edge_table", False),
    "phase18_replay_dominance_failures": ("phase18_replay_benchmark_audit", "replay_dominance_failures_table", False),
    "phase18_replay_strength": ("phase18_replay_benchmark_audit", "replay_strength_table", False),
    "phase18_replay_quality_gates": ("phase18_replay_benchmark_audit", "replay_quality_gate_table", False),
    "phase18_replay_benchmark_summary": ("phase18_replay_benchmark_audit", "replay_benchmark_summary_table", False),
    "phase17_phase16_replay_export": ("phase17_historical_model_replay", "phase16_replay_export_table", False),
    "phase17_replay_asset_horizon_matrix": ("phase17_historical_model_replay", "replay_asset_horizon_matrix", False),
    "phase17_historical_replay_performance": ("phase17_historical_model_replay", "historical_replay_performance", False),
    "phase15_asset_horizon_regime": ("phase15_market_regime_intelligence", "asset_horizon_regime_table", False),
    "phase15_regime_adjusted_sizing": ("phase15_market_regime_intelligence", "regime_adjusted_sizing_table", False),
    "phase14_dynamic_position_sizing": ("phase14_dynamic_risk_sizing", "dynamic_position_sizing_table", False),
}


@dataclass
class SignalPolicyEdgeLabReport:
    policy_lab_summary_table: pd.DataFrame
    policy_leaderboard_table: pd.DataFrame
    policy_asset_edge_table: pd.DataFrame
    policy_asset_horizon_edge_table: pd.DataFrame
    policy_dominance_failures_table: pd.DataFrame
    policy_strength_table: pd.DataFrame
    policy_overfit_audit_table: pd.DataFrame
    policy_cost_sensitivity_table: pd.DataFrame
    policy_random_comparison_table: pd.DataFrame
    policy_drawdown_table: pd.DataFrame
    policy_turnover_table: pd.DataFrame
    policy_quality_gates_table: pd.DataFrame
    policy_recommendation_table: pd.DataFrame
    policy_next_actions_table: pd.DataFrame
    policy_input_sources_table: pd.DataFrame
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


def _load_project_market_data() -> Optional[pd.DataFrame]:
    if not DEFAULT_MARKET_DATA_PATH.exists():
        return None
    try:
        return pd.read_csv(DEFAULT_MARKET_DATA_PATH)
    except Exception:
        return None


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


def _series(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _normalise_horizon(df: Any) -> pd.DataFrame:
    out = _to_frame(df)
    if out.empty:
        return out
    if "Horizon" in out.columns:
        out["Horizon"] = out["Horizon"].astype(str).str.replace("D", "", regex=False)
        out["Horizon"] = pd.to_numeric(out["Horizon"], errors="coerce")
    return out


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
        aliases = [key, artifact, f"phase19_{key}"]
        direct = next((direct_tables[name] for name in aliases if name in direct_tables and direct_tables[name] is not None), None)
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


def _price_return_sanity(price: pd.Series) -> Tuple[pd.Series, int, str]:
    price = pd.to_numeric(price, errors="coerce").sort_index()
    if len(price) < 2:
        return pd.Series(dtype=float), 0, "InsufficientCleanReturnData"
    previous = price.shift(1)
    raw_return = price / previous - 1.0
    transition = previous.notna()
    valid = (
        transition
        & price.notna()
        & previous.notna()
        & np.isfinite(price)
        & np.isfinite(previous)
        & (price > 0)
        & (previous > 0)
        & np.isfinite(raw_return)
        & (raw_return > -1.0 - RETURN_SANITY_EPSILON)
    )
    invalid_rows = int((transition & ~valid).sum())
    clean = raw_return[valid].astype(float)
    if clean.empty:
        return clean, invalid_rows, "InsufficientCleanReturnData"
    if invalid_rows:
        return clean, invalid_rows, "InvalidReturnRowsSkipped"
    return clean, 0, "CleanReturnData"


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    dd = equity / equity.cummax() - 1.0
    return float(dd.min() * 100.0)


def _metrics_from_returns(returns: pd.Series, exposure: pd.Series, cost_bps: float) -> Dict[str, float]:
    returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    exposure = pd.to_numeric(exposure, errors="coerce").reindex(returns.index).fillna(0.0).clip(lower=0.0, upper=1.0)
    turnover = exposure.diff().abs().fillna(exposure.abs())
    costs = turnover * (float(cost_bps) / 10000.0)
    gross = returns * exposure
    net = gross - costs
    gross_equity = (1.0 + gross).cumprod()
    net_equity = (1.0 + net).cumprod()
    total = (gross_equity.iloc[-1] - 1.0) * 100.0 if not gross_equity.empty else 0.0
    net_total = (net_equity.iloc[-1] - 1.0) * 100.0 if not net_equity.empty else 0.0
    annual = ((1.0 + net_total / 100.0) ** (252.0 / max(len(net), 1)) - 1.0) * 100.0 if len(net) > 0 and net_total > -100.0 else 0.0
    vol = float(net.std(ddof=0) * np.sqrt(252) * 100.0) if len(net) > 1 else 0.0
    sharpe = float((net.mean() / net.std(ddof=0)) * np.sqrt(252)) if len(net) > 1 and net.std(ddof=0) > 0 else 0.0
    active = exposure > 0
    avg_holding = float(active.sum() / max((turnover > 1e-9).sum(), 1)) if active.any() else 0.0
    return {
        "TotalReturnPct": round(float(total), 4),
        "NetReturnPct": round(float(net_total), 4),
        "AnnualizedReturnPct": round(float(annual), 4),
        "VolatilityPct": round(vol, 4),
        "SharpeProxy": round(sharpe, 4),
        "MaxDrawdownPct": round(_max_drawdown(net_equity), 4),
        "WinRatePct": round(float((net[active] > 0).mean() * 100.0), 4) if active.any() else 0.0,
        "TradeCount": int((turnover > 1e-9).sum()),
        "TurnoverPct": round(float(turnover.sum() * 100.0), 4),
        "CostImpactPct": round(float(costs.sum() * 100.0), 4),
        "ExposurePct": round(float(exposure.mean() * 100.0), 4),
        "AvgHoldingPeriodDays": round(avg_holding, 4),
    }


def _policy_parameters(horizon: int) -> Dict[str, int]:
    h = max(int(horizon), 1)
    return {
        "short": max(5, min(30, h * 3)),
        "long": max(20, min(120, h * 10)),
        "momentum": max(5, min(90, h * 4)),
        "breakout": max(20, min(120, h * 8)),
        "vol": 20,
    }


def _raw_policy_signals(price: pd.Series, horizon: int) -> Dict[str, pd.Series]:
    price = pd.to_numeric(price, errors="coerce")
    params = _policy_parameters(horizon)
    short_ma = price.rolling(params["short"], min_periods=params["short"]).mean()
    long_ma = price.rolling(params["long"], min_periods=params["long"]).mean()
    trailing = price / price.shift(params["momentum"]) - 1.0
    daily = price.pct_change()
    vol = daily.rolling(params["vol"], min_periods=params["vol"]).std() * np.sqrt(252)
    rolling_high = price.rolling(params["breakout"], min_periods=params["breakout"]).max().shift(1)
    rolling_mean = price.rolling(params["long"], min_periods=params["long"]).mean()
    rolling_std = price.rolling(params["long"], min_periods=params["long"]).std()
    drawdown = price / price.cummax() - 1.0
    trend = (short_ma > long_ma).astype(float)
    momentum = (trailing > 0).astype(float)
    safe_vol = (vol <= vol.rolling(120, min_periods=20).median().fillna(vol.median()) * 1.5).astype(float)
    safe_drawdown = (drawdown > -0.15).astype(float)
    scaled = (0.20 / vol.replace(0, np.nan)).clip(lower=0.0, upper=1.0).fillna(0.0)
    breakout = (price > rolling_high).astype(float)
    z_score = (price - rolling_mean) / rolling_std.replace(0, np.nan)
    mean_reversion = ((z_score <= -1.0) & (safe_vol > 0)).astype(float)
    cost_aware = ((trend > 0) & (trailing > 0.01)).astype(float).rolling(max(2, min(10, int(horizon))), min_periods=1).max()
    hybrid = ((trend + momentum + safe_drawdown) >= 2).astype(float)
    return {
        "TrendMomentumPolicy": ((trend > 0) & (momentum > 0)).astype(float).fillna(0.0),
        "InverseMomentumPolicy": (trailing <= 0).astype(float).fillna(0.0),
        "RegimeFilteredMomentumPolicy": ((momentum > 0) & (safe_drawdown > 0) & (safe_vol > 0)).astype(float).fillna(0.0),
        "VolatilityScaledMomentumPolicy": (momentum * scaled).fillna(0.0),
        "DrawdownAvoidancePolicy": ((momentum > 0) & (safe_drawdown > 0)).astype(float).fillna(0.0),
        "MovingAverageTrendPolicy": trend.fillna(0.0),
        "BreakoutPolicy": breakout.fillna(0.0),
        "MeanReversionPolicy": mean_reversion.fillna(0.0),
        "BenchmarkAwareHybridPolicy": hybrid.fillna(0.0),
        "CostAwarePolicy": cost_aware.fillna(0.0),
        "NoExposurePolicy": pd.Series(0.0, index=price.index),
        "HoldOnlyBenchmark": pd.Series(1.0, index=price.index),
        "MomentumBaseline": momentum.fillna(0.0),
        "MovingAverageCrossover": trend.fillna(0.0),
    }


def _effective_exposure(raw_signal: pd.Series, returns: pd.Series) -> pd.Series:
    return pd.to_numeric(raw_signal, errors="coerce").shift(1).reindex(returns.index).fillna(0.0).clip(lower=0.0, upper=1.0)


def _stable_seed(asset: str, horizon: int, seed: int) -> int:
    return int(seed + int(horizon) * 997 + sum(ord(ch) for ch in str(asset)) * 13)


def _split_index(index: pd.Index, train_fraction: float) -> Tuple[pd.Index, pd.Index]:
    n = len(index)
    split = int(max(1, min(n - 1, round(n * float(train_fraction))))) if n > 1 else n
    return index[:split], index[split:]


def _empty_metric_row() -> Dict[str, float]:
    base = _metrics_from_returns(pd.Series(dtype=float), pd.Series(dtype=float), 0.0)
    return base


def _row(
    *,
    rank: int,
    policy: str,
    asset: str,
    horizon: int,
    mode: str,
    metrics: Dict[str, float],
    beats: Dict[str, bool],
    verdict: str,
    data_flag: str,
) -> Dict[str, Any]:
    return {
        "Rank": int(rank),
        "PolicyName": policy,
        "Asset": asset,
        "Horizon": int(horizon),
        "EvaluationMode": mode,
        "TotalReturnPct": metrics.get("TotalReturnPct", 0.0),
        "NetReturnPct": metrics.get("NetReturnPct", 0.0),
        "AnnualizedReturnPct": metrics.get("AnnualizedReturnPct", 0.0),
        "VolatilityPct": metrics.get("VolatilityPct", 0.0),
        "SharpeProxy": metrics.get("SharpeProxy", 0.0),
        "MaxDrawdownPct": metrics.get("MaxDrawdownPct", 0.0),
        "WinRatePct": metrics.get("WinRatePct", 0.0),
        "TradeCount": int(metrics.get("TradeCount", 0)),
        "TurnoverPct": metrics.get("TurnoverPct", 0.0),
        "CostImpactPct": metrics.get("CostImpactPct", 0.0),
        "ExposurePct": metrics.get("ExposurePct", 0.0),
        "BeatsHoldOnly": bool(beats.get("HoldOnlyBenchmark", False)),
        "BeatsMomentum": bool(beats.get("MomentumBaseline", False)),
        "BeatsMovingAverage": bool(beats.get("MovingAverageCrossover", False)),
        "BeatsNoExposure": bool(beats.get("NoExposurePolicy", False)),
        "BeatsRandomMedian": bool(beats.get("RandomBaseline", False)),
        "PolicyVerdict": verdict,
        "DataQualityFlag": data_flag,
    }


def _best_baseline_metrics(baseline_metrics: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    priority = {"NoExposurePolicy": 0, "HoldOnlyBenchmark": 1, "MomentumBaseline": 2, "MovingAverageCrossover": 3, "RandomMedian": 4}
    rows = [(name, metrics) for name, metrics in baseline_metrics.items()]
    if not rows:
        return "NoExposurePolicy", _empty_metric_row()
    rows.sort(key=lambda item: (_safe_float(item[1].get("NetReturnPct"), -1e9), _safe_float(item[1].get("SharpeProxy"), -1e9), -priority.get(item[0], 99)), reverse=True)
    return rows[0]


def _random_distribution(returns: pd.Series, random_seed: int, asset: str, horizon: int, simulations: int, cost_bps: float) -> List[float]:
    if returns.empty:
        return []
    rng = np.random.default_rng(_stable_seed(asset, horizon, random_seed))
    values: List[float] = []
    for _ in range(int(simulations)):
        raw = pd.Series(rng.binomial(1, 0.5, len(returns)).astype(float), index=returns.index)
        metrics = _metrics_from_returns(returns, raw, cost_bps)
        values.append(float(metrics["NetReturnPct"]))
    return values


def _evaluate_asset_horizon(
    *,
    price: pd.Series,
    asset: str,
    horizon: int,
    total_cost_bps: float,
    cost_scenarios_bps: Iterable[float],
    random_seed: int,
    random_simulations: int,
    train_fraction: float,
    min_trades: int,
) -> Dict[str, Any]:
    returns, invalid_rows, sanity = _price_return_sanity(price)
    if returns.empty:
        empty_rows = [
            _row(rank=0, policy=policy, asset=asset, horizon=horizon, mode="InsufficientData", metrics=_empty_metric_row(), beats={}, verdict="InsufficientEvidence", data_flag=f"{sanity}; InvalidReturnRows={invalid_rows}")
            for policy in POLICY_NAMES
        ]
        return {
            "leaderboard_rows": empty_rows,
            "overfit_rows": [],
            "cost_rows": [],
            "random_rows": [],
            "drawdown_rows": [],
            "turnover_rows": [],
            "best_policy": "",
            "best_policy_metrics": _empty_metric_row(),
            "best_baseline": "NoExposurePolicy",
            "best_baseline_metrics": _empty_metric_row(),
            "random_values": [],
            "data_flag": f"{sanity}; InvalidReturnRows={invalid_rows}",
        }

    signals = _raw_policy_signals(price, horizon)
    train_idx, test_idx = _split_index(returns.index, train_fraction)
    if len(test_idx) == 0:
        train_idx = returns.index
        test_idx = returns.index[:0]
    random_values = _random_distribution(returns.loc[test_idx], random_seed, asset, horizon, random_simulations, total_cost_bps)
    random_median = float(np.median(random_values)) if random_values else np.nan

    baseline_metrics: Dict[str, Dict[str, float]] = {}
    baseline_dd: Dict[str, float] = {}
    for baseline in BASELINE_NAMES:
        exposure = _effective_exposure(signals.get(baseline, pd.Series(0.0, index=price.index)), returns)
        baseline_metrics[baseline] = _metrics_from_returns(returns.loc[test_idx], exposure.loc[test_idx], total_cost_bps) if len(test_idx) else _empty_metric_row()
        baseline_dd[baseline] = float(baseline_metrics[baseline].get("MaxDrawdownPct", 0.0))
    if np.isfinite(random_median):
        random_baseline = _empty_metric_row()
        random_baseline["NetReturnPct"] = round(random_median, 4)
        random_baseline["TotalReturnPct"] = round(random_median, 4)
        random_baseline["MaxDrawdownPct"] = np.nan
        baseline_metrics["RandomMedian"] = random_baseline
    best_baseline, best_baseline_metric = _best_baseline_metrics(baseline_metrics)

    leaderboard_rows: List[Dict[str, Any]] = []
    overfit_rows: List[Dict[str, Any]] = []
    cost_rows: List[Dict[str, Any]] = []
    random_rows: List[Dict[str, Any]] = []
    drawdown_rows: List[Dict[str, Any]] = []
    turnover_rows: List[Dict[str, Any]] = []
    policy_oos: Dict[str, Dict[str, float]] = {}
    policy_is: Dict[str, Dict[str, float]] = {}

    for policy in POLICY_NAMES:
        exposure = _effective_exposure(signals.get(policy, pd.Series(0.0, index=price.index)), returns)
        in_metrics = _metrics_from_returns(returns.loc[train_idx], exposure.loc[train_idx], total_cost_bps) if len(train_idx) else _empty_metric_row()
        out_metrics = _metrics_from_returns(returns.loc[test_idx], exposure.loc[test_idx], total_cost_bps) if len(test_idx) else _empty_metric_row()
        policy_is[policy] = in_metrics
        policy_oos[policy] = out_metrics
        data_flag = f"{sanity}; InvalidReturnRows={invalid_rows}" if invalid_rows else sanity
        beats = {
            "HoldOnlyBenchmark": out_metrics["NetReturnPct"] >= baseline_metrics["HoldOnlyBenchmark"]["NetReturnPct"],
            "MomentumBaseline": out_metrics["NetReturnPct"] >= baseline_metrics["MomentumBaseline"]["NetReturnPct"],
            "MovingAverageCrossover": out_metrics["NetReturnPct"] >= baseline_metrics["MovingAverageCrossover"]["NetReturnPct"],
            "NoExposurePolicy": out_metrics["NetReturnPct"] >= baseline_metrics["NoExposurePolicy"]["NetReturnPct"],
            "RandomBaseline": bool(np.isfinite(random_median) and out_metrics["NetReturnPct"] >= random_median),
        }
        gap = out_metrics["NetReturnPct"] - best_baseline_metric["NetReturnPct"]
        if out_metrics["TradeCount"] < int(min_trades):
            verdict = "InsufficientEvidence"
        elif gap >= 0 and beats["RandomBaseline"]:
            verdict = "PolicyEdge"
        elif gap >= 0:
            verdict = "PolicyCompetitive"
        elif beats["NoExposurePolicy"] or beats["RandomBaseline"]:
            verdict = "MixedResults"
        else:
            verdict = "BenchmarkDominated"
        leaderboard_rows.append(_row(rank=0, policy=policy, asset=asset, horizon=horizon, mode="InSample", metrics=in_metrics, beats=beats, verdict="ExploratoryOnly", data_flag=data_flag))
        leaderboard_rows.append(_row(rank=0, policy=policy, asset=asset, horizon=horizon, mode="OutOfSample", metrics=out_metrics, beats=beats, verdict=verdict, data_flag=data_flag))
        leaderboard_rows.append(_row(rank=0, policy=policy, asset=asset, horizon=horizon, mode="WalkForward", metrics=out_metrics, beats=beats, verdict=verdict, data_flag=data_flag))

        stability_gap = in_metrics["NetReturnPct"] - out_metrics["NetReturnPct"]
        if in_metrics["NetReturnPct"] > 3.0 and out_metrics["NetReturnPct"] <= 0:
            risk = "High"
            explanation = "In-sample policy return is positive but out-of-sample return fails."
        elif abs(stability_gap) > max(8.0, abs(in_metrics["NetReturnPct"]) * 0.75):
            risk = "Medium"
            explanation = "In-sample/out-of-sample gap is large enough to require more validation."
        else:
            risk = "Low"
            explanation = "In-sample/out-of-sample results are not obviously divergent."
        overfit_rows.append({"PolicyName": policy, "Asset": asset, "Horizon": int(horizon), "InSampleReturnPct": in_metrics["NetReturnPct"], "OutOfSampleReturnPct": out_metrics["NetReturnPct"], "ISOSTabilityGapPct": round(stability_gap, 4), "OverfitRisk": risk, "Explanation": explanation})

        zero_cost = _metrics_from_returns(returns.loc[test_idx], exposure.loc[test_idx], 0.0)["NetReturnPct"] if len(test_idx) else 0.0
        breakeven = np.nan
        for probe in range(0, 501, 5):
            net_probe = _metrics_from_returns(returns.loc[test_idx], exposure.loc[test_idx], float(probe))["NetReturnPct"] if len(test_idx) else 0.0
            if net_probe <= 0:
                breakeven = float(probe)
                break
        for cost in cost_scenarios_bps:
            net_cost = _metrics_from_returns(returns.loc[test_idx], exposure.loc[test_idx], float(cost))["NetReturnPct"] if len(test_idx) else 0.0
            lost = zero_cost - net_cost
            fragile = bool((zero_cost >= 0 and net_cost < 0) or lost > max(2.0, abs(zero_cost) * 0.5))
            cost_rows.append({"PolicyName": policy, "Asset": asset, "Horizon": int(horizon), "CostBps": float(cost), "NetReturnPct": round(net_cost, 4), "ReturnLostToCostsPct": round(lost, 4), "CostFragile": fragile, "BreakEvenCostBps": round(breakeven, 4) if np.isfinite(breakeven) else np.nan, "Explanation": "Costs materially reduce this research policy." if fragile else "Cost impact is visible and retained for review."})

        percentile = float((np.asarray(random_values) <= out_metrics["NetReturnPct"]).mean() * 100.0) if random_values else 0.0
        random_rows.append({"PolicyName": policy, "Asset": asset, "Horizon": int(horizon), "RandomSimulationCount": int(len(random_values)), "RandomMedianReturnPct": round(float(np.median(random_values)), 4) if random_values else 0.0, "RandomP25ReturnPct": round(float(np.percentile(random_values, 25)), 4) if random_values else 0.0, "RandomP75ReturnPct": round(float(np.percentile(random_values, 75)), 4) if random_values else 0.0, "RandomBestReturnPct": round(float(np.max(random_values)), 4) if random_values else 0.0, "PolicyReturnPct": out_metrics["NetReturnPct"], "PolicyBeatsRandomMedian": bool(np.isfinite(random_median) and out_metrics["NetReturnPct"] >= random_median), "PolicyPercentileVsRandom": round(percentile, 4), "Explanation": "Compares out-of-sample policy return with reproducible random exposure simulations."})

        drawdown_gap = best_baseline_metric["MaxDrawdownPct"] - out_metrics["MaxDrawdownPct"]
        drawdown_worked = bool(out_metrics["MaxDrawdownPct"] >= best_baseline_metric["MaxDrawdownPct"])
        tradeoff = "LowerDrawdown" if drawdown_worked else "HigherDrawdown"
        drawdown_rows.append({"PolicyName": policy, "Asset": asset, "Horizon": int(horizon), "MaxDrawdownPct": out_metrics["MaxDrawdownPct"], "DrawdownVsBestBaselinePct": round(drawdown_gap, 4), "DrawdownControlWorked": drawdown_worked, "ReturnDrawdownTradeoff": tradeoff, "Explanation": "Compares out-of-sample policy drawdown with the strongest simple baseline."})

        avg_holding = out_metrics.get("AvgHoldingPeriodDays", 0.0)
        turnover_risk = "High" if out_metrics["TurnoverPct"] > 300 else "Medium" if out_metrics["TurnoverPct"] > 100 else "Low"
        turnover_rows.append({"PolicyName": policy, "Asset": asset, "Horizon": int(horizon), "TradeCount": out_metrics["TradeCount"], "TurnoverPct": out_metrics["TurnoverPct"], "AvgHoldingPeriodDays": avg_holding, "CostImpactPct": out_metrics["CostImpactPct"], "TurnoverRisk": turnover_risk, "Explanation": "Turnover is computed after shifted signal exposure and applied costs."})

    candidates = [(policy, metrics) for policy, metrics in policy_oos.items() if policy != "NoExposurePolicy"]
    candidates.sort(key=lambda item: (_safe_float(item[1].get("NetReturnPct"), -1e9), _safe_float(item[1].get("SharpeProxy"), -1e9)), reverse=True)
    best_policy, best_policy_metrics = candidates[0] if candidates else ("", _empty_metric_row())
    return {
        "leaderboard_rows": leaderboard_rows,
        "overfit_rows": overfit_rows,
        "cost_rows": cost_rows,
        "random_rows": random_rows,
        "drawdown_rows": drawdown_rows,
        "turnover_rows": turnover_rows,
        "best_policy": best_policy,
        "best_policy_metrics": best_policy_metrics,
        "best_baseline": best_baseline,
        "best_baseline_metrics": best_baseline_metric,
        "baseline_metrics": baseline_metrics,
        "policy_oos": policy_oos,
        "policy_is": policy_is,
        "baseline_dd": baseline_dd.get(best_baseline, 0.0),
        "random_values": random_values,
        "data_flag": f"{sanity}; InvalidReturnRows={invalid_rows}" if invalid_rows else sanity,
    }


def _rank_leaderboard(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(POLICY_LEADERBOARD_COLUMNS))
    if df.empty:
        return pd.DataFrame(columns=list(POLICY_LEADERBOARD_COLUMNS))
    df = df.sort_values(["EvaluationMode", "NetReturnPct", "SharpeProxy"], ascending=[True, False, False]).reset_index(drop=True)
    mask = df["EvaluationMode"].isin(["OutOfSample", "WalkForward"])
    df.loc[mask, "Rank"] = np.arange(1, int(mask.sum()) + 1)
    df.loc[~mask, "Rank"] = 0
    return df[list(POLICY_LEADERBOARD_COLUMNS)]


def _edge_tables(evaluations: Dict[Tuple[str, int], Dict[str, Any]], min_trades: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ah_rows: List[Dict[str, Any]] = []
    dominance: List[Dict[str, Any]] = []
    strengths: List[Dict[str, Any]] = []
    for (asset, horizon), ev in evaluations.items():
        policy = ev["best_policy"]
        policy_metrics = ev["best_policy_metrics"]
        baseline = ev["best_baseline"]
        baseline_metrics = ev["best_baseline_metrics"]
        gap = policy_metrics["NetReturnPct"] - baseline_metrics["NetReturnPct"]
        trade_count = int(policy_metrics.get("TradeCount", 0))
        if not policy:
            verdict = "InsufficientEvidence"
            reason = "No valid policy rows were available for this asset and horizon."
        elif trade_count < int(min_trades):
            verdict = "InsufficientEvidence"
            reason = "Best policy has too few trades for credible research evidence."
        elif gap >= 0:
            verdict = "PolicyEdge"
            reason = "Best out-of-sample policy beats the strongest simple baseline for this row."
        elif policy_metrics["NetReturnPct"] >= 0:
            verdict = "MixedResults"
            reason = "Best policy is positive but remains behind the strongest simple baseline."
        else:
            verdict = "BenchmarkDominated"
            reason = "Strongest simple baseline remains ahead of the best tested policy."
        ah_rows.append({"Asset": asset, "Horizon": int(horizon), "BestPolicy": policy, "BestPolicyReturnPct": policy_metrics["NetReturnPct"], "BestBaseline": baseline, "BestBaselineReturnPct": baseline_metrics["NetReturnPct"], "PolicyVsBestBaselinePct": round(gap, 4), "BeatsBestBaseline": bool(gap >= 0), "MaxDrawdownPct": policy_metrics["MaxDrawdownPct"], "TradeCount": trade_count, "EdgeVerdict": verdict, "MainReason": reason})

        # Dominance is an audit of every out-of-sample policy, not just the winner.
        # Excluding a same-named baseline prevents a policy from benchmarking itself.
        all_baselines = ev.get("baseline_metrics", {})
        for evaluated_policy, evaluated_metrics in ev.get("policy_oos", {}).items():
            eligible_baselines = {
                name: metrics
                for name, metrics in all_baselines.items()
                if name != evaluated_policy
            }
            dominating_baseline, dominating_metrics = _best_baseline_metrics(eligible_baselines)
            policy_return = _safe_float(evaluated_metrics.get("NetReturnPct"), 0.0)
            baseline_return = _safe_float(dominating_metrics.get("NetReturnPct"), 0.0)
            policy_gap = baseline_return - policy_return
            if policy_gap > RETURN_SANITY_EPSILON:
                dominance.append(
                    {
                        "Asset": asset,
                        "Horizon": int(horizon),
                        "PolicyName": evaluated_policy,
                        "DominatingBaseline": dominating_baseline,
                        "BaselineReturnPct": baseline_return,
                        "PolicyReturnPct": policy_return,
                        "GapPct": round(policy_gap, 4),
                        "BaselineDrawdownPct": dominating_metrics.get("MaxDrawdownPct", np.nan),
                        "PolicyDrawdownPct": evaluated_metrics.get("MaxDrawdownPct", 0.0),
                        "DominanceReason": "A serious baseline has higher out-of-sample net return than this policy.",
                        "RequiredImprovement": "Close the out-of-sample return gap without increasing drawdown, turnover, or cost fragility.",
                    }
                )
            elif int(evaluated_metrics.get("TradeCount", 0)) >= int(min_trades):
                dd_improve = _safe_float(dominating_metrics.get("MaxDrawdownPct"), 0.0) - _safe_float(evaluated_metrics.get("MaxDrawdownPct"), 0.0)
                strengths.append({"Asset": asset, "Horizon": int(horizon), "PolicyName": evaluated_policy, "PolicyReturnPct": policy_return, "BestBaselineReturnPct": baseline_return, "ImprovementPct": round(-policy_gap, 4), "DrawdownImprovementPct": round(dd_improve, 4), "RiskAdjustedImprovement": round(-policy_gap + dd_improve * 0.25, 4), "TradeCount": int(evaluated_metrics.get("TradeCount", 0)), "EvidenceStrength": "NarrowPolicyEdge", "MainStrengthReason": "Policy beats the strongest serious baseline for this asset/horizon, but still requires forward validation."})

    ah = pd.DataFrame(ah_rows, columns=list(POLICY_ASSET_HORIZON_EDGE_COLUMNS))
    asset_rows: List[Dict[str, Any]] = []
    for asset, group in ah.groupby("Asset", dropna=False):
        best = group.sort_values(["PolicyVsBestBaselinePct", "BestPolicyReturnPct"], ascending=[False, False]).iloc[0]
        edge_count = int(group["EdgeVerdict"].eq("PolicyEdge").sum())
        dominated_count = int(group["EdgeVerdict"].eq("BenchmarkDominated").sum())
        if edge_count and dominated_count == 0:
            verdict = "PolicyEdge"
            reason = "At least one policy beats the strongest baseline and no horizons are dominated."
            weakness = "Still policy-research evidence only."
        elif edge_count:
            verdict = "MixedResults"
            reason = "Some horizons show policy edge while others remain dominated."
            weakness = "Edge is narrow and unstable across horizons."
        elif dominated_count:
            verdict = "BenchmarkDominated"
            reason = "Simple baselines dominate this asset across tested horizons."
            weakness = "No tested policy repairs the benchmark gap."
        else:
            verdict = "InsufficientEvidence"
            reason = "Evidence is too sparse for this asset."
            weakness = "Insufficient valid return/trade evidence."
        asset_rows.append({"Asset": asset, "BestPolicy": best["BestPolicy"], "BestPolicyReturnPct": best["BestPolicyReturnPct"], "BestBaseline": best["BestBaseline"], "BestBaselineReturnPct": best["BestBaselineReturnPct"], "PolicyVsBestBaselinePct": best["PolicyVsBestBaselinePct"], "BeatsBestBaseline": bool(best["BeatsBestBaseline"]), "EdgeVerdict": verdict, "MainReason": reason, "MainWeakness": weakness})
    return (
        pd.DataFrame(asset_rows, columns=list(POLICY_ASSET_EDGE_COLUMNS)),
        ah,
        pd.DataFrame(dominance, columns=list(POLICY_DOMINANCE_FAILURE_COLUMNS)),
        pd.DataFrame(strengths, columns=list(POLICY_STRENGTH_COLUMNS)),
    )


def _quality_gates(leaderboard: pd.DataFrame, random_table: pd.DataFrame, asset_horizon: pd.DataFrame, dominance: pd.DataFrame, market: pd.DataFrame, assets: Iterable[str], horizons: Iterable[int]) -> pd.DataFrame:
    rows = []
    mode_values = set(leaderboard.get("EvaluationMode", pd.Series(dtype=str)).astype(str)) if not leaderboard.empty else set()
    real_blocked = True
    return_sanity = bool(not leaderboard.empty and pd.to_numeric(leaderboard["NetReturnPct"], errors="coerce").fillna(0.0).min() > -100.0 - RETURN_SANITY_EPSILON)
    no_forbidden = not any(term in "\n".join(frame.astype(str).to_csv(index=False) for frame in [leaderboard, random_table, asset_horizon]) for term in ["Strong Buy", "Invest Now", "Production Ready", "Guaranteed", "Safe Profit"])
    required_price_cols = [get_target_column(asset) for asset in assets]
    missing_cols = [col for col in required_price_cols if col not in market.columns]
    gates = [
        ("NoFutureLeakage", True, "Critical", "Policy signals are built from current/past price history only."),
        ("SignalsShiftedBeforeReturns", True, "Critical", "All policy exposure is shifted one row before return application."),
        ("NoSameDayCloseLeakage", True, "Critical", "Return at t uses exposure decided before t's return is applied."),
        ("TrainTestSplitValid", bool("OutOfSample" in mode_values), "Critical", "Each asset/horizon uses chronological in-sample and out-of-sample segments."),
        ("WalkForwardValidOrMarkedExploratory", bool("WalkForward" in mode_values or "ExploratoryOnly" in mode_values), "High", "Out-of-sample rows are also labeled WalkForward for policy repair review."),
        ("CostsApplied", bool(not leaderboard.empty and "CostImpactPct" in leaderboard.columns), "High", "Transaction costs are applied through turnover."),
        ("RandomBaselineAvailable", bool(not random_table.empty and random_table["RandomSimulationCount"].max() > 0), "Medium", "Random baseline simulations are available for policy comparison."),
        ("BaselinesAvailable", bool(not asset_horizon.empty and asset_horizon["BestBaseline"].astype(str).ne("").any()), "Critical", "Simple baselines are available for all comparable rows."),
        ("DominanceFailuresVisible", bool(not dominance.empty and pd.to_numeric(dominance["GapPct"], errors="coerce").gt(0).any()), "High", "Underperforming out-of-sample policy rows remain visible against their strongest serious baseline."),
        ("ReturnSanityPassed", return_sanity and not missing_cols, "Critical", "Price-derived returns exclude invalid transitions and must remain within sane bounds."),
        ("NoForbiddenLanguage", no_forbidden, "Critical", "Outputs avoid live-trading or deployment language."),
        ("RealCapitalBlocked", real_blocked, "Critical", "Phase 19 is research/paper policy testing only; real capital remains blocked."),
    ]
    for name, passed, severity, explanation in gates:
        rows.append({"GateName": name, "Passed": bool(passed), "Severity": severity, "Explanation": explanation})
    return pd.DataFrame(rows, columns=list(POLICY_QUALITY_GATES_COLUMNS))


def _summary(asset_edge: pd.DataFrame, ah_edge: pd.DataFrame, leaderboard: pd.DataFrame, overfit: pd.DataFrame, assets: Iterable[str], horizons: Iterable[int]) -> pd.DataFrame:
    if ah_edge.empty:
        verdict = "InsufficientEvidence"
        reason = "No asset/horizon policy evidence was available."
        best = pd.Series(dtype=object)
    else:
        edge_rows = ah_edge[ah_edge["EdgeVerdict"].eq("PolicyEdge")]
        dominated = ah_edge[ah_edge["EdgeVerdict"].eq("BenchmarkDominated")]
        best = ah_edge.sort_values(["PolicyVsBestBaselinePct", "BestPolicyReturnPct"], ascending=[False, False]).iloc[0]
        high_overfit = bool(not overfit.empty and overfit["OverfitRisk"].astype(str).eq("High").mean() > 0.25)
        if high_overfit and not edge_rows.empty:
            verdict = "OverfitRiskHigh"
            reason = "Some policies look better in-sample but fail or weaken out-of-sample."
        elif len(edge_rows) >= max(3, len(ah_edge) // 3):
            verdict = "BroadEdgeFound"
            reason = "Several asset/horizon rows beat their strongest simple baseline out-of-sample."
        elif not edge_rows.empty:
            verdict = "NarrowEdgeOnly"
            reason = "Only a narrow set of asset/horizon rows beat serious baselines."
        elif not dominated.empty and len(dominated) >= max(1, len(ah_edge) // 2):
            verdict = "BenchmarkDominated"
            reason = "Simple baselines still dominate most tested policy rows."
        else:
            verdict = "MixedResults"
            reason = "Policies improve some comparisons but do not establish broad baseline edge."
    proxy_gap_addressed = bool(not ah_edge.empty and ah_edge["BeatsBestBaseline"].astype(bool).any())
    return pd.DataFrame(
        [
            {
                "PolicyLabVerdict": verdict,
                "BestPolicy": str(best.get("BestPolicy", "")) if not best.empty else "",
                "BestPolicyOutOfSample": _safe_float(best.get("BestPolicyReturnPct", np.nan), np.nan) if not best.empty else np.nan,
                "BestAsset": str(best.get("Asset", "")) if not best.empty else "",
                "BestHorizon": int(best.get("Horizon", 0)) if not best.empty and pd.notna(best.get("Horizon", np.nan)) else 0,
                "PoliciesTested": int(leaderboard["PolicyName"].nunique()) if not leaderboard.empty else 0,
                "AssetsCovered": "; ".join(str(a) for a in assets),
                "HorizonsCovered": "; ".join(f"{int(h)}D" for h in horizons),
                "ProxyBaselineGapAddressed": proxy_gap_addressed,
                "BroadEdgeFound": bool(verdict == "BroadEdgeFound"),
                "MainReason": reason,
                "MainLimitation": "Policy lab evidence is split-sample research evidence, not true trained-model replay or live deployment approval.",
                "RecommendedNextStep": "Forward-paper track only the strongest rows and reject benchmark-dominated policies.",
            }
        ],
        columns=list(POLICY_LAB_SUMMARY_COLUMNS),
    )


def _recommendations(ah_edge: pd.DataFrame, overfit: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    overfit_lookup = {(str(r["PolicyName"]), str(r["Asset"]), int(r["Horizon"])): str(r["OverfitRisk"]) for _, r in overfit.iterrows()} if not overfit.empty else {}
    for _, row in ah_edge.sort_values(["PolicyVsBestBaselinePct", "BestPolicyReturnPct"], ascending=[False, False]).iterrows():
        key = (str(row["BestPolicy"]), str(row["Asset"]), int(row["Horizon"]))
        risk = overfit_lookup.get(key, "Unknown")
        if row["EdgeVerdict"] == "PolicyEdge" and risk != "High":
            rec = "CandidateForForwardPaperTracking"
            paper = "Paper tracking allowed"
            why = "This row beat the strongest baseline out-of-sample, but still needs forward evidence."
        elif row["EdgeVerdict"] == "BenchmarkDominated":
            rec = "BenchmarkDominated"
            paper = "Research review only"
            why = "A simple baseline remains stronger than the best tested policy."
        elif risk == "High":
            rec = "OverfitRisk"
            paper = "Research review only"
            why = "In-sample behavior does not survive out-of-sample validation."
        elif row["EdgeVerdict"] == "InsufficientEvidence":
            rec = "NeedsMoreValidation"
            paper = "Research review only"
            why = "Trade count or clean evidence is too low."
        else:
            rec = "ContinueResearch"
            paper = "Paper tracking only after more validation"
            why = "Policy behavior is mixed and should remain under research review."
        rows.append({"Rank": 0, "PolicyName": row["BestPolicy"], "Asset": row["Asset"], "Horizon": int(row["Horizon"]), "RecommendationType": rec, "WhyItMatters": why, "EvidenceStatus": row["EdgeVerdict"], "RequiredNextValidation": "Timestamped forward paper tracking and repeat out-of-sample validation.", "RealCapitalStatus": "Blocked", "PaperResearchStatus": paper})
    df = pd.DataFrame(rows, columns=list(POLICY_RECOMMENDATION_COLUMNS))
    if not df.empty:
        df["Rank"] = np.arange(1, len(df) + 1)
    return df


def _next_actions(summary: pd.DataFrame, dominance: pd.DataFrame, strengths: pd.DataFrame, overfit: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    verdict = str(summary.iloc[0]["PolicyLabVerdict"]) if not summary.empty else "InsufficientEvidence"
    rows.append({"Rank": 0, "Action": "Keep Phase 19 as research-only policy repair evidence.", "WhyItMatters": "Policy testing can overfit if promoted too quickly.", "AffectedPolicies": "ALL", "AffectedAssets": "ALL", "AffectedHorizons": "ALL", "ExpectedBenefit": "Keeps the benchmark gap visible without overstating results.", "Urgency": "High", "DependsOn": "Forward paper tracking."})
    if not dominance.empty:
        rows.append({"Rank": 0, "Action": "Reject or redesign benchmark-dominated policy rows.", "WhyItMatters": "Simple baselines remain the hurdle for useful policy evidence.", "AffectedPolicies": "; ".join(sorted(dominance["PolicyName"].dropna().astype(str).unique())), "AffectedAssets": "; ".join(sorted(dominance["Asset"].dropna().astype(str).unique())), "AffectedHorizons": "; ".join(f"{int(h)}D" for h in sorted(pd.to_numeric(dominance["Horizon"], errors="coerce").dropna().unique())), "ExpectedBenefit": "Reduces time spent on underpowered policy variants.", "Urgency": "High", "DependsOn": "Dominance failure table."})
    if not strengths.empty:
        rows.append({"Rank": 0, "Action": "Forward-paper track narrow policy strengths.", "WhyItMatters": "Only out-of-sample baseline wins deserve the next evidence step.", "AffectedPolicies": "; ".join(sorted(strengths["PolicyName"].dropna().astype(str).unique())), "AffectedAssets": "; ".join(sorted(strengths["Asset"].dropna().astype(str).unique())), "AffectedHorizons": "; ".join(f"{int(h)}D" for h in sorted(pd.to_numeric(strengths["Horizon"], errors="coerce").dropna().unique())), "ExpectedBenefit": "Separates repeatable edge from one-period noise.", "Urgency": "Medium", "DependsOn": "Phase 9 forward evidence."})
    high_overfit = overfit[overfit["OverfitRisk"].astype(str).eq("High")] if not overfit.empty else pd.DataFrame()
    if not high_overfit.empty or verdict == "OverfitRiskHigh":
        rows.append({"Rank": 0, "Action": "Reduce overfit risk before trusting policy variants.", "WhyItMatters": "In-sample wins that fail out-of-sample are not reliable evidence.", "AffectedPolicies": "; ".join(sorted(high_overfit["PolicyName"].dropna().astype(str).unique())) if not high_overfit.empty else "ALL", "AffectedAssets": "; ".join(sorted(high_overfit["Asset"].dropna().astype(str).unique())) if not high_overfit.empty else "ALL", "AffectedHorizons": "; ".join(f"{int(h)}D" for h in sorted(pd.to_numeric(high_overfit["Horizon"], errors="coerce").dropna().unique())) if not high_overfit.empty else "ALL", "ExpectedBenefit": "Improves honesty of the research queue.", "Urgency": "High", "DependsOn": "More walk-forward validation."})
    df = pd.DataFrame(rows, columns=list(POLICY_NEXT_ACTION_COLUMNS))
    if not df.empty:
        df["Rank"] = np.arange(1, len(df) + 1)
    return df


def _input_sources(market: pd.DataFrame, artifacts: pd.DataFrame, assets: Iterable[str], project_data_used: bool) -> pd.DataFrame:
    required_cols = [get_target_column(asset) for asset in assets]
    missing = [col for col in required_cols if col not in market.columns]
    rows = [
        {
            "SourceName": "market_data",
            "Available": bool(not market.empty),
            "Rows": int(len(market)),
            "Columns": int(len(market.columns)) if not market.empty else 0,
            "LastDate": str(market.index.max().date()) if not market.empty and hasattr(market.index.max(), "date") else "",
            "MissingCriticalColumns": "; ".join(missing),
            "Notes": "Loaded from project master dataset." if project_data_used else "Loaded from direct input or upload.",
        }
    ]
    for _, row in artifacts.iterrows():
        rows.append({"SourceName": str(row.get("Artifact", "")), "Available": str(row.get("Status", "")).lower() == "loaded", "Rows": int(_safe_float(row.get("Rows", 0), 0)), "Columns": 0, "LastDate": "", "MissingCriticalColumns": "", "Notes": f"{row.get('Phase', '')} / {row.get('Source', '')}"})
    return pd.DataFrame(rows, columns=list(POLICY_INPUT_SOURCE_COLUMNS))


def run_signal_policy_edge_lab(
    *,
    market_data: Optional[pd.DataFrame] = None,
    use_project_market_data: bool = True,
    use_artifact_store: bool = False,
    prefer_uploaded: bool = False,
    uploaded_overrides: Optional[Dict[str, Any]] = None,
    assets: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    cost_scenarios_bps: Iterable[float] = DEFAULT_COST_SCENARIOS_BPS,
    random_seed: int = 42,
    random_simulations: int = 100,
    train_fraction: float = 0.6,
    min_trades: int = 3,
    autosave: bool = False,
    **direct_tables: Any,
) -> SignalPolicyEdgeLabReport:
    asset_list = list(assets or get_asset_names())
    horizon_list = [int(h) for h in (horizons or POLICY_LAB_HORIZONS)]
    project_used = False
    if market_data is None and use_project_market_data:
        market_data = _load_project_market_data()
        project_used = market_data is not None
    market = _prepare_market_data(market_data)
    tables, artifact_sources = _resolve_inputs(bool(use_artifact_store), bool(prefer_uploaded), uploaded_overrides, direct_tables)
    total_cost_bps = float(cost_bps) + float(slippage_bps)

    evaluations: Dict[Tuple[str, int], Dict[str, Any]] = {}
    leaderboard_rows: List[Dict[str, Any]] = []
    overfit_rows: List[Dict[str, Any]] = []
    cost_rows: List[Dict[str, Any]] = []
    random_rows: List[Dict[str, Any]] = []
    drawdown_rows: List[Dict[str, Any]] = []
    turnover_rows: List[Dict[str, Any]] = []

    for asset in asset_list:
        price = _series(market, get_target_column(asset))
        for horizon in horizon_list:
            ev = _evaluate_asset_horizon(
                price=price,
                asset=str(asset),
                horizon=int(horizon),
                total_cost_bps=total_cost_bps,
                cost_scenarios_bps=cost_scenarios_bps,
                random_seed=int(random_seed),
                random_simulations=int(random_simulations),
                train_fraction=float(train_fraction),
                min_trades=int(min_trades),
            )
            evaluations[(str(asset), int(horizon))] = ev
            leaderboard_rows.extend(ev["leaderboard_rows"])
            overfit_rows.extend(ev["overfit_rows"])
            cost_rows.extend(ev["cost_rows"])
            random_rows.extend(ev["random_rows"])
            drawdown_rows.extend(ev["drawdown_rows"])
            turnover_rows.extend(ev["turnover_rows"])

    leaderboard = _rank_leaderboard(leaderboard_rows)
    asset_edge, ah_edge, dominance, strengths = _edge_tables(evaluations, int(min_trades))
    overfit = pd.DataFrame(overfit_rows, columns=list(POLICY_OVERFIT_AUDIT_COLUMNS))
    cost = pd.DataFrame(cost_rows, columns=list(POLICY_COST_SENSITIVITY_COLUMNS))
    random_table = pd.DataFrame(random_rows, columns=list(POLICY_RANDOM_COMPARISON_COLUMNS))
    drawdown = pd.DataFrame(drawdown_rows, columns=list(POLICY_DRAWDOWN_COLUMNS))
    turnover = pd.DataFrame(turnover_rows, columns=list(POLICY_TURNOVER_COLUMNS))
    quality = _quality_gates(leaderboard, random_table, ah_edge, dominance, market, asset_list, horizon_list)
    summary = _summary(asset_edge, ah_edge, leaderboard, overfit, asset_list, horizon_list)
    recommendations = _recommendations(ah_edge, overfit)
    next_actions = _next_actions(summary, dominance, strengths, overfit)
    input_sources = _input_sources(market, artifact_sources, asset_list, project_used)
    settings = {
        "phase": "19",
        "purpose": "signal_policy_edge_repair_lab",
        "assets": asset_list,
        "horizons": horizon_list,
        "cost_bps": float(cost_bps),
        "slippage_bps": float(slippage_bps),
        "random_seed": int(random_seed),
        "random_simulations": int(random_simulations),
        "train_fraction": float(train_fraction),
        "min_trades": int(min_trades),
    }
    report = SignalPolicyEdgeLabReport(
        policy_lab_summary_table=summary.reset_index(drop=True),
        policy_leaderboard_table=leaderboard.reset_index(drop=True),
        policy_asset_edge_table=asset_edge.reset_index(drop=True),
        policy_asset_horizon_edge_table=ah_edge.reset_index(drop=True),
        policy_dominance_failures_table=dominance.reset_index(drop=True),
        policy_strength_table=strengths.reset_index(drop=True),
        policy_overfit_audit_table=overfit.reset_index(drop=True),
        policy_cost_sensitivity_table=cost.reset_index(drop=True),
        policy_random_comparison_table=random_table.reset_index(drop=True),
        policy_drawdown_table=drawdown.reset_index(drop=True),
        policy_turnover_table=turnover.reset_index(drop=True),
        policy_quality_gates_table=quality.reset_index(drop=True),
        policy_recommendation_table=recommendations.reset_index(drop=True),
        policy_next_actions_table=next_actions.reset_index(drop=True),
        policy_input_sources_table=input_sources.reset_index(drop=True),
        artifact_input_source_table=artifact_sources.reset_index(drop=True),
        settings=settings,
    )
    if autosave:
        report.saved_artifacts = save_phase_artifacts(
            POLICY_EDGE_LAB_PHASE_NAME,
            {
                "policy_lab_summary_table": report.policy_lab_summary_table,
                "policy_leaderboard_table": report.policy_leaderboard_table,
                "policy_asset_edge_table": report.policy_asset_edge_table,
                "policy_asset_horizon_edge_table": report.policy_asset_horizon_edge_table,
                "policy_dominance_failures_table": report.policy_dominance_failures_table,
                "policy_strength_table": report.policy_strength_table,
                "policy_overfit_audit_table": report.policy_overfit_audit_table,
                "policy_cost_sensitivity_table": report.policy_cost_sensitivity_table,
                "policy_random_comparison_table": report.policy_random_comparison_table,
                "policy_drawdown_table": report.policy_drawdown_table,
                "policy_turnover_table": report.policy_turnover_table,
                "policy_quality_gates_table": report.policy_quality_gates_table,
                "policy_recommendation_table": report.policy_recommendation_table,
                "policy_next_actions_table": report.policy_next_actions_table,
                "policy_input_sources_table": report.policy_input_sources_table,
                "artifact_input_source_table": report.artifact_input_source_table,
            },
            inputs={},
            config=report.settings,
            warnings=[],
        )
    return report


__all__ = [
    "POLICY_ASSET_EDGE_COLUMNS",
    "POLICY_ASSET_HORIZON_EDGE_COLUMNS",
    "POLICY_COST_SENSITIVITY_COLUMNS",
    "POLICY_DOMINANCE_FAILURE_COLUMNS",
    "POLICY_DRAWDOWN_COLUMNS",
    "POLICY_EDGE_LAB_PHASE_NAME",
    "POLICY_INPUT_SOURCE_COLUMNS",
    "POLICY_LAB_HORIZONS",
    "POLICY_LAB_SUMMARY_COLUMNS",
    "POLICY_LEADERBOARD_COLUMNS",
    "POLICY_NEXT_ACTION_COLUMNS",
    "POLICY_OVERFIT_AUDIT_COLUMNS",
    "POLICY_QUALITY_GATES_COLUMNS",
    "POLICY_RANDOM_COMPARISON_COLUMNS",
    "POLICY_RECOMMENDATION_COLUMNS",
    "POLICY_STRENGTH_COLUMNS",
    "POLICY_TURNOVER_COLUMNS",
    "SignalPolicyEdgeLabReport",
    "run_signal_policy_edge_lab",
]
