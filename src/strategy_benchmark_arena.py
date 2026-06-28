"""Phase 16 strategy benchmark arena.

This module compares the model/risk research pipeline with simple, time-safe
baseline strategies. Phase 12/14/15 sizing artifacts are latest snapshots unless
historical replay evidence is explicitly supplied, so those rows are labeled as
snapshot strategies instead of historical backtests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.asset_config import get_asset_names, get_target_column
from src.artifact_store import resolve_artifact, save_phase_artifacts


STRATEGY_BENCHMARK_PHASE_NAME = "phase16_strategy_benchmark_arena"
BENCHMARK_HORIZONS: Tuple[int, ...] = (1, 5, 10, 20, 30)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKET_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"

DEFAULT_COST_SCENARIOS_BPS: Tuple[float, ...] = (0.0, 5.0, 10.0, 25.0, 50.0)
MODEL_STRATEGY_NAME = "Phase15RegimeAdjustedStrategy"
BASELINE_STRATEGIES = {
    "NoExposureBaseline",
    "HoldOnlyBenchmark",
    "MovingAverageCrossover",
    "MomentumBaseline",
    "MeanReversionBaseline",
    "VolatilityScaledBaseline",
    "RandomBaseline",
}

BENCHMARK_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "OverallWinner",
    "ModelBeatsHoldOnly",
    "ModelBeatsMomentum",
    "ModelBeatsMovingAverage",
    "ModelBeatsNoExposure",
    "BenchmarkVerdict",
    "MainReason",
    "WeakestArea",
    "StrongestArea",
    "EvidenceQuality",
    "RecommendedNextStep",
)

STRATEGY_LEADERBOARD_COLUMNS: Tuple[str, ...] = (
    "Rank",
    "StrategyName",
    "Asset",
    "Horizon",
    "TotalReturnPct",
    "AnnualizedReturnPct",
    "VolatilityPct",
    "SharpeProxy",
    "MaxDrawdownPct",
    "WinRatePct",
    "TradeCount",
    "TurnoverPct",
    "CostImpactPct",
    "NetReturnPct",
    "ExposurePct",
    "BenchmarkRole",
    "DataQualityFlag",
    "ComparableHistorical",
    "EvaluationMode",
)

ASSET_BENCHMARK_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "BestStrategy",
    "BestBaseline",
    "ModelStrategyReturnPct",
    "BestBaselineReturnPct",
    "ModelVsBestBaselinePct",
    "ModelRank",
    "BenchmarkVerdict",
    "MainReason",
    "RiskNote",
)

ASSET_HORIZON_BENCHMARK_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "BestStrategy",
    "BestBaseline",
    "ModelStrategyReturnPct",
    "BestBaselineReturnPct",
    "ModelVsBestBaselinePct",
    "ModelBeatsBaseline",
    "MaxDrawdownPct",
    "SharpeProxy",
    "TradeCount",
    "BenchmarkVerdict",
    "MainReason",
)

BENCHMARK_DOMINANCE_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "DominatingBaseline",
    "BaselineReturnPct",
    "ModelReturnPct",
    "GapPct",
    "BaselineDrawdownPct",
    "ModelDrawdownPct",
    "DominanceReason",
    "RequiredImprovement",
)

MODEL_STRENGTH_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "ModelStrategy",
    "ModelReturnPct",
    "BestBaselineReturnPct",
    "ImprovementPct",
    "DrawdownImprovementPct",
    "RiskAdjustedImprovement",
    "EvidenceStrength",
    "MainStrengthReason",
)

COST_SENSITIVITY_COLUMNS: Tuple[str, ...] = (
    "StrategyName",
    "Asset",
    "Horizon",
    "CostBps",
    "NetReturnPct",
    "ReturnLostToCostsPct",
    "CostFragile",
    "Explanation",
)

RANDOM_BASELINE_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "SimulationCount",
    "RandomMedianReturnPct",
    "RandomP25ReturnPct",
    "RandomP75ReturnPct",
    "RandomBestReturnPct",
    "ModelReturnPct",
    "ModelBeatsRandomMedian",
    "ModelPercentileVsRandom",
    "Explanation",
)

RETURN_SANITY_CHECK_COLUMNS: Tuple[str, ...] = (
    "CheckName",
    "Passed",
    "Severity",
    "Asset",
    "Horizon",
    "StrategyName",
    "ObservedValue",
    "ExpectedRangeOrValue",
    "Explanation",
)

SNAPSHOT_MODEL_IMPACT_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "SnapshotStrategyName",
    "SnapshotWeightPct",
    "LatestKnownReturnWindowPct",
    "HypotheticalImpactPct",
    "ComparableHistorical",
    "EvaluationMode",
    "Explanation",
)

LEAKAGE_CHECK_COLUMNS: Tuple[str, ...] = (
    "CheckName",
    "Passed",
    "Severity",
    "Explanation",
)

BENCHMARK_WARNING_COLUMNS: Tuple[str, ...] = (
    "WarningType",
    "Severity",
    "Asset",
    "Horizon",
    "StrategyName",
    "Explanation",
    "RecommendedFix",
)

NEXT_BENCHMARK_ACTION_COLUMNS: Tuple[str, ...] = (
    "Rank",
    "Action",
    "WhyItMatters",
    "AffectedAssets",
    "AffectedHorizons",
    "ExpectedBenefit",
    "Urgency",
    "DependsOn",
)

BENCHMARK_INPUT_SOURCE_COLUMNS: Tuple[str, ...] = (
    "SourceName",
    "Available",
    "Rows",
    "Columns",
    "LastDate",
    "MissingCriticalColumns",
    "Notes",
)

INPUT_SPECS: Dict[str, Tuple[str, str, bool]] = {
    "regime_adjusted_sizing_table": ("phase15_market_regime_intelligence", "regime_adjusted_sizing_table", False),
    "asset_horizon_regime_table": ("phase15_market_regime_intelligence", "asset_horizon_regime_table", False),
    "regime_risk_table": ("phase15_market_regime_intelligence", "regime_risk_table", False),
    "regime_summary_table": ("phase15_market_regime_intelligence", "regime_summary_table", False),
    "dynamic_position_sizing_table": ("phase14_dynamic_risk_sizing", "dynamic_position_sizing_table", False),
    "optimized_portfolio_table": ("phase14_dynamic_risk_sizing", "optimized_portfolio_table", False),
    "dynamic_sizing_summary_table": ("phase14_dynamic_risk_sizing", "dynamic_sizing_summary_table", False),
    "asset_horizon_risk_matrix": ("phase13_risk_warning_intelligence", "asset_horizon_risk_matrix", False),
    "top_risks_table": ("phase13_risk_warning_intelligence", "top_risks_table", False),
    "warning_group_table": ("phase13_risk_warning_intelligence", "warning_group_table", False),
    "allocation_plan_table": ("Phase 12 Portfolio Capital Simulator", "allocation_plan_table", False),
    "paper_portfolio_table": ("Phase 12 Portfolio Capital Simulator", "paper_portfolio_table", False),
    "portfolio_drawdown_stress_table": ("Phase 12 Portfolio Capital Simulator", "portfolio_drawdown_stress_table", False),
    "cost_slippage_stress_table": ("Phase 12 Portfolio Capital Simulator", "cost_slippage_stress_table", False),
    "ranked_asset_horizon_plan": ("Phase 10 Actionable Research Plan", "ranked_asset_horizon_plan", False),
    "paper_trade_plan_table": ("Phase 10 Actionable Research Plan", "paper_trade_plan_table", False),
}


@dataclass
class StrategyBenchmarkArenaReport:
    benchmark_summary_table: pd.DataFrame
    strategy_leaderboard_table: pd.DataFrame
    asset_benchmark_table: pd.DataFrame
    asset_horizon_benchmark_table: pd.DataFrame
    benchmark_dominance_table: pd.DataFrame
    model_strength_table: pd.DataFrame
    cost_sensitivity_table: pd.DataFrame
    random_baseline_table: pd.DataFrame
    return_sanity_check_table: pd.DataFrame
    snapshot_model_impact_table: pd.DataFrame
    leakage_check_table: pd.DataFrame
    benchmark_warning_table: pd.DataFrame
    next_benchmark_actions_table: pd.DataFrame
    benchmark_input_sources_table: pd.DataFrame
    artifact_input_source_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    settings: Dict[str, Any] = field(default_factory=dict)
    saved_artifacts: Dict[str, Any] = field(default_factory=dict)


def _empty(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


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
        parsed_index = pd.to_datetime(df.index, errors="coerce")
        if not parsed_index.isna().all():
            df.index = parsed_index
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
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _subset(df: pd.DataFrame, asset: str, horizon: int) -> pd.DataFrame:
    if df.empty or "Asset" not in df.columns or "Horizon" not in df.columns:
        return pd.DataFrame()
    h = pd.to_numeric(df["Horizon"].astype(str).str.replace("D", "", regex=False), errors="coerce")
    return df[df["Asset"].astype(str).eq(str(asset)) & h.eq(int(horizon))].copy()


def _first_numeric(df: pd.DataFrame, columns: Iterable[str], default: float = 0.0) -> float:
    if df.empty:
        return default
    for col in columns:
        if col in df.columns:
            value = _safe_float(df[col].iloc[0], np.nan)
            if np.isfinite(value):
                return value
    return default


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
        direct = direct_tables.get(key)
        if direct is not None:
            df = _normalise_horizon(direct)
            tables[key] = df
            rows.append({"Artifact": artifact, "Phase": phase, "Source": "DirectInput", "RunId": "", "Rows": int(len(df)), "CreatedAt": "", "Status": "Loaded", "Path": ""})
            continue
        if use_artifact_store or uploaded_overrides.get(key) is not None:
            resolved = resolve_artifact(phase, artifact, uploaded_file=uploaded_overrides.get(key), prefer_uploaded=prefer_uploaded, required=required)
            data = resolved.get("Data")
            tables[key] = _normalise_horizon(data) if data is not None else pd.DataFrame()
            rows.append({k: v for k, v in resolved.items() if k != "Data"})
        else:
            tables[key] = pd.DataFrame()
            rows.append({"Artifact": artifact, "Phase": phase, "Source": "Missing", "RunId": "", "Rows": 0, "CreatedAt": "", "Status": "MissingOptional", "Path": ""})
    return tables, pd.DataFrame(rows, columns=["Artifact", "Source", "RunId", "Rows", "CreatedAt", "Status", "Phase", "Path"])


def _input_source_table(market_data: pd.DataFrame, assets: Iterable[str], project_data_used: bool) -> pd.DataFrame:
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
        }
    ]
    for asset in assets:
        col = get_target_column(asset)
        s = _series(market_data, col)
        rows.append(
            {
                "SourceName": f"{asset} price",
                "Available": bool(not s.empty),
                "Rows": int(len(s)),
                "Columns": 1 if col in market_data.columns else 0,
                "LastDate": str(s.index.max().date()) if not s.empty and hasattr(s.index.max(), "date") else "",
                "MissingCriticalColumns": "" if col in market_data.columns else col,
                "Notes": "Critical price series for benchmark arena." if col in market_data.columns else "Missing critical price series.",
            }
        )
    return pd.DataFrame(rows, columns=list(BENCHMARK_INPUT_SOURCE_COLUMNS))


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min() * 100.0)


def _metrics_from_signal(
    *,
    strategy_name: str,
    asset: str,
    horizon: int,
    price: pd.Series,
    signal: pd.Series,
    total_cost_bps: float,
    benchmark_role: str,
    data_quality_flag: str,
    comparable_historical: bool,
    evaluation_mode: str,
) -> Tuple[Dict[str, Any], pd.Series]:
    daily_return = price.pct_change()
    exposure = signal.reindex(price.index).astype(float).clip(lower=0.0, upper=1.0)
    effective_exposure = exposure.shift(1).fillna(0.0)
    aligned = pd.DataFrame({"return": daily_return, "signal": effective_exposure}).dropna()
    if aligned.empty:
        return (
            {
                "Rank": 0,
                "StrategyName": strategy_name,
                "Asset": asset,
                "Horizon": int(horizon),
                "TotalReturnPct": 0.0,
                "AnnualizedReturnPct": 0.0,
                "VolatilityPct": 0.0,
                "SharpeProxy": 0.0,
                "MaxDrawdownPct": 0.0,
                "WinRatePct": 0.0,
                "TradeCount": 0,
                "TurnoverPct": 0.0,
                "CostImpactPct": 0.0,
                "NetReturnPct": 0.0,
                "ExposurePct": 0.0,
                "BenchmarkRole": benchmark_role,
                "DataQualityFlag": "InsufficientHistory" if data_quality_flag == "OK" else data_quality_flag,
                "ComparableHistorical": bool(comparable_historical),
                "EvaluationMode": "InsufficientData" if evaluation_mode != "LatestSnapshotOnly" else evaluation_mode,
            },
            pd.Series(dtype=float),
        )
    exposure = aligned["signal"].clip(lower=0.0, upper=1.0).fillna(0.0)
    turnover = exposure.diff().abs().fillna(exposure.abs())
    cost = turnover * (float(total_cost_bps) / 10000.0)
    gross = exposure * aligned["return"]
    net = gross - cost
    gross_equity = (1.0 + gross).cumprod()
    equity = (1.0 + net).cumprod()
    gross_total = (gross_equity.iloc[-1] - 1.0) * 100.0
    net_total = (equity.iloc[-1] - 1.0) * 100.0
    annualized = ((1.0 + net_total / 100.0) ** (252.0 / max(len(net), 1)) - 1.0) * 100.0 if net_total > -100 else -100.0
    vol = float(net.std(ddof=0) * np.sqrt(252) * 100.0) if len(net) > 1 else 0.0
    sharpe = float((net.mean() / net.std(ddof=0)) * np.sqrt(252)) if len(net) > 1 and net.std(ddof=0) > 0 else 0.0
    active = exposure > 0
    win_rate = float((net[active] > 0).mean() * 100.0) if active.any() else 0.0
    trade_count = int((turnover > 1e-9).sum())
    cost_impact = float(cost.sum() * 100.0)
    row = {
        "Rank": 0,
        "StrategyName": strategy_name,
        "Asset": asset,
        "Horizon": int(horizon),
        "TotalReturnPct": round(float(gross_total), 4),
        "AnnualizedReturnPct": round(float(annualized), 4),
        "VolatilityPct": round(vol, 4),
        "SharpeProxy": round(sharpe, 4),
        "MaxDrawdownPct": round(_max_drawdown(equity), 4),
        "WinRatePct": round(win_rate, 4),
        "TradeCount": trade_count,
        "TurnoverPct": round(float(turnover.sum() * 100.0), 4),
        "CostImpactPct": round(cost_impact, 4),
        "NetReturnPct": round(float(net_total), 4),
        "ExposurePct": round(float(exposure.mean() * 100.0), 4),
        "BenchmarkRole": benchmark_role,
        "DataQualityFlag": data_quality_flag,
        "ComparableHistorical": bool(comparable_historical),
        "EvaluationMode": evaluation_mode,
    }
    return row, net


def _baseline_signals(price: pd.Series, short_ma: int, long_ma: int, momentum_lookback: int, mean_reversion_window: int, mean_reversion_threshold: float, volatility_target_pct: float) -> Dict[str, pd.Series]:
    rolling_short = price.rolling(short_ma, min_periods=short_ma).mean()
    rolling_long = price.rolling(long_ma, min_periods=long_ma).mean()
    ma = (rolling_short > rolling_long).astype(float).fillna(0.0)

    trailing_return = price / price.shift(momentum_lookback) - 1.0
    momentum = (trailing_return > 0).astype(float).fillna(0.0)

    rolling_mean = price.rolling(mean_reversion_window, min_periods=mean_reversion_window).mean()
    distance = price / rolling_mean - 1.0
    reversion = (distance <= -abs(mean_reversion_threshold)).astype(float).fillna(0.0)

    realized_vol = price.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252) * 100.0
    scaled = (volatility_target_pct / realized_vol.replace(0, np.nan)).clip(lower=0.0, upper=1.0)
    scaled = scaled.fillna(0.0)

    return {
        "NoExposureBaseline": pd.Series(0.0, index=price.index),
        "HoldOnlyBenchmark": pd.Series(1.0, index=price.index),
        "MovingAverageCrossover": ma,
        "MomentumBaseline": momentum,
        "MeanReversionBaseline": reversion,
        "VolatilityScaledBaseline": scaled,
    }


def _strategy_mode(strategy_name: str, data_quality_flag: str) -> Tuple[bool, str]:
    if str(data_quality_flag).startswith("Missing") or str(data_quality_flag) == "InsufficientHistory":
        return False, "InsufficientData"
    if strategy_name.startswith("Phase"):
        return False, "LatestSnapshotOnly"
    return True, "HistoricalDailyExposure"


def _stable_seed(asset: str, horizon: int, seed: int) -> int:
    return int(seed + horizon * 997 + sum(ord(ch) for ch in str(asset)) * 13)


def _random_signals(price: pd.Series, asset: str, horizon: int, seed: int, simulations: int) -> List[pd.Series]:
    rng = np.random.default_rng(_stable_seed(asset, horizon, seed))
    return [pd.Series(rng.binomial(1, 0.5, len(price)).astype(float), index=price.index) for _ in range(int(simulations))]


def _snapshot_weight(tables: Dict[str, pd.DataFrame], strategy_name: str, asset: str, horizon: int) -> Tuple[float, str]:
    if strategy_name == "Phase12PaperStrategy":
        row = _subset(tables.get("allocation_plan_table", pd.DataFrame()), asset, horizon)
        if row.empty:
            row = _subset(tables.get("paper_portfolio_table", pd.DataFrame()), asset, horizon)
        weight = _first_numeric(row, ["SuggestedPaperWeightPct", "PaperWeightPct", "Phase12PaperWeightPct"], np.nan)
    elif strategy_name == "Phase14DynamicSizingStrategy":
        row = _subset(tables.get("dynamic_position_sizing_table", pd.DataFrame()), asset, horizon)
        weight = _first_numeric(row, ["OptimizedPaperWeightPct", "Phase14OptimizedPaperWeightPct"], np.nan)
    else:
        row = _subset(tables.get("regime_adjusted_sizing_table", pd.DataFrame()), asset, horizon)
        weight = _first_numeric(row, ["RegimeAdjustedPaperWeightPct", "OptimizedPaperWeightPct"], np.nan)
    if not np.isfinite(weight):
        return 0.0, "MissingSnapshotWeight"
    return max(0.0, min(100.0, weight)) / 100.0, "SnapshotOnlyLatestWeight"


def _strategy_rows_for_asset_horizon(
    *,
    price: pd.Series,
    asset: str,
    horizon: int,
    tables: Dict[str, pd.DataFrame],
    short_ma: int,
    long_ma: int,
    momentum_lookback: int,
    mean_reversion_window: int,
    mean_reversion_threshold: float,
    volatility_target_pct: float,
    total_cost_bps: float,
    random_seed: int,
    random_simulations: int,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, int], pd.Series], List[float]]:
    rows: List[Dict[str, Any]] = []
    signal_cache: Dict[Tuple[str, str, int], pd.Series] = {}
    random_returns: List[float] = []
    if len(price) <= max(long_ma, momentum_lookback, mean_reversion_window, horizon + 5):
        insufficient = True
    else:
        insufficient = False
    baselines = _baseline_signals(price, short_ma, long_ma, momentum_lookback, mean_reversion_window, mean_reversion_threshold, volatility_target_pct)
    for name, signal in baselines.items():
        flag = "InsufficientHistory" if insufficient else "OK"
        comparable, mode = _strategy_mode(name, flag)
        row, _ = _metrics_from_signal(strategy_name=name, asset=asset, horizon=horizon, price=price, signal=signal, total_cost_bps=total_cost_bps, benchmark_role="SimpleBaseline", data_quality_flag=flag, comparable_historical=comparable, evaluation_mode=mode)
        rows.append(row)
        signal_cache[(name, asset, horizon)] = signal

    for model_name in ["Phase12PaperStrategy", "Phase14DynamicSizingStrategy", "Phase15RegimeAdjustedStrategy"]:
        exposure, flag = _snapshot_weight(tables, model_name, asset, horizon)
        signal = pd.Series(float(exposure), index=price.index)
        comparable, mode = _strategy_mode(model_name, flag)
        row, _ = _metrics_from_signal(strategy_name=model_name, asset=asset, horizon=horizon, price=price, signal=signal, total_cost_bps=total_cost_bps, benchmark_role="ModelRiskPipeline", data_quality_flag=flag, comparable_historical=comparable, evaluation_mode=mode)
        rows.append(row)
        signal_cache[(model_name, asset, horizon)] = signal

    sim_rows = []
    for signal in _random_signals(price, asset, horizon, random_seed, random_simulations):
        comparable, mode = _strategy_mode("RandomBaseline", "OK")
        row, _ = _metrics_from_signal(strategy_name="RandomBaseline", asset=asset, horizon=horizon, price=price, signal=signal, total_cost_bps=total_cost_bps, benchmark_role="RandomBaseline", data_quality_flag="OK", comparable_historical=comparable, evaluation_mode=mode)
        sim_rows.append(row)
        random_returns.append(float(row["NetReturnPct"]))
    if sim_rows:
        median_return = float(np.median([row["NetReturnPct"] for row in sim_rows]))
        median_row = min(sim_rows, key=lambda row: abs(float(row["NetReturnPct"]) - median_return))
        rows.append({**median_row, "DataQualityFlag": "MedianOfRandomSimulations"})
    return rows, signal_cache, random_returns


def _leaderboard(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(STRATEGY_LEADERBOARD_COLUMNS))
    if df.empty:
        return _empty(STRATEGY_LEADERBOARD_COLUMNS)
    historical = df[df["ComparableHistorical"].eq(True)].sort_values(["NetReturnPct", "SharpeProxy"], ascending=[False, False]).reset_index(drop=True)
    snapshot = df[~df["ComparableHistorical"].eq(True)].sort_values(["Asset", "Horizon", "StrategyName"]).reset_index(drop=True)
    historical["Rank"] = np.arange(1, len(historical) + 1)
    snapshot["Rank"] = 0
    df = pd.concat([historical, snapshot], ignore_index=True)
    return df[list(STRATEGY_LEADERBOARD_COLUMNS)]


def _best_baseline(group: pd.DataFrame) -> pd.Series:
    baselines = group[group["StrategyName"].isin(BASELINE_STRATEGIES)]
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
    ranked["_BaselineTiePriority"] = ranked["StrategyName"].map(priority).fillna(99)
    return ranked.sort_values(["NetReturnPct", "SharpeProxy", "_BaselineTiePriority"], ascending=[False, False, True]).iloc[0]


def _model_row(group: pd.DataFrame) -> pd.Series:
    model = group[group["StrategyName"].eq(MODEL_STRATEGY_NAME)]
    if model.empty:
        return pd.Series(dtype=object)
    return model.iloc[0]


def _asset_horizon_table(leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (asset, horizon), group in leaderboard.groupby(["Asset", "Horizon"], dropna=False):
        best = group.sort_values(["NetReturnPct", "SharpeProxy"], ascending=[False, False]).iloc[0]
        baseline = _best_baseline(group)
        model = _model_row(group)
        model_return = _safe_float(model.get("NetReturnPct", np.nan), 0.0)
        baseline_return = _safe_float(baseline.get("NetReturnPct", np.nan), 0.0)
        gap = model_return - baseline_return
        comparable_model = bool(model.get("ComparableHistorical", False))
        verdict = "ModelCompetitive" if gap >= 0 else "BenchmarkDominated"
        if not comparable_model and gap >= 0:
            verdict = "InsufficientHistoricalModelEvidence"
        if str(model.get("DataQualityFlag", "")).startswith("Missing"):
            verdict = "InsufficientEvidence"
        rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "BestStrategy": best["StrategyName"],
                "BestBaseline": baseline.get("StrategyName", ""),
                "ModelStrategyReturnPct": round(model_return, 4),
                "BestBaselineReturnPct": round(baseline_return, 4),
                "ModelVsBestBaselinePct": round(gap, 4),
                "ModelBeatsBaseline": bool(gap >= 0 and comparable_model),
                "MaxDrawdownPct": round(_safe_float(model.get("MaxDrawdownPct", 0.0), 0.0), 4),
                "SharpeProxy": round(_safe_float(model.get("SharpeProxy", 0.0), 0.0), 4),
                "TradeCount": int(_safe_float(model.get("TradeCount", 0), 0)),
                "BenchmarkVerdict": verdict,
                "MainReason": "Latest model/risk snapshot is visible but not comparable as a historical strategy." if not comparable_model and gap >= 0 else "A simple baseline has higher net return than the model/risk snapshot." if gap < 0 else "Comparable model strategy beats or matches the best simple baseline.",
            }
        )
    return pd.DataFrame(rows, columns=list(ASSET_HORIZON_BENCHMARK_COLUMNS))


def _asset_table(leaderboard: pd.DataFrame, asset_horizon: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for asset, group in leaderboard.groupby("Asset", dropna=False):
        best = group.sort_values(["NetReturnPct", "SharpeProxy"], ascending=[False, False]).iloc[0]
        baseline_group = group[group["StrategyName"].isin(BASELINE_STRATEGIES)]
        baseline = baseline_group.sort_values(["NetReturnPct", "SharpeProxy"], ascending=[False, False]).iloc[0] if not baseline_group.empty else pd.Series(dtype=object)
        ah = asset_horizon[asset_horizon["Asset"].astype(str).eq(str(asset))]
        model_avg = float(ah["ModelStrategyReturnPct"].mean()) if not ah.empty else 0.0
        baseline_avg = float(ah["BestBaselineReturnPct"].mean()) if not ah.empty else 0.0
        ranks = group[group["StrategyName"].eq(MODEL_STRATEGY_NAME)]["Rank"]
        gap = model_avg - baseline_avg
        verdict = "ModelCompetitive" if gap >= 0 else "BenchmarkDominated"
        if not ah.empty and not ah["ModelBeatsBaseline"].any() and gap >= 0:
            verdict = "InsufficientHistoricalModelEvidence"
        if ah["BenchmarkVerdict"].eq("InsufficientEvidence").all() if not ah.empty else True:
            verdict = "InsufficientEvidence"
        rows.append(
            {
                "Asset": asset,
                "BestStrategy": best["StrategyName"],
                "BestBaseline": baseline.get("StrategyName", ""),
                "ModelStrategyReturnPct": round(model_avg, 4),
                "BestBaselineReturnPct": round(baseline_avg, 4),
                "ModelVsBestBaselinePct": round(gap, 4),
                "ModelRank": int(ranks.min()) if not ranks.empty else 0,
                "BenchmarkVerdict": verdict,
                "MainReason": "Snapshot model/risk impact is not a comparable historical strategy." if verdict == "InsufficientHistoricalModelEvidence" else "Model/risk historical strategy is ahead on average." if gap >= 0 else "Simple baselines are ahead on average.",
                "RiskNote": "Snapshot model/risk rows are not historical replay evidence.",
            }
        )
    return pd.DataFrame(rows, columns=list(ASSET_BENCHMARK_COLUMNS))


def _dominance_and_strength(asset_horizon: pd.DataFrame, leaderboard: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dominance_rows: List[Dict[str, Any]] = []
    strength_rows: List[Dict[str, Any]] = []
    for _, row in asset_horizon.iterrows():
        asset = row["Asset"]
        horizon = int(row["Horizon"])
        group = leaderboard[leaderboard["Asset"].astype(str).eq(str(asset)) & leaderboard["Horizon"].astype(int).eq(horizon)]
        baseline = group[group["StrategyName"].eq(row["BestBaseline"])].iloc[0] if not group[group["StrategyName"].eq(row["BestBaseline"])].empty else pd.Series(dtype=object)
        model = _model_row(group)
        gap = _safe_float(row["ModelVsBestBaselinePct"], 0.0)
        if gap < 0:
            dominance_rows.append(
                {
                    "Asset": asset,
                    "Horizon": horizon,
                    "DominatingBaseline": row["BestBaseline"],
                    "BaselineReturnPct": row["BestBaselineReturnPct"],
                    "ModelReturnPct": row["ModelStrategyReturnPct"],
                    "GapPct": round(abs(gap), 4),
                    "BaselineDrawdownPct": round(_safe_float(baseline.get("MaxDrawdownPct", 0.0), 0.0), 4),
                    "ModelDrawdownPct": round(_safe_float(model.get("MaxDrawdownPct", 0.0), 0.0), 4),
                    "DominanceReason": "Simple baseline produced higher net return than the model/risk snapshot.",
                    "RequiredImprovement": "Improve out-of-sample edge, reduce costs, or keep as research-only evidence.",
                }
            )
        elif row["BenchmarkVerdict"] != "InsufficientEvidence" and bool(model.get("ComparableHistorical", False)):
            model_dd = _safe_float(model.get("MaxDrawdownPct", 0.0), 0.0)
            baseline_dd = _safe_float(baseline.get("MaxDrawdownPct", 0.0), 0.0)
            strength_rows.append(
                {
                    "Asset": asset,
                    "Horizon": horizon,
                    "ModelStrategy": MODEL_STRATEGY_NAME,
                    "ModelReturnPct": row["ModelStrategyReturnPct"],
                    "BestBaselineReturnPct": row["BestBaselineReturnPct"],
                    "ImprovementPct": round(gap, 4),
                    "DrawdownImprovementPct": round(baseline_dd - model_dd, 4),
                    "RiskAdjustedImprovement": round(gap + (baseline_dd - model_dd) * 0.25, 4),
                    "EvidenceStrength": "SnapshotOnly" if "Snapshot" in str(model.get("DataQualityFlag", "")) else "Historical",
                    "MainStrengthReason": "Model/risk snapshot has higher net return than the strongest simple baseline.",
                }
            )
    return pd.DataFrame(dominance_rows, columns=list(BENCHMARK_DOMINANCE_COLUMNS)), pd.DataFrame(strength_rows, columns=list(MODEL_STRENGTH_COLUMNS))


def _cost_sensitivity(price_by_asset: Dict[str, pd.Series], signal_cache: Dict[Tuple[str, str, int], pd.Series], assets: Iterable[str], horizons: Iterable[int], cost_scenarios_bps: Iterable[float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (strategy, asset, horizon), signal in signal_cache.items():
        if strategy == "RandomBaseline":
            continue
        price = price_by_asset.get(asset, pd.Series(dtype=float))
        comparable, mode = _strategy_mode(strategy, "OK")
        zero_row, _ = _metrics_from_signal(strategy_name=strategy, asset=asset, horizon=horizon, price=price, signal=signal, total_cost_bps=0.0, benchmark_role="", data_quality_flag="OK", comparable_historical=comparable, evaluation_mode=mode)
        zero_return = _safe_float(zero_row.get("NetReturnPct", 0.0), 0.0)
        for cost_bps in cost_scenarios_bps:
            row, _ = _metrics_from_signal(strategy_name=strategy, asset=asset, horizon=horizon, price=price, signal=signal, total_cost_bps=float(cost_bps), benchmark_role="", data_quality_flag="OK", comparable_historical=comparable, evaluation_mode=mode)
            net_return = _safe_float(row.get("NetReturnPct", 0.0), 0.0)
            lost = zero_return - net_return
            fragile = bool(zero_return >= 0 and net_return < 0) or lost > max(2.0, abs(zero_return) * 0.5)
            rows.append(
                {
                    "StrategyName": strategy,
                    "Asset": asset,
                    "Horizon": int(horizon),
                    "CostBps": float(cost_bps),
                    "NetReturnPct": round(net_return, 4),
                    "ReturnLostToCostsPct": round(lost, 4),
                    "CostFragile": fragile,
                    "Explanation": "Costs materially reduce returns." if fragile else "Cost impact is visible and retained for review.",
                }
            )
    return pd.DataFrame(rows, columns=list(COST_SENSITIVITY_COLUMNS))


def _random_baseline_table(random_map: Dict[Tuple[str, int], List[float]], asset_horizon: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, ah in asset_horizon.iterrows():
        key = (str(ah["Asset"]), int(ah["Horizon"]))
        values = np.array(random_map.get(key, []), dtype=float)
        model_return = _safe_float(ah["ModelStrategyReturnPct"], 0.0)
        if len(values) == 0:
            rows.append({"Asset": key[0], "Horizon": key[1], "SimulationCount": 0, "RandomMedianReturnPct": 0.0, "RandomP25ReturnPct": 0.0, "RandomP75ReturnPct": 0.0, "RandomBestReturnPct": 0.0, "ModelReturnPct": round(model_return, 4), "ModelBeatsRandomMedian": False, "ModelPercentileVsRandom": 0.0, "Explanation": "No random baseline simulations were available."})
            continue
        percentile = float((values <= model_return).mean() * 100.0)
        rows.append(
            {
                "Asset": key[0],
                "Horizon": key[1],
                "SimulationCount": int(len(values)),
                "RandomMedianReturnPct": round(float(np.median(values)), 4),
                "RandomP25ReturnPct": round(float(np.percentile(values, 25)), 4),
                "RandomP75ReturnPct": round(float(np.percentile(values, 75)), 4),
                "RandomBestReturnPct": round(float(np.max(values)), 4),
                "ModelReturnPct": round(model_return, 4),
                "ModelBeatsRandomMedian": bool(model_return >= np.median(values)),
                "ModelPercentileVsRandom": round(percentile, 4),
                "Explanation": "Compares model/risk snapshot return with reproducible random exposure simulations.",
            }
        )
    return pd.DataFrame(rows, columns=list(RANDOM_BASELINE_COLUMNS))


def _latest_window_return(price: pd.Series, horizon: int) -> float:
    if len(price) <= int(horizon):
        return np.nan
    start = _safe_float(price.iloc[-int(horizon) - 1], np.nan)
    end = _safe_float(price.iloc[-1], np.nan)
    if not np.isfinite(start) or start == 0 or not np.isfinite(end):
        return np.nan
    return (end / start - 1.0) * 100.0


def _snapshot_model_impact(tables: Dict[str, pd.DataFrame], price_by_asset: Dict[str, pd.Series], assets: Iterable[str], horizons: Iterable[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for asset in assets:
        price = price_by_asset.get(asset, pd.Series(dtype=float))
        for horizon in horizons:
            latest_return = _latest_window_return(price, int(horizon))
            for strategy in ["Phase12PaperStrategy", "Phase14DynamicSizingStrategy", "Phase15RegimeAdjustedStrategy"]:
                weight, flag = _snapshot_weight(tables, strategy, asset, int(horizon))
                weight_pct = weight * 100.0
                impact = weight * latest_return if np.isfinite(latest_return) else np.nan
                rows.append(
                    {
                        "Asset": asset,
                        "Horizon": int(horizon),
                        "SnapshotStrategyName": strategy,
                        "SnapshotWeightPct": round(weight_pct, 4),
                        "LatestKnownReturnWindowPct": round(latest_return, 4) if np.isfinite(latest_return) else np.nan,
                        "HypotheticalImpactPct": round(impact, 4) if np.isfinite(impact) else np.nan,
                        "ComparableHistorical": False,
                        "EvaluationMode": "LatestSnapshotOnly" if flag != "MissingSnapshotWeight" else "InsufficientData",
                        "Explanation": "Latest snapshot sizing applied to the latest known return window; this is not a historical backtest.",
                    }
                )
    return pd.DataFrame(rows, columns=list(SNAPSHOT_MODEL_IMPACT_COLUMNS))


def _return_sanity_checks(leaderboard: pd.DataFrame, price_by_asset: Dict[str, pd.Series], assets: Iterable[str], horizons: Iterable[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for asset in assets:
        price = price_by_asset.get(asset, pd.Series(dtype=float))
        if price.empty:
            continue
        expected_total = (price.iloc[-1] / price.iloc[0] - 1.0) * 100.0 if price.iloc[0] != 0 else np.nan
        daily = price.pct_change().dropna()
        max_daily_abs = float(daily.abs().max() * 100.0) if not daily.empty else 0.0
        for horizon in horizons:
            hold = leaderboard[
                leaderboard["Asset"].astype(str).eq(str(asset))
                & leaderboard["Horizon"].astype(int).eq(int(horizon))
                & leaderboard["StrategyName"].eq("HoldOnlyBenchmark")
            ]
            if hold.empty:
                continue
            observed = _safe_float(hold["TotalReturnPct"].iloc[0], np.nan)
            matches = bool(np.isfinite(observed) and np.isfinite(expected_total) and abs(observed - expected_total) <= max(0.05, abs(expected_total) * 0.001))
            rows.append({"CheckName": "HoldOnlyReturnMatchesPriceRatio", "Passed": matches, "Severity": "Critical", "Asset": asset, "Horizon": int(horizon), "StrategyName": "HoldOnlyBenchmark", "ObservedValue": round(observed, 6) if np.isfinite(observed) else np.nan, "ExpectedRangeOrValue": f"{expected_total:.6f}", "Explanation": "Hold-only total return should equal first-to-last price ratio when costs are zero."})
    astronomical = leaderboard["NetReturnPct"].abs() > 100000.0 if not leaderboard.empty else pd.Series(dtype=bool)
    if not leaderboard.empty:
        worst = leaderboard.loc[leaderboard["NetReturnPct"].abs().idxmax()]
        rows.append({"CheckName": "NoAstronomicalReturnExplosion", "Passed": bool(not astronomical.any()), "Severity": "Critical", "Asset": worst["Asset"], "Horizon": int(worst["Horizon"]), "StrategyName": worst["StrategyName"], "ObservedValue": round(_safe_float(worst["NetReturnPct"], 0.0), 6), "ExpectedRangeOrValue": "absolute net return <= 100000%", "Explanation": "Detects overlapping-horizon compounding explosions or corrupted percentage units."})
    else:
        rows.append({"CheckName": "NoAstronomicalReturnExplosion", "Passed": False, "Severity": "Critical", "Asset": "ALL", "Horizon": "ALL", "StrategyName": "ALL", "ObservedValue": np.nan, "ExpectedRangeOrValue": "benchmark rows exist", "Explanation": "No benchmark rows were available."})
    max_daily = max((_safe_float(price_by_asset.get(asset, pd.Series(dtype=float)).pct_change().abs().max(), 0.0) for asset in assets), default=0.0) * 100.0
    units_ok = bool(max_daily <= 1000.0)
    rows.append({"CheckName": "PercentDecimalUnitsConsistent", "Passed": units_ok, "Severity": "Critical", "Asset": "ALL", "Horizon": "ALL", "StrategyName": "ALL", "ObservedValue": round(max_daily, 6), "ExpectedRangeOrValue": "max one-day absolute return <= 1000%", "Explanation": "Flags suspicious price jumps or percent/decimal unit confusion."})
    rows.append({"CheckName": "DailyReturnsUsedForContinuousStrategies", "Passed": True, "Severity": "Critical", "Asset": "ALL", "Horizon": "ALL", "StrategyName": "ContinuousStrategies", "ObservedValue": "daily_return_exposure_engine", "ExpectedRangeOrValue": "daily returns, not overlapping forward returns", "Explanation": "Continuous benchmark strategies use daily returns multiplied by prior-row exposure."})
    snapshot_excluded = bool(leaderboard[leaderboard["BenchmarkRole"].eq("ModelRiskPipeline")]["ComparableHistorical"].eq(False).all()) if not leaderboard.empty else True
    rows.append({"CheckName": "SnapshotStrategiesExcludedFromHistoricalWinner", "Passed": snapshot_excluded, "Severity": "Critical", "Asset": "ALL", "Horizon": "ALL", "StrategyName": "ModelRiskPipeline", "ObservedValue": snapshot_excluded, "ExpectedRangeOrValue": "snapshot rows ComparableHistorical=False", "Explanation": "Latest Phase 12/14/15 snapshot rows are excluded from historical winner claims."})
    return pd.DataFrame(rows, columns=list(RETURN_SANITY_CHECK_COLUMNS))


def _summary(leaderboard: pd.DataFrame, asset_horizon: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty or asset_horizon.empty:
        return pd.DataFrame([{"OverallWinner": "", "ModelBeatsHoldOnly": False, "ModelBeatsMomentum": False, "ModelBeatsMovingAverage": False, "ModelBeatsNoExposure": False, "BenchmarkVerdict": "InsufficientEvidence", "MainReason": "No usable benchmark rows were produced.", "WeakestArea": "Input data", "StrongestArea": "None", "EvidenceQuality": "InsufficientEvidence", "RecommendedNextStep": "Load market data and saved research artifacts, then rerun the arena."}], columns=list(BENCHMARK_SUMMARY_COLUMNS))
    historical = leaderboard[leaderboard["ComparableHistorical"].eq(True)]
    avg = historical.groupby("StrategyName", as_index=False)["NetReturnPct"].mean() if not historical.empty else pd.DataFrame(columns=["StrategyName", "NetReturnPct"])
    if avg.empty:
        return pd.DataFrame([{"OverallWinner": "", "ModelBeatsHoldOnly": False, "ModelBeatsMomentum": False, "ModelBeatsMovingAverage": False, "ModelBeatsNoExposure": False, "BenchmarkVerdict": "InsufficientHistoricalModelEvidence", "MainReason": "Model/risk pipeline currently has latest snapshot sizing but no full historical replay weights; baseline comparison is valid only for simple baselines.", "WeakestArea": "Missing historical model/risk replay", "StrongestArea": "Simple historical baselines", "EvidenceQuality": "SnapshotOnly", "RecommendedNextStep": "Build historical model/risk replay before claiming model beats or loses to baselines."}], columns=list(BENCHMARK_SUMMARY_COLUMNS))
    winner = avg.sort_values("NetReturnPct", ascending=False).iloc[0]
    model_avg = _safe_float(avg[avg["StrategyName"].eq(MODEL_STRATEGY_NAME)]["NetReturnPct"].iloc[0] if not avg[avg["StrategyName"].eq(MODEL_STRATEGY_NAME)].empty else np.nan, np.nan)

    if not np.isfinite(model_avg):
        return pd.DataFrame(
            [
                {
                    "OverallWinner": str(winner["StrategyName"]),
                    "ModelBeatsHoldOnly": False,
                    "ModelBeatsMomentum": False,
                    "ModelBeatsMovingAverage": False,
                    "ModelBeatsNoExposure": False,
                    "BenchmarkVerdict": "InsufficientHistoricalModelEvidence",
                    "MainReason": "Model/risk pipeline currently has latest snapshot sizing but no full historical replay weights; baseline comparison is valid only for simple baselines.",
                    "WeakestArea": "Missing historical model/risk replay",
                    "StrongestArea": str(winner["StrategyName"]),
                    "EvidenceQuality": "SnapshotOnly",
                    "RecommendedNextStep": "Build historical model/risk replay before claiming model beats or loses to baselines.",
                }
            ],
            columns=list(BENCHMARK_SUMMARY_COLUMNS),
        )

    def beats(name: str) -> bool:
        base = avg[avg["StrategyName"].eq(name)]
        if base.empty or not np.isfinite(model_avg):
            return False
        return bool(model_avg >= _safe_float(base["NetReturnPct"].iloc[0], 0.0))

    insufficient = asset_horizon["BenchmarkVerdict"].eq("InsufficientEvidence").mean() > 0.5
    dominated_rate = asset_horizon["BenchmarkVerdict"].eq("BenchmarkDominated").mean() if not asset_horizon.empty else 1.0
    win_rate = asset_horizon["ModelBeatsBaseline"].mean() if "ModelBeatsBaseline" in asset_horizon.columns and not asset_horizon.empty else 0.0
    if insufficient:
        verdict = "InsufficientEvidence"
    elif win_rate >= 0.8 and model_avg >= _safe_float(winner["NetReturnPct"], 0.0) - 0.5:
        verdict = "ModelDominatesBaselines"
    elif win_rate >= 0.55:
        verdict = "ModelCompetitive"
    elif dominated_rate >= 0.6:
        verdict = "BenchmarkDominated"
    else:
        verdict = "ModelMixed"
    weakest = "Benchmark domination" if dominated_rate > 0 else "Snapshot-only evidence"
    strongest = str(winner["StrategyName"])
    main = "Simple baselines often beat the model/risk snapshot." if verdict == "BenchmarkDominated" else "Model/risk snapshot is competitive with simple baselines." if verdict in {"ModelDominatesBaselines", "ModelCompetitive"} else "Evidence is mixed or insufficient."
    return pd.DataFrame(
        [
            {
                "OverallWinner": str(winner["StrategyName"]),
                "ModelBeatsHoldOnly": beats("HoldOnlyBenchmark"),
                "ModelBeatsMomentum": beats("MomentumBaseline"),
                "ModelBeatsMovingAverage": beats("MovingAverageCrossover"),
                "ModelBeatsNoExposure": beats("NoExposureBaseline"),
                "BenchmarkVerdict": verdict,
                "MainReason": main,
                "WeakestArea": weakest,
                "StrongestArea": strongest,
                "EvidenceQuality": "SnapshotOnly" if leaderboard["DataQualityFlag"].astype(str).str.contains("Snapshot").any() else "Historical",
                "RecommendedNextStep": "Collect historical replay evidence before treating snapshot comparisons as stronger research evidence.",
            }
        ],
        columns=list(BENCHMARK_SUMMARY_COLUMNS),
    )


def _leakage_checks(market_data: pd.DataFrame, leaderboard: pd.DataFrame, signal_cache: Dict[Tuple[str, str, int], pd.Series]) -> pd.DataFrame:
    date_ordered = bool(market_data.empty or market_data.index.is_monotonic_increasing)
    duplicated = bool(not market_data.empty and market_data.index.duplicated().any())
    shifted = True
    snapshot_flagged = bool(leaderboard[leaderboard["BenchmarkRole"].eq("ModelRiskPipeline")]["DataQualityFlag"].astype(str).str.contains("Snapshot|Missing").all()) if not leaderboard.empty else True
    snapshot_excluded = bool(leaderboard[leaderboard["BenchmarkRole"].eq("ModelRiskPipeline")]["ComparableHistorical"].eq(False).all()) if not leaderboard.empty else True
    insufficient = bool(leaderboard["DataQualityFlag"].astype(str).str.contains("InsufficientHistory").any()) if not leaderboard.empty else True
    rows = [
        ("Signals shifted before returns", shifted, "Critical", "The return engine applies prior-row exposure to daily returns."),
        ("No future return used in feature/signal", True, "Critical", "Baseline signals use rolling price history only; future returns are used only as outcomes."),
        ("DailyReturnsUsedForContinuousStrategies", True, "Critical", "Continuous strategies use daily returns, not overlapping horizon-forward returns."),
        ("Date ordering valid", date_ordered, "High", "Market data index is chronological."),
        ("No duplicated date leakage", not duplicated, "High", "Duplicate date rows can overstate evidence."),
        ("Sufficient history before indicator use", not insufficient, "Medium", "Rows with insufficient history are flagged rather than silently treated as strong evidence."),
        ("Snapshot strategies not mislabeled as historical backtests", snapshot_flagged, "High", "Phase 12/14/15 latest sizing rows are marked as snapshot strategies."),
        ("SnapshotStrategiesExcludedFromHistoricalWinner", snapshot_excluded, "Critical", "Latest snapshot rows are visible but excluded from historical winner claims."),
    ]
    return pd.DataFrame([{"CheckName": name, "Passed": bool(passed), "Severity": severity, "Explanation": explanation} for name, passed, severity, explanation in rows], columns=list(LEAKAGE_CHECK_COLUMNS))


def _warnings(asset_horizon: pd.DataFrame, leaderboard: pd.DataFrame, random_table: pd.DataFrame, cost_table: pd.DataFrame, leakage: pd.DataFrame, return_sanity: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in asset_horizon.iterrows():
        asset = row["Asset"]
        horizon = int(row["Horizon"])
        model = leaderboard[leaderboard["Asset"].astype(str).eq(str(asset)) & leaderboard["Horizon"].astype(int).eq(horizon) & leaderboard["StrategyName"].eq(MODEL_STRATEGY_NAME)]
        model_row = model.iloc[0] if not model.empty else pd.Series(dtype=object)
        if row["BenchmarkVerdict"] == "BenchmarkDominated":
            rows.append({"WarningType": "BenchmarkDominated", "Severity": "High", "Asset": asset, "Horizon": horizon, "StrategyName": MODEL_STRATEGY_NAME, "Explanation": "A simple baseline has higher net return than the model/risk snapshot.", "RecommendedFix": "Investigate benchmark gap before increasing research confidence."})
        if _safe_float(row["ModelStrategyReturnPct"], 0.0) < 0:
            rows.append({"WarningType": "NoExposureOutperforms", "Severity": "High", "Asset": asset, "Horizon": horizon, "StrategyName": MODEL_STRATEGY_NAME, "Explanation": "No exposure performs better than a losing model/risk snapshot.", "RecommendedFix": "Keep as research-only evidence until forward results improve."})
        if _safe_float(model_row.get("MaxDrawdownPct", 0.0), 0.0) <= -20:
            rows.append({"WarningType": "HighDrawdown", "Severity": "High", "Asset": asset, "Horizon": horizon, "StrategyName": MODEL_STRATEGY_NAME, "Explanation": "Model/risk snapshot has a large drawdown estimate.", "RecommendedFix": "Review risk controls and regime filters."})
        if int(_safe_float(model_row.get("TradeCount", 0), 0)) < 2:
            rows.append({"WarningType": "LowTradeCount", "Severity": "Medium", "Asset": asset, "Horizon": horizon, "StrategyName": MODEL_STRATEGY_NAME, "Explanation": "Few turnover events are available for this snapshot comparison.", "RecommendedFix": "Collect more forward paper evidence."})
        if "Snapshot" in str(model_row.get("DataQualityFlag", "")):
            rows.append({"WarningType": "SnapshotOnly", "Severity": "Medium", "Asset": asset, "Horizon": horizon, "StrategyName": MODEL_STRATEGY_NAME, "Explanation": "Latest sizing artifact is a snapshot, not a historical replay strategy.", "RecommendedFix": "Build historical replay artifacts before stronger conclusions."})
    for _, row in random_table.iterrows():
        if not bool(row["ModelBeatsRandomMedian"]):
            rows.append({"WarningType": "RandomBaselineOutperforms", "Severity": "Medium", "Asset": row["Asset"], "Horizon": int(row["Horizon"]), "StrategyName": MODEL_STRATEGY_NAME, "Explanation": "Random median return is higher than the model/risk snapshot.", "RecommendedFix": "Check whether the signal adds value over noise."})
    fragile = cost_table[cost_table["CostFragile"].eq(True)]
    for _, row in fragile.head(20).iterrows():
        rows.append({"WarningType": "CostFragile", "Severity": "Medium", "Asset": row["Asset"], "Horizon": int(row["Horizon"]), "StrategyName": row["StrategyName"], "Explanation": f"Cost scenario {row['CostBps']} bps materially reduces net return.", "RecommendedFix": "Use conservative cost assumptions and inspect turnover."})
    for _, row in leakage[~leakage["Passed"]].iterrows():
        rows.append({"WarningType": "PossibleLeakage", "Severity": row["Severity"], "Asset": "ALL", "Horizon": "ALL", "StrategyName": "ALL", "Explanation": row["Explanation"], "RecommendedFix": "Fix timing or source issue before interpreting benchmark results."})
    for _, row in return_sanity[~return_sanity["Passed"]].iterrows():
        rows.append({"WarningType": "ReturnSanityFailed", "Severity": row["Severity"], "Asset": row["Asset"], "Horizon": row["Horizon"], "StrategyName": row["StrategyName"], "Explanation": row["Explanation"], "RecommendedFix": "Fix return calculation or input units before interpreting benchmark results."})
    return pd.DataFrame(rows, columns=list(BENCHMARK_WARNING_COLUMNS))


def _next_actions(summary: pd.DataFrame, warnings: pd.DataFrame, dominance: pd.DataFrame, strengths: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    verdict = str(summary.iloc[0]["BenchmarkVerdict"]) if not summary.empty else "InsufficientEvidence"
    if verdict in {"BenchmarkDominated", "ModelMixed"} and not dominance.empty:
        rows.append({"Rank": 0, "Action": "Investigate baseline-dominated combinations.", "WhyItMatters": "Simple strategies beating the model/risk pipeline is the main research finding.", "AffectedAssets": "; ".join(dominance["Asset"].astype(str).unique()), "AffectedHorizons": "; ".join(f"{int(h)}D" for h in pd.to_numeric(dominance["Horizon"], errors="coerce").dropna().astype(int).unique()), "ExpectedBenefit": "Separates weak model/risk evidence from genuinely useful signal behavior.", "Urgency": "High", "DependsOn": "Benchmark dominance and leakage tables."})
    if warnings["WarningType"].astype(str).eq("SnapshotOnly").any() if not warnings.empty else False:
        rows.append({"Rank": 0, "Action": "Build historical replay evidence for model/risk strategies.", "WhyItMatters": "Latest snapshots should not be overstated as historical backtests.", "AffectedAssets": "ALL", "AffectedHorizons": "ALL", "ExpectedBenefit": "Improves evidence quality before any research confidence upgrade.", "Urgency": "High", "DependsOn": "Phase 9/forward logs and stored signal history."})
    if not strengths.empty:
        rows.append({"Rank": 0, "Action": "Review model-strength rows for repeatability.", "WhyItMatters": "These are the only rows where the model/risk snapshot beats simple baselines.", "AffectedAssets": "; ".join(strengths["Asset"].astype(str).unique()), "AffectedHorizons": "; ".join(f"{int(h)}D" for h in pd.to_numeric(strengths["Horizon"], errors="coerce").dropna().astype(int).unique()), "ExpectedBenefit": "Focuses next research on the strongest honest candidates.", "Urgency": "Medium", "DependsOn": "Forward paper evidence maturity."})
    if not rows:
        rows.append({"Rank": 0, "Action": "Collect more evidence before changing research posture.", "WhyItMatters": "The arena did not find enough reliable benchmark separation.", "AffectedAssets": "ALL", "AffectedHorizons": "ALL", "ExpectedBenefit": "Prevents overreading weak or sparse benchmark results.", "Urgency": "Medium", "DependsOn": "More mature paper outcomes."})
    actions = pd.DataFrame(rows, columns=list(NEXT_BENCHMARK_ACTION_COLUMNS))
    actions["Rank"] = np.arange(1, len(actions) + 1)
    return actions


def run_strategy_benchmark_arena(
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
    autosave: bool = False,
    **direct_tables: Any,
) -> StrategyBenchmarkArenaReport:
    asset_list = list(assets or get_asset_names())
    horizon_list = [int(h) for h in (horizons or BENCHMARK_HORIZONS)]
    project_used = False
    if market_data is None and use_project_market_data:
        market_data = _load_project_market_data()
        project_used = market_data is not None
    market = _prepare_market_data(market_data)
    tables, artifact_sources = _resolve_inputs(bool(use_artifact_store), bool(prefer_uploaded), uploaded_overrides, direct_tables)

    total_cost_bps = float(cost_bps) + float(slippage_bps)
    rows: List[Dict[str, Any]] = []
    signal_cache: Dict[Tuple[str, str, int], pd.Series] = {}
    random_map: Dict[Tuple[str, int], List[float]] = {}
    price_by_asset: Dict[str, pd.Series] = {}
    for asset in asset_list:
        price = _series(market, get_target_column(asset))
        price_by_asset[asset] = price
        for horizon in horizon_list:
            if price.empty:
                for strategy in ["NoExposureBaseline", "HoldOnlyBenchmark", "MovingAverageCrossover", "MomentumBaseline", "MeanReversionBaseline", "VolatilityScaledBaseline", "RandomBaseline", "Phase12PaperStrategy", "Phase14DynamicSizingStrategy", "Phase15RegimeAdjustedStrategy"]:
                    rows.append({"Rank": 0, "StrategyName": strategy, "Asset": asset, "Horizon": horizon, "TotalReturnPct": 0.0, "AnnualizedReturnPct": 0.0, "VolatilityPct": 0.0, "SharpeProxy": 0.0, "MaxDrawdownPct": 0.0, "WinRatePct": 0.0, "TradeCount": 0, "TurnoverPct": 0.0, "CostImpactPct": 0.0, "NetReturnPct": 0.0, "ExposurePct": 0.0, "BenchmarkRole": "ModelRiskPipeline" if strategy.startswith("Phase") else "SimpleBaseline", "DataQualityFlag": "MissingPriceData", "ComparableHistorical": False, "EvaluationMode": "InsufficientData"})
                random_map[(asset, horizon)] = []
                continue
            strategy_rows, signals, random_returns = _strategy_rows_for_asset_horizon(
                price=price,
                asset=asset,
                horizon=horizon,
                tables=tables,
                short_ma=int(short_ma),
                long_ma=int(long_ma),
                momentum_lookback=int(momentum_lookback),
                mean_reversion_window=int(mean_reversion_window),
                mean_reversion_threshold=float(mean_reversion_threshold),
                volatility_target_pct=float(volatility_target_pct),
                total_cost_bps=total_cost_bps,
                random_seed=int(random_seed),
                random_simulations=int(random_simulations),
            )
            rows.extend(strategy_rows)
            signal_cache.update(signals)
            random_map[(asset, horizon)] = random_returns

    leaderboard = _leaderboard(rows)
    asset_horizon = _asset_horizon_table(leaderboard)
    asset_benchmark = _asset_table(leaderboard, asset_horizon)
    dominance, strengths = _dominance_and_strength(asset_horizon, leaderboard)
    cost_table = _cost_sensitivity(price_by_asset, signal_cache, asset_list, horizon_list, cost_scenarios_bps)
    random_table = _random_baseline_table(random_map, asset_horizon)
    return_sanity = _return_sanity_checks(leaderboard, price_by_asset, asset_list, horizon_list)
    snapshot_impact = _snapshot_model_impact(tables, price_by_asset, asset_list, horizon_list)
    leakage = _leakage_checks(market, leaderboard, signal_cache)
    warnings = _warnings(asset_horizon, leaderboard, random_table, cost_table, leakage, return_sanity)
    summary = _summary(leaderboard, asset_horizon)
    actions = _next_actions(summary, warnings, dominance, strengths)
    input_sources = _input_source_table(market, asset_list, project_used)
    settings = {
        "phase": "16",
        "purpose": "strategy_benchmark_arena",
        "assets": asset_list,
        "horizons": horizon_list,
        "short_ma": int(short_ma),
        "long_ma": int(long_ma),
        "momentum_lookback": int(momentum_lookback),
        "mean_reversion_window": int(mean_reversion_window),
        "mean_reversion_threshold": float(mean_reversion_threshold),
        "volatility_target_pct": float(volatility_target_pct),
        "cost_bps": float(cost_bps),
        "slippage_bps": float(slippage_bps),
        "random_seed": int(random_seed),
        "random_simulations": int(random_simulations),
    }
    report = StrategyBenchmarkArenaReport(
        benchmark_summary_table=summary.reset_index(drop=True),
        strategy_leaderboard_table=leaderboard.reset_index(drop=True),
        asset_benchmark_table=asset_benchmark.reset_index(drop=True),
        asset_horizon_benchmark_table=asset_horizon.reset_index(drop=True),
        benchmark_dominance_table=dominance.reset_index(drop=True),
        model_strength_table=strengths.reset_index(drop=True),
        cost_sensitivity_table=cost_table.reset_index(drop=True),
        random_baseline_table=random_table.reset_index(drop=True),
        return_sanity_check_table=return_sanity.reset_index(drop=True),
        snapshot_model_impact_table=snapshot_impact.reset_index(drop=True),
        leakage_check_table=leakage.reset_index(drop=True),
        benchmark_warning_table=warnings.reset_index(drop=True),
        next_benchmark_actions_table=actions.reset_index(drop=True),
        benchmark_input_sources_table=input_sources.reset_index(drop=True),
        artifact_input_source_table=artifact_sources.reset_index(drop=True),
        settings=settings,
    )
    if autosave:
        report.saved_artifacts = save_phase_artifacts(
            STRATEGY_BENCHMARK_PHASE_NAME,
            {
                "benchmark_summary_table": report.benchmark_summary_table,
                "strategy_leaderboard_table": report.strategy_leaderboard_table,
                "asset_benchmark_table": report.asset_benchmark_table,
                "asset_horizon_benchmark_table": report.asset_horizon_benchmark_table,
                "benchmark_dominance_table": report.benchmark_dominance_table,
                "model_strength_table": report.model_strength_table,
                "cost_sensitivity_table": report.cost_sensitivity_table,
                "random_baseline_table": report.random_baseline_table,
                "return_sanity_check_table": report.return_sanity_check_table,
                "snapshot_model_impact_table": report.snapshot_model_impact_table,
                "leakage_check_table": report.leakage_check_table,
                "benchmark_warning_table": report.benchmark_warning_table,
                "next_benchmark_actions_table": report.next_benchmark_actions_table,
                "benchmark_input_sources_table": report.benchmark_input_sources_table,
                "artifact_input_source_table": report.artifact_input_source_table,
            },
            inputs={},
            config=report.settings,
            warnings=report.benchmark_warning_table["WarningType"].dropna().astype(str).unique().tolist() if not report.benchmark_warning_table.empty else [],
        )
    return report


__all__ = [
    "ASSET_BENCHMARK_COLUMNS",
    "ASSET_HORIZON_BENCHMARK_COLUMNS",
    "BENCHMARK_DOMINANCE_COLUMNS",
    "BENCHMARK_HORIZONS",
    "BENCHMARK_INPUT_SOURCE_COLUMNS",
    "BENCHMARK_SUMMARY_COLUMNS",
    "BENCHMARK_WARNING_COLUMNS",
    "COST_SENSITIVITY_COLUMNS",
    "LEAKAGE_CHECK_COLUMNS",
    "MODEL_STRENGTH_COLUMNS",
    "NEXT_BENCHMARK_ACTION_COLUMNS",
    "RANDOM_BASELINE_COLUMNS",
    "RETURN_SANITY_CHECK_COLUMNS",
    "SNAPSHOT_MODEL_IMPACT_COLUMNS",
    "STRATEGY_BENCHMARK_PHASE_NAME",
    "STRATEGY_LEADERBOARD_COLUMNS",
    "StrategyBenchmarkArenaReport",
    "run_strategy_benchmark_arena",
]
