# src/signal_engine.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
VALID_SIGNAL_MODES = {"long_only", "long_short", "avoid_only", "no_trade_zone"}
VALID_BACKTEST_STYLES = {"overlapping_research", "non_overlapping_realistic", "both"}


@dataclass
class SignalBacktestResult:
    metrics: Dict[str, Any]
    signal_frame: pd.DataFrame
    threshold_sweep: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_sweep: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_signal_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    locked_test_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_test_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    selected_threshold: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalResearchScanReport:
    full_results: pd.DataFrame
    verdict_counts: Dict[str, int]
    top_robust_candidates: pd.DataFrame
    failed_candidates: pd.DataFrame
    errors: pd.DataFrame
    settings: Dict[str, Any]
    candidate_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    signal_outputs: Dict[tuple, Any] = field(default_factory=dict)


@dataclass
class CandidateDiagnosticsReport:
    candidate_summary: pd.DataFrame
    trade_log: pd.DataFrame
    trade_diagnostics: pd.DataFrame
    monthly_returns: pd.DataFrame
    quarterly_returns: pd.DataFrame
    equity_curve: pd.DataFrame
    drawdown_curve: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    validation_split_sensitivity: pd.DataFrame
    probability_diagnostics: pd.DataFrame
    probability_bins: pd.DataFrame
    warnings: List[str]
    base_result: SignalBacktestResult
    scan_report: SignalResearchScanReport
    settings: Dict[str, Any]


@dataclass
class RiskControlUpgradeReport:
    baseline_vs_best: pd.DataFrame
    full_variant_table: pd.DataFrame
    cost_stress_table: pd.DataFrame
    warnings: List[str]
    selected_variant: Dict[str, Any]
    baseline_result: SignalBacktestResult
    selected_result: SignalBacktestResult
    settings: Dict[str, Any]


@dataclass
class WalkForwardValidationReport:
    aggregate_summary: pd.DataFrame
    window_results: pd.DataFrame
    verdict_counts: Dict[str, int]
    warnings: List[str]
    errors: pd.DataFrame
    settings: Dict[str, Any]


SIGNAL_SCAN_COLUMNS = [
    "Asset",
    "Horizon",
    "ModelDepth",
    "Phase5Enabled",
    "SignalMode",
    "BestValidationThreshold",
    "CooldownRows",
    "ValidationScore",
    "ValidationTrades",
    "ValidationWinRate_%",
    "ValidationStrategyReturn_%",
    "ValidationBuyHoldReturn_%",
    "ValidationVsBuyHold_%",
    "LockedTestTrades",
    "LockedTestWinRate_%",
    "LockedTestStrategyReturn_%",
    "LockedTestBuyHoldReturn_%",
    "LockedTestVsBuyHold_%",
    "LockedTestMaxDrawdown_%",
    "LockedTestSharpe",
    "Exposure_%",
    "RobustnessVerdict",
    "StabilityFlag",
    "FailureReason",
]


WALK_FORWARD_WINDOW_COLUMNS = [
    "WindowId",
    "Asset",
    "Horizon",
    "ValidationStart",
    "ValidationEnd",
    "LockedTestStart",
    "LockedTestEnd",
    "SelectedThreshold",
    "SelectedCooldown",
    "ValidationTrades",
    "ValidationStrategyReturn_%",
    "ValidationBuyHoldReturn_%",
    "ValidationVsBuyHold_%",
    "ValidationMaxDrawdown_%",
    "LockedTrades",
    "LockedWinRate_%",
    "LockedStrategyReturn_%",
    "LockedBuyHoldReturn_%",
    "LockedVsBuyHold_%",
    "LockedMaxDrawdown_%",
    "LockedSharpe",
    "LockedExposure_%",
    "BeatBuyHold",
    "PositiveStrategyReturn",
    "WindowVerdict",
    "FailureReason",
]


WALK_FORWARD_AGG_COLUMNS = [
    "Asset",
    "Horizon",
    "NumberOfWindows",
    "WindowsBeatingBuyHold",
    "BeatBuyHoldRate_%",
    "PositiveReturnWindows",
    "PositiveReturnRate_%",
    "AvgLockedStrategyReturn_%",
    "MedianLockedStrategyReturn_%",
    "AvgLockedVsBuyHold_%",
    "MedianLockedVsBuyHold_%",
    "WorstLockedVsBuyHold_%",
    "BestLockedVsBuyHold_%",
    "AvgLockedMaxDrawdown_%",
    "WorstLockedMaxDrawdown_%",
    "AvgLockedSharpe",
    "MedianLockedSharpe",
    "AvgTradesPerWindow",
    "LowTradeWindowCount",
    "ThresholdStability",
    "CooldownStability",
    "WalkForwardStabilityScore",
    "WalkForwardVerdict",
    "FailureReason",
]


RISK_CONTROL_COLUMNS = [
    "Asset",
    "Horizon",
    "ModelDepth",
    "Phase5Enabled",
    "SignalMode",
    "BaseThreshold",
    "BaseCooldown",
    "RiskVariantName",
    "RiskVariantParams",
    "ValidationStrategyReturn_%",
    "ValidationBuyHoldReturn_%",
    "ValidationVsBuyHold_%",
    "ValidationMaxDrawdown_%",
    "ValidationSharpe",
    "ValidationTrades",
    "ValidationExposure_%",
    "LockedStrategyReturn_%",
    "LockedBuyHoldReturn_%",
    "LockedVsBuyHold_%",
    "LockedMaxDrawdown_%",
    "LockedSharpe",
    "LockedTrades",
    "LockedExposure_%",
    "ReturnChangeVsBaseline_%",
    "DrawdownImprovementVsBaseline_%",
    "SharpeChangeVsBaseline",
    "TradesRemoved",
    "CostFragilityFlag",
    "RiskControlVerdict",
    "FailureReason",
]


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _normalize_mode(mode: str) -> str:
    key = str(mode).lower().strip().replace("-", "_").replace(" ", "_")
    if key not in VALID_SIGNAL_MODES:
        valid = ", ".join(sorted(VALID_SIGNAL_MODES))
        raise ValueError(f"Unknown signal mode {mode!r}. Valid modes: {valid}")
    return "avoid_only" if key == "no_trade_zone" else key


def _normalize_backtest_style(style: str) -> str:
    key = str(style).lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "overlapping": "overlapping_research",
        "research": "overlapping_research",
        "optimistic": "overlapping_research",
        "non_overlapping": "non_overlapping_realistic",
        "realistic": "non_overlapping_realistic",
    }
    key = aliases.get(key, key)
    if key not in VALID_BACKTEST_STYLES:
        valid = ", ".join(sorted(VALID_BACKTEST_STYLES))
        raise ValueError(f"Unknown backtest style {style!r}. Valid styles: {valid}")
    return key


def generate_signals(
    probabilities_up: Iterable[float],
    *,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    mode: str = "long_only",
) -> np.ndarray:
    """Convert P(up) into {-1, 0, 1} positions without looking at returns."""
    p_up = np.asarray(probabilities_up, dtype=float).flatten()
    long_t = float(long_threshold)
    short_t = float(short_threshold)
    signal_mode = _normalize_mode(mode)

    if not 0.0 <= short_t <= 1.0 or not 0.0 <= long_t <= 1.0:
        raise ValueError("Thresholds must be between 0 and 1")
    if short_t >= long_t:
        raise ValueError("short_threshold must be lower than long_threshold")

    signals = np.zeros(len(p_up), dtype=int)
    signals[p_up > long_t] = 1
    if signal_mode == "long_short":
        signals[p_up < short_t] = -1
    return signals


def _classification_metrics(actual_direction: np.ndarray, long_signal: np.ndarray) -> Dict[str, float]:
    actual_up = np.asarray(actual_direction, dtype=int).flatten() == 1
    predicted_up = np.asarray(long_signal, dtype=bool).flatten()

    tp = int(np.sum(predicted_up & actual_up))
    fp = int(np.sum(predicted_up & ~actual_up))
    fn = int(np.sum(~predicted_up & actual_up))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "Precision_UpSignals": round(float(precision * 100.0), 2),
        "Recall_UpSignals": round(float(recall * 100.0), 2),
        "F1_UpSignals": round(float(f1 * 100.0), 2),
    }


def _max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min() * 100.0)


def _sharpe_ratio(returns: np.ndarray, horizon: int) -> float:
    ret = np.asarray(returns, dtype=float)
    ret = ret[np.isfinite(ret)]
    if len(ret) < 2:
        return 0.0
    std = float(np.std(ret, ddof=1))
    if std <= 0.0 or not np.isfinite(std):
        return 0.0
    annualizer = np.sqrt(TRADING_DAYS_PER_YEAR / max(int(horizon), 1))
    return float(annualizer * np.mean(ret) / std)


def _threshold_verdict(
    *,
    signal_count: int,
    total_rows: int,
    strategy_vs_buy_hold_pct: float,
    active_direction_accuracy: float,
    baseline_direction_accuracy: float,
    win_rate: float,
) -> tuple[str, str]:
    min_signals = max(5, int(np.ceil(total_rows * 0.05)))
    warnings: List[str] = []

    if signal_count < min_signals:
        warnings.append("too few active signals")
    if np.isfinite(strategy_vs_buy_hold_pct) and strategy_vs_buy_hold_pct <= 0.0:
        warnings.append("strategy fails buy-and-hold")
    if np.isfinite(active_direction_accuracy) and np.isfinite(baseline_direction_accuracy) and active_direction_accuracy <= baseline_direction_accuracy:
        warnings.append("active direction fails baseline")

    if warnings:
        return "Do not trust for signals", "; ".join(warnings)

    edge = active_direction_accuracy - baseline_direction_accuracy if np.isfinite(baseline_direction_accuracy) else 0.0
    if signal_count >= max(20, min_signals) and strategy_vs_buy_hold_pct > 0.0 and edge >= 5.0 and win_rate >= 55.0:
        return "Medium signal candidate / research only", ""
    return "Low trust / research only", ""


def _minimum_trade_count(total_rows: int) -> int:
    return max(5, int(np.ceil(int(max(total_rows, 0)) * 0.05)))


def _segment_direction_baseline_accuracy(actual_direction: Iterable[int]) -> float:
    y_dir = np.asarray(actual_direction, dtype=float).flatten()
    y_dir = y_dir[np.isfinite(y_dir)]
    if len(y_dir) == 0:
        return np.nan
    up_rate = float(np.mean(y_dir == 1.0) * 100.0)
    down_rate = float(np.mean(y_dir == 0.0) * 100.0)
    return max(up_rate, down_rate)


def _aligned_signal_arrays(
    *,
    probabilities_up: Iterable[float],
    future_returns: Iterable[float],
    actual_direction: Iterable[int],
    test_index: Optional[Iterable[Any]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    p_up = np.asarray(probabilities_up, dtype=float).flatten()
    log_returns = np.asarray(future_returns, dtype=float).flatten()
    y_dir_float = np.asarray(actual_direction, dtype=float).flatten()

    n = min(len(p_up), len(log_returns), len(y_dir_float))
    if n == 0:
        raise ValueError("No signal rows available")

    p_up = p_up[:n]
    log_returns = log_returns[:n]
    y_dir_float = y_dir_float[:n]
    index = pd.Index(test_index[:n]) if test_index is not None else pd.RangeIndex(n)

    finite_mask = np.isfinite(p_up) & np.isfinite(log_returns) & np.isfinite(y_dir_float)
    p_up = p_up[finite_mask]
    log_returns = log_returns[finite_mask]
    y_dir = y_dir_float[finite_mask].astype(int)
    index = index[finite_mask]
    if len(p_up) == 0:
        raise ValueError("No finite signal rows available")
    return p_up, log_returns, y_dir, index


def _chronological_validation_test_split(
    *,
    probabilities_up: Iterable[float],
    future_returns: Iterable[float],
    actual_direction: Iterable[int],
    test_index: Optional[Iterable[Any]] = None,
    validation_fraction: float = 0.5,
) -> Dict[str, Any]:
    p_up, log_returns, y_dir, index = _aligned_signal_arrays(
        probabilities_up=probabilities_up,
        future_returns=future_returns,
        actual_direction=actual_direction,
        test_index=test_index,
    )
    if len(p_up) < 4:
        raise ValueError("At least four out-of-sample rows are required for validation-locked selection")

    frac = float(np.clip(validation_fraction, 0.2, 0.8))
    split_at = int(np.floor(len(p_up) * frac))
    split_at = max(1, min(len(p_up) - 1, split_at))

    return {
        "validation": {
            "probabilities_up": p_up[:split_at],
            "future_returns": log_returns[:split_at],
            "actual_direction": y_dir[:split_at],
            "index": index[:split_at],
        },
        "locked_test": {
            "probabilities_up": p_up[split_at:],
            "future_returns": log_returns[split_at:],
            "actual_direction": y_dir[split_at:],
            "index": index[split_at:],
        },
        "split_at": split_at,
        "rows": len(p_up),
    }


def _metric_value(row: Any, names: Iterable[str], default: float = 0.0) -> float:
    for name in names:
        try:
            if name in row:
                value = _safe_float(row[name], default=np.nan)
                if np.isfinite(value):
                    return value
        except Exception:
            continue
    return float(default)


def _selection_score_components(row: Any) -> Dict[str, float]:
    rows = int(max(0.0, _metric_value(row, ["Rows"], default=0.0)))
    trade_count = _metric_value(row, ["NumberOfTrades", "SignalCount"], default=0.0)
    strategy_reward = _metric_value(row, ["StrategyMinusBuyHold_%"], default=-100.0)
    win_rate = _metric_value(row, ["WinRate_%", "WinRateActive_%"], default=0.0)
    active_accuracy = _metric_value(row, ["DirectionAccuracyActive_%"], default=0.0)
    baseline_accuracy = _metric_value(row, ["BaselineDirectionAccuracy_%"], default=50.0)
    max_drawdown = _metric_value(row, ["MaxDrawdown_%"], default=0.0)

    win_rate_edge = win_rate - 50.0
    direction_edge = active_accuracy - baseline_accuracy
    drawdown_penalty = abs(min(max_drawdown, 0.0)) * 0.5
    low_trade_penalty = max(0.0, float(_minimum_trade_count(rows)) - trade_count) * 5.0

    verdict = str(row.get("ThresholdVerdict", row.get("Verdict", ""))).lower()
    if "do not trust" in verdict:
        verdict_penalty = 40.0
    elif "low" in verdict:
        verdict_penalty = 15.0
    else:
        verdict_penalty = 0.0

    score = (
        strategy_reward
        + win_rate_edge * 0.5
        + direction_edge
        - drawdown_penalty
        - low_trade_penalty
        - verdict_penalty
    )
    return {
        "Selection_StrategyReward": round(float(strategy_reward), 4),
        "Selection_WinRateEdge": round(float(win_rate_edge), 4),
        "Selection_DirectionEdge": round(float(direction_edge), 4),
        "Selection_DrawdownPenalty": round(float(drawdown_penalty), 4),
        "Selection_LowTradePenalty": round(float(low_trade_penalty), 4),
        "Selection_VerdictPenalty": round(float(verdict_penalty), 4),
        "ValidationSelectionScore": round(float(score), 4),
        "MinimumTradeCount": int(_minimum_trade_count(rows)),
    }


def _score_validation_sweep(validation_sweep: pd.DataFrame) -> pd.DataFrame:
    scored = validation_sweep.copy()
    if scored.empty:
        return scored

    component_rows = [_selection_score_components(row) for _, row in scored.iterrows()]
    components = pd.DataFrame(component_rows, index=scored.index)
    scored = pd.concat([scored, components], axis=1)
    if "BacktestStyle" in scored.columns:
        scored["ResearchOnly"] = scored["BacktestStyle"].astype(str).eq("overlapping_research")
    scored["ValidationOnly"] = True
    scored["SelectionSource"] = "validation_segment_within_available_out_of_sample"
    scored["SelectedLockedThreshold"] = False

    sort_cols = ["ValidationSelectionScore", "StrategyMinusBuyHold_%", "NumberOfTrades", "SignalCount"]
    usable_sort_cols = [col for col in sort_cols if col in scored.columns]
    if usable_sort_cols:
        ordered = scored.sort_values(usable_sort_cols, ascending=[False] * len(usable_sort_cols), kind="mergesort")
    else:
        ordered = scored
    if not ordered.empty:
        scored.loc[ordered.index[0], "SelectedLockedThreshold"] = True
    return scored


def _run_backtest_for_style(
    *,
    backtest_style: str,
    probabilities_up: Iterable[float],
    future_returns: Iterable[float],
    actual_direction: Iterable[int],
    test_index: Optional[Iterable[Any]] = None,
    asset: str = "",
    baseline_direction_accuracy: float = np.nan,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    mode: str = "long_only",
    transaction_cost: float = 0.001,
    horizon: int = 1,
    cooldown: int = 0,
) -> SignalBacktestResult:
    style = _normalize_backtest_style(backtest_style)
    if style == "both":
        raise ValueError("Validation-locked evaluation requires one concrete backtest style")
    if style == "non_overlapping_realistic":
        return run_realistic_trade_backtest(
            probabilities_up=probabilities_up,
            future_returns=future_returns,
            actual_direction=actual_direction,
            test_index=test_index,
            asset=asset,
            baseline_direction_accuracy=baseline_direction_accuracy,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            mode=mode,
            transaction_cost=transaction_cost,
            horizon=horizon,
            cooldown=cooldown,
        )
    return run_signal_backtest(
        probabilities_up=probabilities_up,
        future_returns=future_returns,
        actual_direction=actual_direction,
        test_index=test_index,
        baseline_direction_accuracy=baseline_direction_accuracy,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        mode=mode,
        transaction_cost=transaction_cost,
        horizon=horizon,
    )


def _append_warning(existing: Any, warning: str) -> str:
    base = str(existing or "").strip()
    if not warning:
        return base
    return f"{base}; {warning}" if base else warning


def _comparison_row(segment: str, metrics: Dict[str, Any], selection_score: float = np.nan) -> Dict[str, Any]:
    return {
        "Segment": segment,
        "Rows": metrics.get("Rows", np.nan),
        "TradesOrSignals": metrics.get("NumberOfTrades", metrics.get("SignalCount", np.nan)),
        "LongThreshold": metrics.get("LongThreshold", np.nan),
        "ShortThreshold": metrics.get("ShortThreshold", np.nan),
        "WinRate_%": metrics.get("WinRate_%", metrics.get("WinRateActive_%", np.nan)),
        "StrategyMinusBuyHold_%": metrics.get("StrategyMinusBuyHold_%", np.nan),
        "MaxDrawdown_%": metrics.get("MaxDrawdown_%", np.nan),
        "DirectionAccuracyActive_%": metrics.get("DirectionAccuracyActive_%", np.nan),
        "BaselineDirectionAccuracy_%": metrics.get("BaselineDirectionAccuracy_%", np.nan),
        "ThresholdVerdict": metrics.get("ThresholdVerdict", metrics.get("Verdict", "")),
        "ValidationSelectionScore": selection_score,
        "Warnings": metrics.get("Warnings", ""),
    }


def run_signal_backtest(
    *,
    probabilities_up: Iterable[float],
    future_returns: Iterable[float],
    actual_direction: Iterable[int],
    test_index: Optional[Iterable[Any]] = None,
    baseline_direction_accuracy: float = np.nan,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    mode: str = "long_only",
    transaction_cost: float = 0.001,
    horizon: int = 1,
) -> SignalBacktestResult:
    """
    Backtest thresholded direction-probability signals on realized future returns.

    future_returns are Phase 6 future log returns for the selected horizon.
    Thresholds are caller-provided; this function does not tune them on test.
    """
    p_up = np.asarray(probabilities_up, dtype=float).flatten()
    log_returns = np.asarray(future_returns, dtype=float).flatten()
    y_dir = np.asarray(actual_direction, dtype=int).flatten()

    n = min(len(p_up), len(log_returns), len(y_dir))
    if n == 0:
        raise ValueError("No signal rows available")

    p_up = p_up[:n]
    log_returns = log_returns[:n]
    y_dir = y_dir[:n]
    index = pd.DatetimeIndex(test_index[:n]) if test_index is not None else pd.RangeIndex(n)

    finite_mask = np.isfinite(p_up) & np.isfinite(log_returns)
    p_up = p_up[finite_mask]
    log_returns = log_returns[finite_mask]
    y_dir = y_dir[finite_mask]
    index = index[finite_mask]
    if len(p_up) == 0:
        raise ValueError("No finite signal rows available")

    signals = generate_signals(
        p_up,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        mode=mode,
    )
    simple_returns = np.expm1(log_returns)
    gross_strategy_return = signals.astype(float) * simple_returns
    turnover = np.abs(np.diff(signals.astype(float), prepend=0.0))
    costs = turnover * float(transaction_cost)
    strategy_return = gross_strategy_return - costs

    active = signals != 0
    long_mask = signals == 1
    short_mask = signals == -1
    no_trade_mask = signals == 0

    equity = pd.Series((1.0 + strategy_return).cumprod(), index=index)
    buy_hold_equity = pd.Series((1.0 + simple_returns).cumprod(), index=index)

    signal_count = int(active.sum())
    long_count = int(long_mask.sum())
    short_count = int(short_mask.sum())
    no_trade_count = int(no_trade_mask.sum())

    if signal_count:
        win_rate = float(np.mean(strategy_return[active] > 0.0) * 100.0)
        avg_active_return = float(np.mean(strategy_return[active]) * 100.0)
        active_correct = ((signals[active] == 1) & (y_dir[active] == 1)) | ((signals[active] == -1) & (y_dir[active] == 0))
        active_dir_acc = float(np.mean(active_correct) * 100.0)
    else:
        win_rate = 0.0
        avg_active_return = 0.0
        active_dir_acc = 0.0

    strategy_total = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    buy_hold_total = float(buy_hold_equity.iloc[-1] - 1.0) if len(buy_hold_equity) else 0.0
    strategy_vs_bh = (strategy_total - buy_hold_total) * 100.0
    baseline_acc = _safe_float(baseline_direction_accuracy)
    verdict, warnings = _threshold_verdict(
        signal_count=signal_count,
        total_rows=len(p_up),
        strategy_vs_buy_hold_pct=strategy_vs_bh,
        active_direction_accuracy=active_dir_acc,
        baseline_direction_accuracy=baseline_acc,
        win_rate=win_rate,
    )

    class_metrics = _classification_metrics(y_dir, long_mask)
    metrics: Dict[str, Any] = {
        "Mode": _normalize_mode(mode),
        "BacktestStyle": "overlapping_research",
        "LongThreshold": round(float(long_threshold), 4),
        "ShortThreshold": round(float(short_threshold), 4),
        "TransactionCost_%": round(float(transaction_cost) * 100.0, 4),
        "Rows": int(len(p_up)),
        "SignalCount": signal_count,
        "SignalFrequency_%": round(signal_count / max(len(p_up), 1) * 100.0, 2),
        "LongCount": long_count,
        "ShortCount": short_count,
        "NoTradeCount": no_trade_count,
        "WinRateActive_%": round(win_rate, 2),
        "AvgReturnPerActiveSignal_%": round(avg_active_return, 4),
        "StrategyTotalReturn_%": round(strategy_total * 100.0, 4),
        "BuyHoldReturn_%": round(buy_hold_total * 100.0, 4),
        "StrategyMinusBuyHold_%": round(strategy_vs_bh, 4),
        "Sharpe": round(_sharpe_ratio(strategy_return, int(horizon)), 4),
        "MaxDrawdown_%": round(_max_drawdown_pct(equity), 4),
        "DirectionAccuracyActive_%": round(active_dir_acc, 2),
        "BaselineDirectionAccuracy_%": round(baseline_acc, 2) if np.isfinite(baseline_acc) else np.nan,
        "ThresholdVerdict": verdict,
        "Warnings": warnings,
    }
    metrics.update(class_metrics)

    frame = pd.DataFrame(
        {
            "ProbabilityUp": p_up,
            "Signal": signals,
            "ActualDirection": y_dir,
            "FutureLogReturn": log_returns,
            "FutureReturn_%": simple_returns * 100.0,
            "GrossStrategyReturn_%": gross_strategy_return * 100.0,
            "TransactionCost_%": costs * 100.0,
            "StrategyReturn_%": strategy_return * 100.0,
            "StrategyEquity": equity.values,
            "BuyHoldEquity": buy_hold_equity.values,
        },
        index=index,
    )
    return SignalBacktestResult(metrics=metrics, signal_frame=frame)


def run_realistic_trade_backtest(
    *,
    probabilities_up: Iterable[float],
    future_returns: Iterable[float],
    actual_direction: Iterable[int],
    test_index: Optional[Iterable[Any]] = None,
    asset: str = "",
    baseline_direction_accuracy: float = np.nan,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    mode: str = "long_only",
    transaction_cost: float = 0.001,
    horizon: int = 1,
    cooldown: int = 0,
) -> SignalBacktestResult:
    """
    Trade-based realistic backtest with non-overlapping horizon holds.

    A signal at row i opens one trade, holds for horizon rows, exits, then
    waits cooldown rows before the next eligible entry. Rows without enough
    remaining test history to show an exit date are skipped.
    """
    p_up = np.asarray(probabilities_up, dtype=float).flatten()
    log_returns = np.asarray(future_returns, dtype=float).flatten()
    y_dir = np.asarray(actual_direction, dtype=int).flatten()

    n = min(len(p_up), len(log_returns), len(y_dir))
    if n == 0:
        raise ValueError("No signal rows available")

    p_up = p_up[:n]
    log_returns = log_returns[:n]
    y_dir = y_dir[:n]
    index = pd.DatetimeIndex(test_index[:n]) if test_index is not None else pd.RangeIndex(n)

    finite_mask = np.isfinite(p_up) & np.isfinite(log_returns)
    p_up = p_up[finite_mask]
    log_returns = log_returns[finite_mask]
    y_dir = y_dir[finite_mask]
    index = index[finite_mask]
    if len(p_up) == 0:
        raise ValueError("No finite signal rows available")

    h = int(max(1, horizon))
    cooldown_rows = int(max(0, cooldown))
    signals = generate_signals(
        p_up,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        mode=mode,
    )
    simple_returns = np.expm1(log_returns)

    trades: List[Dict[str, Any]] = []
    i = 0
    while i < len(signals):
        signal = int(signals[i])
        if signal == 0:
            i += 1
            continue

        exit_i = i + h
        if exit_i >= len(signals):
            break

        realized = float(simple_returns[i])
        gross_strategy = float(signal * realized)
        strategy_after_cost = gross_strategy - float(transaction_cost) * 2.0
        win = strategy_after_cost > 0.0

        trades.append(
            {
                "EntryRow": int(i),
                "ExitRow": int(exit_i),
                "EntryDate": index[i],
                "ExitDate": index[exit_i],
                "Asset": asset,
                "Horizon": h,
                "Signal": signal,
                "ProbabilityUp": float(p_up[i]),
                "EntryReturnTarget": float(log_returns[i]),
                "RealizedReturn": realized,
                "StrategyReturnAfterCost": strategy_after_cost,
                "HoldingDays": h,
                "Win/Loss": "Win" if win else "Loss",
                "LongThreshold": float(long_threshold),
                "ShortThreshold": float(short_threshold),
                "Mode": _normalize_mode(mode),
                "BacktestStyle": "non_overlapping_realistic",
            }
        )
        i = exit_i + 1 + cooldown_rows

    trade_log = pd.DataFrame(trades)
    baseline_acc = _safe_float(baseline_direction_accuracy)

    if trade_log.empty:
        trade_returns = np.array([], dtype=float)
        strategy_total = 0.0
        win_rate = 0.0
        avg_trade = 0.0
        median_trade = 0.0
        best_trade = 0.0
        worst_trade = 0.0
        max_dd = 0.0
        active_dir_acc = 0.0
        exposure = 0.0
        avg_holding = 0.0
    else:
        trade_returns = trade_log["StrategyReturnAfterCost"].astype(float).to_numpy()
        equity = pd.Series((1.0 + trade_returns).cumprod(), index=pd.to_datetime(trade_log["ExitDate"]))
        strategy_total = float(equity.iloc[-1] - 1.0)
        win_rate = float((trade_returns > 0.0).mean() * 100.0)
        avg_trade = float(np.mean(trade_returns) * 100.0)
        median_trade = float(np.median(trade_returns) * 100.0)
        best_trade = float(np.max(trade_returns) * 100.0)
        worst_trade = float(np.min(trade_returns) * 100.0)
        max_dd = _max_drawdown_pct(equity)
        avg_holding = float(trade_log["HoldingDays"].astype(float).mean())
        exposure = float(min(100.0, trade_log["HoldingDays"].sum() / max(len(p_up), 1) * 100.0))

        entry_positions = trade_log["EntryRow"].astype(int).to_numpy()
        trade_signals = trade_log["Signal"].astype(int).to_numpy()
        trade_dirs = y_dir[entry_positions]
        correct = ((trade_signals == 1) & (trade_dirs == 1)) | ((trade_signals == -1) & (trade_dirs == 0))
        active_dir_acc = float(np.mean(correct) * 100.0)

    possible_bh_entries = np.arange(0, max(len(simple_returns) - h, 0), h)
    buy_hold_returns = simple_returns[possible_bh_entries] if len(possible_bh_entries) else np.array([], dtype=float)
    buy_hold_total = float(np.prod(1.0 + buy_hold_returns) - 1.0) if len(buy_hold_returns) else 0.0
    strategy_vs_bh = (strategy_total - buy_hold_total) * 100.0

    verdict, warnings = _threshold_verdict(
        signal_count=int(len(trade_log)),
        total_rows=len(p_up),
        strategy_vs_buy_hold_pct=strategy_vs_bh,
        active_direction_accuracy=active_dir_acc,
        baseline_direction_accuracy=baseline_acc,
        win_rate=win_rate,
    )

    metrics: Dict[str, Any] = {
        "Mode": _normalize_mode(mode),
        "BacktestStyle": "non_overlapping_realistic",
        "LongThreshold": round(float(long_threshold), 4),
        "ShortThreshold": round(float(short_threshold), 4),
        "TransactionCost_%": round(float(transaction_cost) * 100.0, 4),
        "CooldownRows": cooldown_rows,
        "Rows": int(len(p_up)),
        "NumberOfTrades": int(len(trade_log)),
        "SignalCount": int(len(trade_log)),
        "TradeFrequency_%": round(len(trade_log) / max(len(p_up), 1) * 100.0, 2),
        "SignalFrequency_%": round(len(trade_log) / max(len(p_up), 1) * 100.0, 2),
        "LongCount": int((trade_log["Signal"].eq(1)).sum()) if not trade_log.empty else 0,
        "ShortCount": int((trade_log["Signal"].eq(-1)).sum()) if not trade_log.empty else 0,
        "NoTradeCount": int(max(len(p_up) - len(trade_log), 0)),
        "WinRate_%": round(win_rate, 2),
        "WinRateActive_%": round(win_rate, 2),
        "AverageTradeReturn_%": round(avg_trade, 4),
        "AvgReturnPerActiveSignal_%": round(avg_trade, 4),
        "MedianTradeReturn_%": round(median_trade, 4),
        "TotalCompoundedReturn_%": round(strategy_total * 100.0, 4),
        "StrategyTotalReturn_%": round(strategy_total * 100.0, 4),
        "BuyHoldReturn_%": round(buy_hold_total * 100.0, 4),
        "StrategyMinusBuyHold_%": round(strategy_vs_bh, 4),
        "Sharpe": round(_sharpe_ratio(trade_returns, h), 4),
        "MaxDrawdown_%": round(max_dd, 4),
        "BestTrade_%": round(best_trade, 4),
        "WorstTrade_%": round(worst_trade, 4),
        "AverageHoldingPeriod": round(avg_holding, 2),
        "Exposure_%": round(exposure, 2),
        "DirectionAccuracyActive_%": round(active_dir_acc, 2),
        "BaselineDirectionAccuracy_%": round(baseline_acc, 2) if np.isfinite(baseline_acc) else np.nan,
        "ThresholdVerdict": verdict,
        "Verdict": verdict,
        "Warnings": warnings,
    }

    return SignalBacktestResult(metrics=metrics, signal_frame=trade_log)


def run_threshold_sweep(
    *,
    probabilities_up: Iterable[float],
    future_returns: Iterable[float],
    actual_direction: Iterable[int],
    test_index: Optional[Iterable[Any]] = None,
    baseline_direction_accuracy: float = np.nan,
    mode: str = "long_only",
    long_thresholds: Iterable[float] = (0.55, 0.60, 0.65),
    short_thresholds: Iterable[float] = (0.45, 0.40, 0.35),
    transaction_cost: float = 0.001,
    horizon: int = 1,
    backtest_style: str = "overlapping_research",
    cooldown: int = 0,
    asset: str = "",
) -> pd.DataFrame:
    """
    Research/reporting threshold sweep.

    This table is descriptive only. It must not be treated as a production
    threshold optimizer because all rows are evaluated on the same test split.
    """
    signal_mode = _normalize_mode(mode)
    style = _normalize_backtest_style(backtest_style)
    rows: List[Dict[str, Any]] = []

    if signal_mode == "long_short":
        threshold_pairs = list(zip(long_thresholds, short_thresholds))
    else:
        threshold_pairs = [(float(lt), np.nan) for lt in long_thresholds]

    if style == "both":
        styles_to_run = ["overlapping_research", "non_overlapping_realistic"]
    else:
        styles_to_run = [style]

    for long_t, short_t in threshold_pairs:
        effective_short = float(short_t) if np.isfinite(short_t) else min(0.45, float(long_t) - 0.05)
        try:
            for style_name in styles_to_run:
                if style_name == "non_overlapping_realistic":
                    result = run_realistic_trade_backtest(
                        probabilities_up=probabilities_up,
                        future_returns=future_returns,
                        actual_direction=actual_direction,
                        test_index=test_index,
                        asset=asset,
                        baseline_direction_accuracy=baseline_direction_accuracy,
                        long_threshold=float(long_t),
                        short_threshold=effective_short,
                        mode=signal_mode,
                        transaction_cost=transaction_cost,
                        horizon=horizon,
                        cooldown=cooldown,
                    )
                else:
                    result = run_signal_backtest(
                        probabilities_up=probabilities_up,
                        future_returns=future_returns,
                        actual_direction=actual_direction,
                        test_index=test_index,
                        baseline_direction_accuracy=baseline_direction_accuracy,
                        long_threshold=float(long_t),
                        short_threshold=effective_short,
                        mode=signal_mode,
                        transaction_cost=transaction_cost,
                        horizon=horizon,
                    )
                row = dict(result.metrics)
                row["ResearchOnly"] = True
                rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "Mode": signal_mode,
                    "BacktestStyle": style,
                    "LongThreshold": float(long_t),
                    "ShortThreshold": effective_short,
                    "ThresholdVerdict": "Do not trust for signals",
                    "Warnings": str(exc),
                    "ResearchOnly": True,
                }
            )

    return pd.DataFrame(rows)


def run_validation_locked_signal_engine(
    *,
    signal_output: Any,
    mode: str = "long_only",
    transaction_cost: float = 0.001,
    backtest_style: str = "non_overlapping_realistic",
    cooldown: int = 0,
    validation_fraction: float = 0.5,
    long_thresholds: Iterable[float] = (0.55, 0.60, 0.65, 0.70),
    short_thresholds: Iterable[float] = (0.45, 0.40, 0.35, 0.30),
) -> SignalBacktestResult:
    """
    Select a signal threshold on validation rows, then evaluate once on locked test rows.

    Phase 6 currently exposes one safe out-of-sample signal segment. This
    function splits that segment chronologically into validation and locked
    test slices. Thresholds are swept only on the validation slice; locked-test
    returns are used only after the threshold is fixed.
    """
    style = _normalize_backtest_style(backtest_style)
    if style == "both":
        raise ValueError("Validation-locked selection requires one concrete backtest style")

    split = _chronological_validation_test_split(
        probabilities_up=signal_output.probabilities_up_test,
        future_returns=signal_output.actual_return_test,
        actual_direction=signal_output.actual_direction_test,
        test_index=signal_output.test_index,
        validation_fraction=validation_fraction,
    )
    validation = split["validation"]
    locked_test = split["locked_test"]

    validation_baseline_acc = _segment_direction_baseline_accuracy(validation["actual_direction"])
    locked_baseline_acc = _segment_direction_baseline_accuracy(locked_test["actual_direction"])

    validation_sweep = run_threshold_sweep(
        probabilities_up=validation["probabilities_up"],
        future_returns=validation["future_returns"],
        actual_direction=validation["actual_direction"],
        test_index=validation["index"],
        baseline_direction_accuracy=validation_baseline_acc,
        mode=mode,
        long_thresholds=long_thresholds,
        short_thresholds=short_thresholds,
        transaction_cost=transaction_cost,
        horizon=signal_output.horizon,
        backtest_style=style,
        cooldown=cooldown,
        asset=getattr(signal_output, "asset", ""),
    )
    validation_sweep = _score_validation_sweep(validation_sweep)
    if validation_sweep.empty:
        raise ValueError("Validation threshold sweep produced no candidates")

    selected_rows = validation_sweep[validation_sweep["SelectedLockedThreshold"].eq(True)]
    selected_row = selected_rows.iloc[0] if not selected_rows.empty else validation_sweep.iloc[0]
    selected_long = float(selected_row["LongThreshold"])
    selected_short = _safe_float(selected_row.get("ShortThreshold"), default=np.nan)
    if not np.isfinite(selected_short):
        selected_short = min(0.45, selected_long - 0.05)

    validation_result = _run_backtest_for_style(
        backtest_style=style,
        probabilities_up=validation["probabilities_up"],
        future_returns=validation["future_returns"],
        actual_direction=validation["actual_direction"],
        test_index=validation["index"],
        asset=getattr(signal_output, "asset", ""),
        baseline_direction_accuracy=validation_baseline_acc,
        long_threshold=selected_long,
        short_threshold=selected_short,
        mode=mode,
        transaction_cost=transaction_cost,
        horizon=signal_output.horizon,
        cooldown=cooldown,
    )
    locked_result = _run_backtest_for_style(
        backtest_style=style,
        probabilities_up=locked_test["probabilities_up"],
        future_returns=locked_test["future_returns"],
        actual_direction=locked_test["actual_direction"],
        test_index=locked_test["index"],
        asset=getattr(signal_output, "asset", ""),
        baseline_direction_accuracy=locked_baseline_acc,
        long_threshold=selected_long,
        short_threshold=selected_short,
        mode=mode,
        transaction_cost=transaction_cost,
        horizon=signal_output.horizon,
        cooldown=cooldown,
    )

    validation_result.metrics["Segment"] = "validation"
    locked_result.metrics["Segment"] = "locked_test"

    selected_score = _safe_float(selected_row.get("ValidationSelectionScore"), default=np.nan)
    validation_min_trades = int(selected_row.get("MinimumTradeCount", _minimum_trade_count(len(validation["probabilities_up"]))))
    locked_min_trades = _minimum_trade_count(len(locked_test["probabilities_up"]))
    validation_trades = int(_metric_value(selected_row, ["NumberOfTrades", "SignalCount"], default=0.0))
    locked_trades = int(locked_result.metrics.get("NumberOfTrades", locked_result.metrics.get("SignalCount", 0)))

    validation_warning = ""
    if validation_trades < validation_min_trades:
        validation_warning = f"validation segment has too few trades/signals ({validation_trades} < {validation_min_trades})"
    locked_warning = ""
    if locked_trades < locked_min_trades:
        locked_warning = f"locked test segment has too few trades/signals ({locked_trades} < {locked_min_trades})"

    validation_verdict = str(selected_row.get("ThresholdVerdict", ""))
    locked_verdict = str(locked_result.metrics.get("ThresholdVerdict", ""))
    validation_good = (
        "do not trust" not in validation_verdict.lower()
        and _metric_value(selected_row, ["StrategyMinusBuyHold_%"], default=-1.0) > 0.0
        and validation_trades >= validation_min_trades
    )
    locked_fails = (
        "do not trust" in locked_verdict.lower()
        or _safe_float(locked_result.metrics.get("StrategyMinusBuyHold_%"), default=-1.0) <= 0.0
    )
    if validation_good and locked_fails:
        locked_warning = _append_warning(
            locked_warning,
            "validation looked promising but locked test failed; treat as unstable research evidence",
        )

    if validation_warning:
        validation_result.metrics["Warnings"] = _append_warning(validation_result.metrics.get("Warnings"), validation_warning)
    if locked_warning:
        locked_result.metrics["Warnings"] = _append_warning(locked_result.metrics.get("Warnings"), locked_warning)

    validation_comparison_metrics = dict(validation_result.metrics)
    locked_comparison_metrics = dict(locked_result.metrics)
    comparison = pd.DataFrame(
        [
            _comparison_row("Validation Selection", validation_comparison_metrics, selected_score),
            _comparison_row("Locked Test", locked_comparison_metrics, np.nan),
        ]
    )

    selected_threshold = {
        "ThresholdPolicy": "validation_locked",
        "SelectionSource": "validation_segment_within_available_out_of_sample",
        "BacktestStyle": style,
        "SelectedLongThreshold": round(selected_long, 4),
        "SelectedShortThreshold": round(float(selected_short), 4),
        "ValidationSelectionScore": round(float(selected_score), 4) if np.isfinite(selected_score) else np.nan,
        "ValidationThresholdVerdict": validation_verdict,
        "LockedTestThresholdVerdict": locked_verdict,
        "ValidationRows": int(len(validation["probabilities_up"])),
        "LockedTestRows": int(len(locked_test["probabilities_up"])),
        "ValidationStart": str(validation["index"][0]) if len(validation["index"]) else "",
        "ValidationEnd": str(validation["index"][-1]) if len(validation["index"]) else "",
        "LockedTestStart": str(locked_test["index"][0]) if len(locked_test["index"]) else "",
        "LockedTestEnd": str(locked_test["index"][-1]) if len(locked_test["index"]) else "",
        "ValidationWarning": validation_warning,
        "LockedTestWarning": locked_warning,
    }

    locked_result.metrics.update(selected_threshold)
    locked_result.threshold_sweep = validation_sweep.copy()
    locked_result.validation_sweep = validation_sweep.copy()
    locked_result.validation_signal_frame = validation_result.signal_frame.copy()
    locked_result.validation_metrics = validation_result.metrics.copy()
    locked_result.locked_test_metrics = locked_result.metrics.copy()
    locked_result.validation_test_comparison = comparison
    locked_result.selected_threshold = selected_threshold
    return locked_result


def run_signal_engine(
    *,
    signal_output: Any,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    mode: str = "long_only",
    transaction_cost: float = 0.001,
    backtest_style: str = "overlapping_research",
    cooldown: int = 0,
) -> SignalBacktestResult:
    """Run the signal engine directly from a Phase 6 DirectForecastSignalOutput."""
    style = _normalize_backtest_style(backtest_style)
    if style == "non_overlapping_realistic":
        result = run_realistic_trade_backtest(
            probabilities_up=signal_output.probabilities_up_test,
            future_returns=signal_output.actual_return_test,
            actual_direction=signal_output.actual_direction_test,
            test_index=signal_output.test_index,
            asset=getattr(signal_output, "asset", ""),
            baseline_direction_accuracy=signal_output.direction_baseline_accuracy,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            mode=mode,
            transaction_cost=transaction_cost,
            horizon=signal_output.horizon,
            cooldown=cooldown,
        )
    else:
        result = run_signal_backtest(
            probabilities_up=signal_output.probabilities_up_test,
            future_returns=signal_output.actual_return_test,
            actual_direction=signal_output.actual_direction_test,
            test_index=signal_output.test_index,
            baseline_direction_accuracy=signal_output.direction_baseline_accuracy,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            mode=mode,
            transaction_cost=transaction_cost,
            horizon=signal_output.horizon,
        )
    result.threshold_sweep = run_threshold_sweep(
        probabilities_up=signal_output.probabilities_up_test,
        future_returns=signal_output.actual_return_test,
        actual_direction=signal_output.actual_direction_test,
        test_index=signal_output.test_index,
        baseline_direction_accuracy=signal_output.direction_baseline_accuracy,
        mode=mode,
        transaction_cost=transaction_cost,
        horizon=signal_output.horizon,
        backtest_style="both",
        cooldown=cooldown,
        asset=getattr(signal_output, "asset", ""),
    )
    return result


def score_validation_threshold(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Public wrapper for the transparent validation-only threshold score."""
    return _selection_score_components(metrics)


def robust_signal_verdict(
    *,
    validation_metrics: Dict[str, Any],
    locked_test_metrics: Dict[str, Any],
    validation_score: float = np.nan,
) -> Dict[str, str]:
    """Conservative research verdict. This never labels a signal production-ready."""
    validation_rows = int(_metric_value(validation_metrics, ["Rows"], default=0.0))
    locked_rows = int(_metric_value(locked_test_metrics, ["Rows", "LockedTestRows"], default=0.0))
    validation_trades = int(_metric_value(validation_metrics, ["NumberOfTrades", "SignalCount"], default=0.0))
    locked_trades = int(_metric_value(locked_test_metrics, ["NumberOfTrades", "SignalCount"], default=0.0))
    validation_min_trades = _minimum_trade_count(validation_rows)
    locked_min_trades = _minimum_trade_count(locked_rows)

    validation_vs_bh = _metric_value(validation_metrics, ["StrategyMinusBuyHold_%"], default=np.nan)
    locked_vs_bh = _metric_value(locked_test_metrics, ["StrategyMinusBuyHold_%"], default=np.nan)
    locked_drawdown = _metric_value(locked_test_metrics, ["MaxDrawdown_%"], default=0.0)
    locked_sharpe = _metric_value(locked_test_metrics, ["Sharpe"], default=0.0)
    locked_win_rate = _metric_value(locked_test_metrics, ["WinRate_%", "WinRateActive_%"], default=0.0)

    reasons: List[str] = []
    if validation_trades < validation_min_trades:
        reasons.append(f"validation trades too few ({validation_trades} < {validation_min_trades})")
    if locked_trades < locked_min_trades:
        reasons.append(f"locked test trades too few ({locked_trades} < {locked_min_trades})")
    if np.isfinite(validation_vs_bh) and validation_vs_bh <= 0.0:
        reasons.append("validation fails buy-and-hold")
    if np.isfinite(locked_vs_bh) and locked_vs_bh <= 0.0:
        reasons.append("locked test fails buy-and-hold")
    if np.isfinite(locked_vs_bh) and locked_vs_bh <= -2.0:
        reasons.append("locked test fails buy-and-hold badly")
    if np.isfinite(locked_drawdown) and locked_drawdown < -25.0:
        reasons.append("locked test drawdown is high")
    if np.isfinite(validation_score) and validation_score <= 0.0:
        reasons.append("validation selection score is weak")

    enough_trades = validation_trades >= validation_min_trades and locked_trades >= locked_min_trades
    both_positive = (
        np.isfinite(validation_vs_bh)
        and np.isfinite(locked_vs_bh)
        and validation_vs_bh > 0.0
        and locked_vs_bh > 0.0
    )

    if locked_trades < locked_min_trades or validation_trades < validation_min_trades:
        verdict = "Do Not Trust"
        stability = "LowEvidence"
    elif np.isfinite(locked_vs_bh) and locked_vs_bh <= -2.0:
        verdict = "Do Not Trust"
        stability = "LockedTestFailed"
    elif not both_positive:
        verdict = "Weak / unstable research only"
        if np.isfinite(locked_vs_bh) and locked_vs_bh <= 0.0:
            stability = "LockedTestFailed"
        elif np.isfinite(validation_vs_bh) and validation_vs_bh <= 0.0:
            stability = "ValidationFailed"
        else:
            stability = "MixedWeak"
    elif (
        enough_trades
        and locked_vs_bh >= 5.0
        and locked_drawdown >= -20.0
        and locked_trades >= max(10, locked_min_trades)
        and locked_sharpe > 0.5
        and locked_win_rate >= 55.0
        and np.isfinite(validation_score)
        and validation_score > 0.0
    ):
        verdict = "Strong research candidate / validation-locked"
        stability = "StablePositive"
    else:
        verdict = "Research candidate / validation-locked"
        stability = "StablePositive" if locked_drawdown >= -25.0 else "DrawdownRisk"

    return {
        "RobustnessVerdict": verdict,
        "StabilityFlag": stability,
        "FailureReason": "; ".join(dict.fromkeys(reasons)),
    }


def _scan_metric(metrics: Dict[str, Any], names: Iterable[str], default: float = np.nan) -> float:
    return _metric_value(metrics, names, default=default)


def _signal_scan_row(
    *,
    asset: str,
    horizon: int,
    model_depth: str,
    use_phase5_features: bool,
    signal_mode: str,
    cooldown: int,
    result: SignalBacktestResult,
) -> Dict[str, Any]:
    validation_metrics = result.validation_metrics or {}
    locked_metrics = result.metrics or {}
    selected = result.selected_threshold or {}
    validation_score = _safe_float(selected.get("ValidationSelectionScore"), default=np.nan)
    robust = robust_signal_verdict(
        validation_metrics=validation_metrics,
        locked_test_metrics=locked_metrics,
        validation_score=validation_score,
    )

    row = {
        "Asset": asset,
        "Horizon": int(horizon),
        "ModelDepth": str(model_depth),
        "Phase5Enabled": bool(use_phase5_features),
        "SignalMode": _normalize_mode(signal_mode),
        "BestValidationThreshold": _safe_float(selected.get("SelectedLongThreshold"), default=np.nan),
        "CooldownRows": int(cooldown),
        "ValidationScore": validation_score,
        "ValidationTrades": int(_scan_metric(validation_metrics, ["NumberOfTrades", "SignalCount"], default=0.0)),
        "ValidationWinRate_%": _scan_metric(validation_metrics, ["WinRate_%", "WinRateActive_%"], default=np.nan),
        "ValidationStrategyReturn_%": _scan_metric(validation_metrics, ["TotalCompoundedReturn_%", "StrategyTotalReturn_%"], default=np.nan),
        "ValidationBuyHoldReturn_%": _scan_metric(validation_metrics, ["BuyHoldReturn_%"], default=np.nan),
        "ValidationVsBuyHold_%": _scan_metric(validation_metrics, ["StrategyMinusBuyHold_%"], default=np.nan),
        "LockedTestTrades": int(_scan_metric(locked_metrics, ["NumberOfTrades", "SignalCount"], default=0.0)),
        "LockedTestWinRate_%": _scan_metric(locked_metrics, ["WinRate_%", "WinRateActive_%"], default=np.nan),
        "LockedTestStrategyReturn_%": _scan_metric(locked_metrics, ["TotalCompoundedReturn_%", "StrategyTotalReturn_%"], default=np.nan),
        "LockedTestBuyHoldReturn_%": _scan_metric(locked_metrics, ["BuyHoldReturn_%"], default=np.nan),
        "LockedTestVsBuyHold_%": _scan_metric(locked_metrics, ["StrategyMinusBuyHold_%"], default=np.nan),
        "LockedTestMaxDrawdown_%": _scan_metric(locked_metrics, ["MaxDrawdown_%"], default=np.nan),
        "LockedTestSharpe": _scan_metric(locked_metrics, ["Sharpe"], default=np.nan),
        "Exposure_%": _scan_metric(locked_metrics, ["Exposure_%", "SignalFrequency_%"], default=np.nan),
        "ValidationThresholdVerdict": selected.get("ValidationThresholdVerdict", validation_metrics.get("ThresholdVerdict", "")),
        "LockedTestThresholdVerdict": selected.get("LockedTestThresholdVerdict", locked_metrics.get("ThresholdVerdict", "")),
        "SelectedShortThreshold": _safe_float(selected.get("SelectedShortThreshold"), default=np.nan),
        "SelectionBasis": "validation_score_only",
    }
    row.update(robust)
    return row


def _validation_only_cooldown_candidate_row(
    *,
    asset: str,
    horizon: int,
    model_depth: str,
    use_phase5_features: bool,
    signal_mode: str,
    cooldown: int,
    signal_output: Any,
    thresholds: Iterable[float],
    short_thresholds: Iterable[float],
    validation_fraction: float,
    transaction_cost: float,
) -> Dict[str, Any]:
    split = _chronological_validation_test_split(
        probabilities_up=signal_output.probabilities_up_test,
        future_returns=signal_output.actual_return_test,
        actual_direction=signal_output.actual_direction_test,
        test_index=signal_output.test_index,
        validation_fraction=validation_fraction,
    )
    validation = split["validation"]
    validation_baseline_acc = _segment_direction_baseline_accuracy(validation["actual_direction"])
    validation_sweep = run_threshold_sweep(
        probabilities_up=validation["probabilities_up"],
        future_returns=validation["future_returns"],
        actual_direction=validation["actual_direction"],
        test_index=validation["index"],
        baseline_direction_accuracy=validation_baseline_acc,
        mode=signal_mode,
        long_thresholds=thresholds,
        short_thresholds=short_thresholds,
        transaction_cost=transaction_cost,
        horizon=horizon,
        backtest_style="non_overlapping_realistic",
        cooldown=cooldown,
        asset=asset,
    )
    validation_sweep = _score_validation_sweep(validation_sweep)
    if validation_sweep.empty:
        raise ValueError("Validation-only cooldown sweep produced no candidates")

    selected_rows = validation_sweep[validation_sweep["SelectedLockedThreshold"].eq(True)]
    selected = selected_rows.iloc[0] if not selected_rows.empty else validation_sweep.iloc[0]
    validation_score = _safe_float(selected.get("ValidationSelectionScore"), default=np.nan)
    validation_trades = int(_metric_value(selected, ["NumberOfTrades", "SignalCount"], default=0.0))
    validation_min_trades = _minimum_trade_count(int(_metric_value(selected, ["Rows"], default=0.0)))
    failure_reason = str(selected.get("Warnings", "") or "")
    if validation_trades < validation_min_trades:
        failure_reason = _append_warning(
            failure_reason,
            f"validation trades too few ({validation_trades} < {validation_min_trades})",
        )

    row = {col: np.nan for col in SIGNAL_SCAN_COLUMNS}
    row.update(
        {
            "Asset": asset,
            "Horizon": int(horizon),
            "ModelDepth": str(model_depth),
            "Phase5Enabled": bool(use_phase5_features),
            "SignalMode": _normalize_mode(signal_mode),
            "BestValidationThreshold": _safe_float(selected.get("LongThreshold"), default=np.nan),
            "CooldownRows": int(cooldown),
            "ValidationScore": validation_score,
            "ValidationTrades": validation_trades,
            "ValidationWinRate_%": _metric_value(selected, ["WinRate_%", "WinRateActive_%"], default=np.nan),
            "ValidationStrategyReturn_%": _metric_value(selected, ["TotalCompoundedReturn_%", "StrategyTotalReturn_%"], default=np.nan),
            "ValidationBuyHoldReturn_%": _metric_value(selected, ["BuyHoldReturn_%"], default=np.nan),
            "ValidationVsBuyHold_%": _metric_value(selected, ["StrategyMinusBuyHold_%"], default=np.nan),
            "RobustnessVerdict": "Validation candidate only",
            "StabilityFlag": "ValidationOnly",
            "FailureReason": failure_reason,
            "ValidationThresholdVerdict": selected.get("ThresholdVerdict", ""),
            "SelectedShortThreshold": _safe_float(selected.get("ShortThreshold"), default=np.nan),
            "SelectionBasis": "validation_score_only",
            "SelectionCandidate": True,
            "SelectedCooldownForAssetHorizon": False,
        }
    )
    return row


def _failed_signal_scan_row(
    *,
    asset: str,
    horizon: int,
    model_depth: str,
    use_phase5_features: bool,
    signal_mode: str,
    failure_reason: str,
) -> Dict[str, Any]:
    row = {col: np.nan for col in SIGNAL_SCAN_COLUMNS}
    row.update(
        {
            "Asset": asset,
            "Horizon": int(horizon),
            "ModelDepth": str(model_depth),
            "Phase5Enabled": bool(use_phase5_features),
            "SignalMode": _normalize_mode(signal_mode),
            "RobustnessVerdict": "Do Not Trust",
            "StabilityFlag": "ScanFailed",
            "FailureReason": str(failure_reason),
        }
    )
    return row


def summarize_signal_scan(results: pd.DataFrame) -> Dict[str, Any]:
    """Return verdict counts and sorted candidate/failure views for scanner output."""
    if results is None or results.empty:
        empty = pd.DataFrame(columns=SIGNAL_SCAN_COLUMNS)
        return {
            "verdict_counts": {},
            "top_robust_candidates": empty,
            "failed_candidates": empty,
        }

    df = results.copy()
    counts = df["RobustnessVerdict"].fillna("Unknown").value_counts().to_dict()
    positive_mask = df["RobustnessVerdict"].astype(str).str.contains("candidate", case=False, na=False)
    top = df[positive_mask].sort_values(
        ["LockedTestVsBuyHold_%", "ValidationScore", "LockedTestTrades"],
        ascending=[False, False, False],
        na_position="last",
    )
    failed = df[~positive_mask].sort_values(
        ["RobustnessVerdict", "LockedTestVsBuyHold_%", "ValidationScore"],
        ascending=[True, True, True],
        na_position="last",
    )
    return {
        "verdict_counts": counts,
        "top_robust_candidates": top,
        "failed_candidates": failed,
    }


def _lookup_signal_output(signal_outputs: Optional[Dict[Any, Any]], asset: str, horizon: int) -> Optional[Any]:
    if not signal_outputs:
        return None
    keys = [
        (asset, int(horizon)),
        (asset, str(horizon)),
        f"{asset}:{int(horizon)}",
        f"{asset}_{int(horizon)}",
    ]
    for key in keys:
        if key in signal_outputs:
            return signal_outputs[key]
    asset_outputs = signal_outputs.get(asset)
    if isinstance(asset_outputs, dict):
        return asset_outputs.get(int(horizon)) or asset_outputs.get(str(horizon))
    return None


def _build_signal_output(
    *,
    raw_df: Optional[pd.DataFrame],
    asset: str,
    horizon: int,
    model_depth: str,
    use_phase5_features: bool,
    signal_outputs: Optional[Dict[Any, Any]],
    signal_output_factory: Optional[Any],
) -> Any:
    existing = _lookup_signal_output(signal_outputs, asset, horizon)
    if existing is not None:
        return existing

    if signal_output_factory is not None:
        try:
            return signal_output_factory(
                asset_name=asset,
                horizon=int(horizon),
                model_depth=model_depth,
                use_phase5_features=use_phase5_features,
            )
        except TypeError:
            return signal_output_factory(asset, int(horizon))

    if raw_df is None:
        raise ValueError("raw_df is required when no signal_outputs or signal_output_factory are supplied")

    from src.direct_forecast_models import run_direct_forecast_signal_output

    return run_direct_forecast_signal_output(
        raw_df=raw_df,
        asset_name=asset,
        horizon=int(horizon),
        model_depth=model_depth,
        use_phase5_features=use_phase5_features,
    )


def run_signal_research_scan(
    *,
    raw_df: Optional[pd.DataFrame] = None,
    asset_names: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    model_depth: str = "core",
    use_phase5_features: bool = True,
    signal_mode: str = "long_only",
    threshold_candidates: Iterable[float] = (0.50, 0.55, 0.60, 0.65, 0.70),
    cooldown_candidates: Iterable[int] = (0, 2, 5),
    validation_fraction: float = 0.5,
    transaction_cost: float = 0.001,
    signal_outputs: Optional[Dict[Any, Any]] = None,
    signal_output_factory: Optional[Any] = None,
    progress_callback: Optional[Any] = None,
) -> SignalResearchScanReport:
    """
    Robust multi-asset signal scanner using Phase 7C validation-locked logic.

    Thresholds are selected only inside run_validation_locked_signal_engine's
    validation segment. Cooldown selection is then based only on the validation
    score from each cooldown candidate; locked-test metrics are reported after
    selection and are never used to pick the threshold or cooldown.
    """
    if asset_names is None:
        from src.asset_config import get_asset_names

        assets = list(get_asset_names())
    else:
        assets = list(asset_names)

    if horizons is None:
        try:
            from src.direct_forecast_models import DIRECT_FORECAST_HORIZONS

            scan_horizons = list(DIRECT_FORECAST_HORIZONS)
        except Exception:
            scan_horizons = [1, 5, 10, 20, 30]
    else:
        scan_horizons = [int(h) for h in horizons]

    thresholds = [float(t) for t in threshold_candidates]
    cooldowns = [int(max(0, c)) for c in cooldown_candidates]
    if not assets:
        raise ValueError("Select at least one asset")
    if not scan_horizons:
        raise ValueError("Select at least one horizon")
    if not thresholds:
        raise ValueError("Select at least one threshold candidate")
    if not cooldowns:
        raise ValueError("Select at least one cooldown candidate")

    mode = _normalize_mode(signal_mode)
    if mode != "long_only":
        # Kept available for later research, but the serious default remains long_only.
        short_thresholds = [max(0.0, min(0.49, 1.0 - t)) for t in thresholds]
    else:
        short_thresholds = tuple(min(0.45, t - 0.05) for t in thresholds)

    rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    outputs: Dict[tuple, Any] = {}
    total = len(assets) * len(scan_horizons)
    done = 0

    for asset in assets:
        for horizon in scan_horizons:
            try:
                if progress_callback is not None:
                    progress_callback(done, total, f"Preparing {asset} {horizon}D")

                signal_output = _build_signal_output(
                    raw_df=raw_df,
                    asset=asset,
                    horizon=horizon,
                    model_depth=model_depth,
                    use_phase5_features=use_phase5_features,
                    signal_outputs=signal_outputs,
                    signal_output_factory=signal_output_factory,
                )
                outputs[(asset, int(horizon))] = signal_output

                asset_horizon_candidates: List[Dict[str, Any]] = []
                for cooldown in cooldowns:
                    row = _validation_only_cooldown_candidate_row(
                        asset=asset,
                        horizon=int(horizon),
                        model_depth=model_depth,
                        use_phase5_features=use_phase5_features,
                        signal_mode=mode,
                        cooldown=cooldown,
                        signal_output=signal_output,
                        thresholds=thresholds,
                        short_thresholds=short_thresholds,
                        validation_fraction=validation_fraction,
                        transaction_cost=transaction_cost,
                    )
                    asset_horizon_candidates.append(row)
                    candidate_rows.append(row.copy())

                ordered_candidates = sorted(
                    asset_horizon_candidates,
                    key=lambda r: (
                        -_safe_float(r.get("ValidationScore"), default=-np.inf),
                        -_safe_float(r.get("ValidationTrades"), default=-np.inf),
                        _safe_float(r.get("CooldownRows"), default=np.inf),
                    ),
                )
                selected = dict(ordered_candidates[0])
                selected["SelectedCooldownForAssetHorizon"] = True
                for candidate in candidate_rows[-len(asset_horizon_candidates):]:
                    candidate["SelectedCooldownForAssetHorizon"] = (
                        int(candidate.get("CooldownRows", -1)) == int(selected.get("CooldownRows", -2))
                    )

                locked_result = run_validation_locked_signal_engine(
                    signal_output=signal_output,
                    mode=mode,
                    transaction_cost=transaction_cost,
                    backtest_style="non_overlapping_realistic",
                    cooldown=int(selected["CooldownRows"]),
                    validation_fraction=validation_fraction,
                    long_thresholds=thresholds,
                    short_thresholds=short_thresholds,
                )
                locked_row = _signal_scan_row(
                    asset=asset,
                    horizon=int(horizon),
                    model_depth=model_depth,
                    use_phase5_features=use_phase5_features,
                    signal_mode=mode,
                    cooldown=int(selected["CooldownRows"]),
                    result=locked_result,
                )
                locked_row["SelectedCooldownForAssetHorizon"] = True
                rows.append(locked_row)
            except Exception as exc:
                err = {"Asset": asset, "Horizon": int(horizon), "Error": str(exc)}
                errors.append(err)
                rows.append(
                    _failed_signal_scan_row(
                        asset=asset,
                        horizon=int(horizon),
                        model_depth=model_depth,
                        use_phase5_features=use_phase5_features,
                        signal_mode=mode,
                        failure_reason=str(exc),
                    )
                )
            finally:
                done += 1
                if progress_callback is not None:
                    progress_callback(done, total, f"Completed {asset} {horizon}D")

    results = pd.DataFrame(rows)
    if not results.empty:
        for col in SIGNAL_SCAN_COLUMNS:
            if col not in results.columns:
                results[col] = np.nan
        leading_cols = SIGNAL_SCAN_COLUMNS
        extra_cols = [c for c in results.columns if c not in leading_cols]
        results = results[leading_cols + extra_cols]

    candidate_results = pd.DataFrame(candidate_rows)
    summary = summarize_signal_scan(results)
    settings = {
        "assets": assets,
        "horizons": scan_horizons,
        "model_depth": model_depth,
        "use_phase5_features": bool(use_phase5_features),
        "signal_mode": mode,
        "backtest_style": "non_overlapping_realistic",
        "threshold_policy": "validation_locked",
        "threshold_candidates": thresholds,
        "cooldown_candidates": cooldowns,
        "validation_fraction": float(validation_fraction),
        "transaction_cost": float(transaction_cost),
        "cooldown_selection_basis": "validation_score_only",
    }
    return SignalResearchScanReport(
        full_results=results,
        verdict_counts=summary["verdict_counts"],
        top_robust_candidates=summary["top_robust_candidates"],
        failed_candidates=summary["failed_candidates"],
        errors=pd.DataFrame(errors),
        settings=settings,
        candidate_results=candidate_results,
        signal_outputs=outputs,
    )


def summarize_trade_diagnostics(trade_log: pd.DataFrame, metrics: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Summarize locked-test trade behavior without changing signal selection."""
    metrics = metrics or {}
    if trade_log is None or trade_log.empty:
        row = {
            "NumberOfTrades": 0,
            "WinRate_%": 0.0,
            "AverageWin_%": 0.0,
            "AverageLoss_%": 0.0,
            "ProfitFactor": 0.0,
            "BestTrade_%": 0.0,
            "WorstTrade_%": 0.0,
            "LongestWinStreak": 0,
            "LongestLossStreak": 0,
            "MedianReturn_%": 0.0,
            "ReturnStd_%": 0.0,
            "Exposure_%": _safe_float(metrics.get("Exposure_%"), default=0.0),
        }
        return pd.DataFrame([row])

    returns = trade_log["StrategyReturnAfterCost"].astype(float).to_numpy()
    wins = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    labels = ["Win" if r > 0.0 else "Loss" for r in returns]

    longest_win = 0
    longest_loss = 0
    current_label = ""
    current_len = 0
    for label in labels:
        if label == current_label:
            current_len += 1
        else:
            current_label = label
            current_len = 1
        if label == "Win":
            longest_win = max(longest_win, current_len)
        else:
            longest_loss = max(longest_loss, current_len)

    gross_profit = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) else 0.0
    if gross_loss > 0.0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0.0:
        profit_factor = np.inf
    else:
        profit_factor = 0.0

    row = {
        "NumberOfTrades": int(len(returns)),
        "WinRate_%": round(float(np.mean(returns > 0.0) * 100.0), 2),
        "AverageWin_%": round(float(np.mean(wins) * 100.0), 4) if len(wins) else 0.0,
        "AverageLoss_%": round(float(np.mean(losses) * 100.0), 4) if len(losses) else 0.0,
        "ProfitFactor": round(float(profit_factor), 4) if np.isfinite(profit_factor) else np.inf,
        "BestTrade_%": round(float(np.max(returns) * 100.0), 4),
        "WorstTrade_%": round(float(np.min(returns) * 100.0), 4),
        "LongestWinStreak": int(longest_win),
        "LongestLossStreak": int(longest_loss),
        "MedianReturn_%": round(float(np.median(returns) * 100.0), 4),
        "ReturnStd_%": round(float(np.std(returns, ddof=1) * 100.0), 4) if len(returns) > 1 else 0.0,
        "Exposure_%": _safe_float(metrics.get("Exposure_%"), default=0.0),
    }
    return pd.DataFrame([row])


def _period_key(index: Iterable[Any], period: str) -> pd.Series:
    dates = pd.to_datetime(pd.Index(index))
    freq = "Q" if str(period).upper().startswith("Q") else "M"
    return pd.Series(dates.to_period(freq).astype(str), index=np.arange(len(dates)))


def _compound_percent(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    return float((np.prod(1.0 + arr) - 1.0) * 100.0)


def build_monthly_return_table(
    trade_log: pd.DataFrame,
    locked_test_index: Iterable[Any],
    locked_test_future_returns: Iterable[float],
    *,
    period: str = "M",
) -> pd.DataFrame:
    """Build monthly or quarterly diagnostic returns for strategy vs buy-and-hold."""
    freq = "Q" if str(period).upper().startswith("Q") else "M"
    period_label = "Quarter" if freq == "Q" else "Month"

    if trade_log is not None and not trade_log.empty:
        trade_df = trade_log.copy()
        trade_df["Period"] = pd.to_datetime(trade_df["ExitDate"]).dt.to_period(freq).astype(str)
        strategy = trade_df.groupby("Period")["StrategyReturnAfterCost"].apply(_compound_percent)
    else:
        strategy = pd.Series(dtype=float)

    locked_dates = pd.to_datetime(pd.Index(locked_test_index))
    locked_returns = np.expm1(np.asarray(locked_test_future_returns, dtype=float).flatten())
    n = min(len(locked_dates), len(locked_returns))
    if n:
        bh_df = pd.DataFrame(
            {
                "Period": locked_dates[:n].to_period(freq).astype(str),
                "BuyHoldReturn": locked_returns[:n],
            }
        )
        buy_hold = bh_df.groupby("Period")["BuyHoldReturn"].apply(_compound_percent)
    else:
        buy_hold = pd.Series(dtype=float)

    periods = sorted(set(strategy.index.astype(str)).union(set(buy_hold.index.astype(str))))
    rows: List[Dict[str, Any]] = []
    for key in periods:
        strat = float(strategy.get(key, 0.0))
        bh = float(buy_hold.get(key, 0.0))
        rows.append(
            {
                period_label: key,
                "StrategyReturn_%": round(strat, 4),
                "BuyHoldReturn_%": round(bh, 4),
                "VsBuyHold_%": round(strat - bh, 4),
                "MissedLargeBuyHoldRally": bool(bh >= 5.0 and strat < bh - 5.0),
            }
        )
    return pd.DataFrame(rows)


def build_equity_drawdown_table(trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log is None or trade_log.empty:
        return pd.DataFrame(columns=["Date", "Equity", "RunningPeak", "Drawdown_%"])

    returns = trade_log["StrategyReturnAfterCost"].astype(float).to_numpy()
    dates = pd.to_datetime(trade_log["ExitDate"])
    equity = pd.Series((1.0 + returns).cumprod(), index=dates)
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return pd.DataFrame(
        {
            "Date": equity.index,
            "Equity": equity.values,
            "RunningPeak": running_peak.values,
            "Drawdown_%": drawdown.values * 100.0,
        }
    )


def _drawdown_summary(equity_curve: pd.DataFrame) -> Dict[str, Any]:
    if equity_curve is None or equity_curve.empty:
        return {"MaxDrawdownStart": "", "MaxDrawdownEnd": "", "RecoveredAfterMaxDrawdown": False}

    dd = equity_curve["Drawdown_%"].astype(float)
    end_idx = int(dd.idxmin())
    end_date = equity_curve.loc[end_idx, "Date"]
    peak_rows = equity_curve.loc[:end_idx]
    if peak_rows.empty:
        start_date = ""
    else:
        start_idx = int(peak_rows["Equity"].astype(float).idxmax())
        start_date = equity_curve.loc[start_idx, "Date"]

    post = equity_curve.loc[end_idx:]
    recovered = False
    if not post.empty:
        peak_value = float(equity_curve.loc[:end_idx, "RunningPeak"].iloc[-1])
        recovered = bool((post["Equity"].astype(float) >= peak_value).any())
    return {
        "MaxDrawdownStart": str(start_date),
        "MaxDrawdownEnd": str(end_date),
        "RecoveredAfterMaxDrawdown": recovered,
    }


def diagnose_benchmark_dependency(metrics: Dict[str, Any]) -> Dict[str, Any]:
    strategy_return = _metric_value(metrics, ["TotalCompoundedReturn_%", "StrategyTotalReturn_%"], default=np.nan)
    vs_buy_hold = _metric_value(metrics, ["StrategyMinusBuyHold_%"], default=np.nan)
    flag = bool(np.isfinite(strategy_return) and np.isfinite(vs_buy_hold) and strategy_return < 0.0 and vs_buy_hold > 0.0)
    warning = "BenchmarkWeakness: strategy only beats buy-and-hold because benchmark was worse." if flag else ""
    return {"BenchmarkWeakness": flag, "BenchmarkDependencyWarning": warning}


def build_cost_sensitivity_table(
    *,
    signal_output: Any,
    selected_long_threshold: float,
    selected_short_threshold: float,
    selected_cooldown: int,
    validation_fraction: float,
    mode: str = "long_only",
    costs: Iterable[float] = (0.0, 0.0005, 0.001, 0.002, 0.005),
) -> pd.DataFrame:
    """Re-evaluate locked-test economics at fixed validation-selected settings."""
    split = _chronological_validation_test_split(
        probabilities_up=signal_output.probabilities_up_test,
        future_returns=signal_output.actual_return_test,
        actual_direction=signal_output.actual_direction_test,
        test_index=signal_output.test_index,
        validation_fraction=validation_fraction,
    )
    locked = split["locked_test"]
    locked_baseline = _segment_direction_baseline_accuracy(locked["actual_direction"])
    rows: List[Dict[str, Any]] = []
    for cost in costs:
        result = _run_backtest_for_style(
            backtest_style="non_overlapping_realistic",
            probabilities_up=locked["probabilities_up"],
            future_returns=locked["future_returns"],
            actual_direction=locked["actual_direction"],
            test_index=locked["index"],
            asset=getattr(signal_output, "asset", ""),
            baseline_direction_accuracy=locked_baseline,
            long_threshold=selected_long_threshold,
            short_threshold=selected_short_threshold,
            mode=mode,
            transaction_cost=float(cost),
            horizon=signal_output.horizon,
            cooldown=selected_cooldown,
        )
        metrics = result.metrics
        rows.append(
            {
                "TransactionCost_%": round(float(cost) * 100.0, 4),
                "SelectedThreshold": round(float(selected_long_threshold), 4),
                "SelectedCooldown": int(selected_cooldown),
                "Trades": int(metrics.get("NumberOfTrades", metrics.get("SignalCount", 0))),
                "StrategyReturn_%": metrics.get("TotalCompoundedReturn_%", metrics.get("StrategyTotalReturn_%", np.nan)),
                "BuyHoldReturn_%": metrics.get("BuyHoldReturn_%", np.nan),
                "VsBuyHold_%": metrics.get("StrategyMinusBuyHold_%", np.nan),
                "MaxDrawdown_%": metrics.get("MaxDrawdown_%", np.nan),
                "Sharpe": metrics.get("Sharpe", np.nan),
                "Verdict": metrics.get("ThresholdVerdict", metrics.get("Verdict", "")),
            }
        )
    return pd.DataFrame(rows)


def build_validation_split_sensitivity_table(
    *,
    signal_output: Any,
    asset_name: str,
    horizon: int,
    model_depth: str,
    use_phase5_features: bool,
    mode: str = "long_only",
    threshold_candidates: Iterable[float] = (0.50, 0.55, 0.60, 0.65, 0.70),
    cooldown_candidates: Iterable[int] = (0, 2, 5),
    split_fractions: Iterable[float] = (0.40, 0.50, 0.60),
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """Rerun validation-locked selection independently for each split fraction."""
    rows: List[Dict[str, Any]] = []
    outputs = {(asset_name, int(horizon)): signal_output}
    for split in split_fractions:
        report = run_signal_research_scan(
            asset_names=[asset_name],
            horizons=[int(horizon)],
            model_depth=model_depth,
            use_phase5_features=use_phase5_features,
            signal_mode=mode,
            threshold_candidates=threshold_candidates,
            cooldown_candidates=cooldown_candidates,
            validation_fraction=float(split),
            transaction_cost=transaction_cost,
            signal_outputs=outputs,
        )
        if report.full_results.empty:
            rows.append({"ValidationSegment_%": round(float(split) * 100.0, 2), "FailureReason": "no split result"})
            continue
        result = report.full_results.iloc[0].to_dict()
        rows.append(
            {
                "ValidationSegment_%": round(float(split) * 100.0, 2),
                "SelectedThreshold": result.get("BestValidationThreshold", np.nan),
                "SelectedCooldown": result.get("CooldownRows", np.nan),
                "ValidationScore": result.get("ValidationScore", np.nan),
                "LockedTestTrades": result.get("LockedTestTrades", np.nan),
                "LockedTestStrategyReturn_%": result.get("LockedTestStrategyReturn_%", np.nan),
                "LockedTestBuyHoldReturn_%": result.get("LockedTestBuyHoldReturn_%", np.nan),
                "LockedTestVsBuyHold_%": result.get("LockedTestVsBuyHold_%", np.nan),
                "LockedTestMaxDrawdown_%": result.get("LockedTestMaxDrawdown_%", np.nan),
                "RobustnessVerdict": result.get("RobustnessVerdict", ""),
                "StabilityFlag": result.get("StabilityFlag", ""),
                "FailureReason": result.get("FailureReason", ""),
            }
        )
    return pd.DataFrame(rows)


def _probability_diagnostics(trade_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trade_log is None or trade_log.empty or "ProbabilityUp" not in trade_log.columns:
        empty_summary = pd.DataFrame(
            [
                {
                    "TradeCount": 0,
                    "MeanProbabilityUp": np.nan,
                    "AvgProbabilityUp_Winners": np.nan,
                    "AvgProbabilityUp_Losers": np.nan,
                }
            ]
        )
        return empty_summary, pd.DataFrame()

    df = trade_log.copy()
    df["ProbabilityUp"] = df["ProbabilityUp"].astype(float)
    df["Won"] = df["StrategyReturnAfterCost"].astype(float) > 0.0
    winners = df[df["Won"]]
    losers = df[~df["Won"]]
    summary = pd.DataFrame(
        [
            {
                "TradeCount": int(len(df)),
                "MeanProbabilityUp": round(float(df["ProbabilityUp"].mean()), 4),
                "MedianProbabilityUp": round(float(df["ProbabilityUp"].median()), 4),
                "MinProbabilityUp": round(float(df["ProbabilityUp"].min()), 4),
                "MaxProbabilityUp": round(float(df["ProbabilityUp"].max()), 4),
                "AvgProbabilityUp_Winners": round(float(winners["ProbabilityUp"].mean()), 4) if not winners.empty else np.nan,
                "AvgProbabilityUp_Losers": round(float(losers["ProbabilityUp"].mean()), 4) if not losers.empty else np.nan,
            }
        ]
    )

    bins = pd.cut(df["ProbabilityUp"], bins=[0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0], include_lowest=True)
    bin_table = (
        df.assign(ProbabilityBin=bins)
        .groupby("ProbabilityBin", observed=False)
        .agg(
            Trades=("ProbabilityUp", "size"),
            WinRate_pct=("Won", lambda x: float(np.mean(x) * 100.0) if len(x) else 0.0),
            AvgTradeReturn_pct=("StrategyReturnAfterCost", lambda x: float(np.mean(x) * 100.0) if len(x) else 0.0),
        )
        .reset_index()
    )
    bin_table["ProbabilityBin"] = bin_table["ProbabilityBin"].astype(str)
    return summary, bin_table


def _diagnostic_warning_flags(
    *,
    candidate_row: Dict[str, Any],
    trade_diagnostics: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    split_sensitivity: pd.DataFrame,
) -> List[str]:
    warnings: List[str] = []
    benchmark = diagnose_benchmark_dependency(candidate_row)
    if benchmark["BenchmarkWeakness"]:
        warnings.append(benchmark["BenchmarkDependencyWarning"])

    locked_drawdown = _safe_float(candidate_row.get("LockedTestMaxDrawdown_%"), default=0.0)
    if locked_drawdown <= -20.0:
        warnings.append(f"DrawdownRisk: locked-test max drawdown is {locked_drawdown:.2f}%.")

    locked_trades = int(_safe_float(candidate_row.get("LockedTestTrades"), default=0.0))
    locked_rows = int(_safe_float(candidate_row.get("LockedTestRows"), default=0.0))
    min_trades = _minimum_trade_count(locked_rows)
    if locked_trades < min_trades:
        warnings.append(f"LowTradeCount: locked test has {locked_trades} trades; minimum evidence threshold is {min_trades}.")

    if cost_sensitivity is not None and not cost_sensitivity.empty:
        zero_rows = cost_sensitivity[cost_sensitivity["TransactionCost_%"].astype(float).le(0.0001)]
        realistic_rows = cost_sensitivity[cost_sensitivity["TransactionCost_%"].astype(float).ge(0.10)]
        zero_ok = bool(not zero_rows.empty and _safe_float(zero_rows.iloc[0].get("VsBuyHold_%"), default=-np.inf) > 0.0)
        realistic_fails = bool(not realistic_rows.empty and (realistic_rows["VsBuyHold_%"].astype(float) <= 0.0).any())
        if zero_ok and realistic_fails:
            warnings.append("CostFragile: candidate only works at very low transaction cost.")

    if split_sensitivity is not None and not split_sensitivity.empty:
        thresholds = split_sensitivity["SelectedThreshold"].dropna().astype(float)
        cooldowns = split_sensitivity["SelectedCooldown"].dropna().astype(float)
        vs_bh = split_sensitivity["LockedTestVsBuyHold_%"].dropna().astype(float)
        unstable = False
        reasons: List[str] = []
        if len(thresholds) and thresholds.max() - thresholds.min() >= 0.10:
            unstable = True
            reasons.append("threshold changes materially")
        if cooldowns.nunique() > 1:
            unstable = True
            reasons.append("cooldown changes")
        if len(vs_bh) and (vs_bh.max() - vs_bh.min() >= 15.0 or (vs_bh.gt(0).any() and vs_bh.le(0).any())):
            unstable = True
            reasons.append("locked-test performance changes heavily")
        if unstable:
            warnings.append(f"SplitUnstable: {', '.join(reasons)}.")

    failure_reason = str(candidate_row.get("FailureReason", "") or "")
    if failure_reason:
        warnings.append(f"CandidateWarning: {failure_reason}")
    return list(dict.fromkeys([w for w in warnings if w]))


def run_candidate_deep_diagnostics(
    *,
    raw_df: Optional[pd.DataFrame] = None,
    asset_name: str = "Silver",
    horizon: int = 5,
    model_depth: str = "core",
    use_phase5_features: bool = True,
    signal_mode: str = "long_only",
    threshold_candidates: Iterable[float] = (0.50, 0.55, 0.60, 0.65, 0.70),
    cooldown_candidates: Iterable[int] = (0, 2, 5),
    validation_fraction: float = 0.5,
    transaction_cost: float = 0.001,
    cost_values: Iterable[float] = (0.0, 0.0005, 0.001, 0.002, 0.005),
    split_fractions: Iterable[float] = (0.40, 0.50, 0.60),
    signal_output: Optional[Any] = None,
    signal_output_factory: Optional[Any] = None,
) -> CandidateDiagnosticsReport:
    """
    Deep diagnostics for one validation-locked signal candidate.

    Cooldown is selected through the Phase 7D scanner using validation score
    only. Threshold selection and locked-test evaluation then use Phase 7C.
    """
    if signal_output is None:
        signal_output = _build_signal_output(
            raw_df=raw_df,
            asset=asset_name,
            horizon=int(horizon),
            model_depth=model_depth,
            use_phase5_features=use_phase5_features,
            signal_outputs=None,
            signal_output_factory=signal_output_factory,
        )
    asset = str(asset_name or getattr(signal_output, "asset", ""))
    h = int(horizon or getattr(signal_output, "horizon", 1))
    thresholds = [float(t) for t in threshold_candidates]
    cooldowns = [int(max(0, c)) for c in cooldown_candidates]
    mode = _normalize_mode(signal_mode)
    short_thresholds = [max(0.0, min(0.49, 1.0 - t)) for t in thresholds] if mode != "long_only" else [min(0.45, t - 0.05) for t in thresholds]

    scan_report = run_signal_research_scan(
        asset_names=[asset],
        horizons=[h],
        model_depth=model_depth,
        use_phase5_features=use_phase5_features,
        signal_mode=mode,
        threshold_candidates=thresholds,
        cooldown_candidates=cooldowns,
        validation_fraction=validation_fraction,
        transaction_cost=transaction_cost,
        signal_outputs={(asset, h): signal_output},
    )
    if scan_report.full_results.empty:
        raise ValueError("Candidate scanner produced no result for diagnostics")

    candidate_row = scan_report.full_results.iloc[0].to_dict()
    selected_cooldown = int(_safe_float(candidate_row.get("CooldownRows"), default=0.0))
    base_result = run_validation_locked_signal_engine(
        signal_output=signal_output,
        mode=mode,
        transaction_cost=transaction_cost,
        backtest_style="non_overlapping_realistic",
        cooldown=selected_cooldown,
        validation_fraction=validation_fraction,
        long_thresholds=thresholds,
        short_thresholds=short_thresholds,
    )

    selected = base_result.selected_threshold or {}
    selected_threshold = _safe_float(selected.get("SelectedLongThreshold"), default=np.nan)
    selected_short = _safe_float(selected.get("SelectedShortThreshold"), default=min(0.45, selected_threshold - 0.05))
    candidate_row.update(
        {
            "SelectedThreshold": selected_threshold,
            "SelectedCooldown": selected_cooldown,
            "ValidationPeriod": f"{selected.get('ValidationStart', '')} to {selected.get('ValidationEnd', '')}",
            "LockedTestPeriod": f"{selected.get('LockedTestStart', '')} to {selected.get('LockedTestEnd', '')}",
            "ValidationThresholdVerdict": selected.get("ValidationThresholdVerdict", ""),
            "LockedTestThresholdVerdict": selected.get("LockedTestThresholdVerdict", ""),
        }
    )

    trade_log = base_result.signal_frame.copy()
    trade_diagnostics = summarize_trade_diagnostics(trade_log, base_result.metrics)

    split = _chronological_validation_test_split(
        probabilities_up=signal_output.probabilities_up_test,
        future_returns=signal_output.actual_return_test,
        actual_direction=signal_output.actual_direction_test,
        test_index=signal_output.test_index,
        validation_fraction=validation_fraction,
    )
    locked = split["locked_test"]
    monthly = build_monthly_return_table(trade_log, locked["index"], locked["future_returns"], period="M")
    quarterly = build_monthly_return_table(trade_log, locked["index"], locked["future_returns"], period="Q")
    equity_curve = build_equity_drawdown_table(trade_log)
    drawdown_curve = equity_curve[["Date", "Drawdown_%"]].copy() if not equity_curve.empty else pd.DataFrame(columns=["Date", "Drawdown_%"])
    candidate_row.update(_drawdown_summary(equity_curve))

    cost_sensitivity = build_cost_sensitivity_table(
        signal_output=signal_output,
        selected_long_threshold=selected_threshold,
        selected_short_threshold=selected_short,
        selected_cooldown=selected_cooldown,
        validation_fraction=validation_fraction,
        mode=mode,
        costs=cost_values,
    )
    split_sensitivity = build_validation_split_sensitivity_table(
        signal_output=signal_output,
        asset_name=asset,
        horizon=h,
        model_depth=model_depth,
        use_phase5_features=use_phase5_features,
        mode=mode,
        threshold_candidates=thresholds,
        cooldown_candidates=cooldowns,
        split_fractions=split_fractions,
        transaction_cost=transaction_cost,
    )
    probability_diagnostics, probability_bins = _probability_diagnostics(trade_log)
    warnings = _diagnostic_warning_flags(
        candidate_row=candidate_row,
        trade_diagnostics=trade_diagnostics,
        cost_sensitivity=cost_sensitivity,
        split_sensitivity=split_sensitivity,
    )

    summary_order = [
        "Asset",
        "Horizon",
        "ModelDepth",
        "Phase5Enabled",
        "SignalMode",
        "SelectedThreshold",
        "SelectedCooldown",
        "ValidationPeriod",
        "LockedTestPeriod",
        "ValidationScore",
        "ValidationTrades",
        "ValidationWinRate_%",
        "ValidationStrategyReturn_%",
        "ValidationBuyHoldReturn_%",
        "ValidationVsBuyHold_%",
        "LockedTestTrades",
        "LockedTestWinRate_%",
        "LockedTestStrategyReturn_%",
        "LockedTestBuyHoldReturn_%",
        "LockedTestVsBuyHold_%",
        "LockedTestMaxDrawdown_%",
        "LockedTestSharpe",
        "Exposure_%",
        "RobustnessVerdict",
        "StabilityFlag",
        "FailureReason",
        "ValidationThresholdVerdict",
        "LockedTestThresholdVerdict",
        "MaxDrawdownStart",
        "MaxDrawdownEnd",
        "RecoveredAfterMaxDrawdown",
    ]
    candidate_summary = pd.DataFrame([{key: candidate_row.get(key, np.nan) for key in summary_order}])
    settings = {
        "asset": asset,
        "horizon": h,
        "model_depth": model_depth,
        "use_phase5_features": bool(use_phase5_features),
        "signal_mode": mode,
        "threshold_candidates": thresholds,
        "cooldown_candidates": cooldowns,
        "validation_fraction": float(validation_fraction),
        "transaction_cost": float(transaction_cost),
        "cost_values": [float(c) for c in cost_values],
        "split_fractions": [float(s) for s in split_fractions],
        "selection_basis": "validation_locked_threshold_and_validation_score_cooldown",
    }
    return CandidateDiagnosticsReport(
        candidate_summary=candidate_summary,
        trade_log=trade_log,
        trade_diagnostics=trade_diagnostics,
        monthly_returns=monthly,
        quarterly_returns=quarterly,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        cost_sensitivity=cost_sensitivity,
        validation_split_sensitivity=split_sensitivity,
        probability_diagnostics=probability_diagnostics,
        probability_bins=probability_bins,
        warnings=warnings,
        base_result=base_result,
        scan_report=scan_report,
        settings=settings,
    )


def build_risk_variant_grid(selected_variants: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Build the Phase 7F risk-control variant grid."""
    requested = {str(v).lower().strip().replace(" ", "_") for v in selected_variants} if selected_variants else None

    def include(name: str) -> bool:
        if requested is None:
            return True
        key = name.lower().replace(" ", "_")
        aliases = {
            "baseline_signal": {"baseline", "baseline_signal"},
            "volatility_filter": {"volatility", "volatility_filter"},
            "drawdown_stop": {"drawdown", "drawdown_stop", "drawdown_stop_rule"},
            "loss_streak_stop": {"loss_streak", "loss_streak_stop", "loss_streak_stop_rule"},
            "probability_band_filter": {"probability", "probability_band", "probability_band_filter"},
            "position_sizing": {"position", "position_sizing", "position_sizing_simulation"},
        }
        return bool(requested.intersection(aliases.get(key, {key})))

    variants: List[Dict[str, Any]] = []
    if include("baseline_signal"):
        variants.append({"RiskVariantName": "Baseline signal", "RiskVariantParams": {"type": "baseline"}})
    if include("volatility_filter"):
        for percentile in (70, 80, 90):
            variants.append(
                {
                    "RiskVariantName": "Volatility filter",
                    "RiskVariantParams": {"type": "volatility_filter", "max_vol_percentile": percentile, "lookback": 20},
                }
            )
    if include("drawdown_stop"):
        for threshold in (10, 15, 20):
            variants.append(
                {
                    "RiskVariantName": "Drawdown stop",
                    "RiskVariantParams": {"type": "drawdown_stop", "max_drawdown_pct": threshold, "resume_after_rows": 10},
                }
            )
    if include("loss_streak_stop"):
        for losses in (2, 3):
            variants.append(
                {
                    "RiskVariantName": "Loss-streak stop",
                    "RiskVariantParams": {"type": "loss_streak_stop", "max_losses": losses, "resume_after_rows": 10},
                }
            )
    if include("probability_band_filter"):
        for max_probability in (0.95, 0.98):
            variants.append(
                {
                    "RiskVariantName": "Probability band filter",
                    "RiskVariantParams": {"type": "probability_band_filter", "max_probability": max_probability},
                }
            )
    if include("position_sizing"):
        variants.extend(
            [
                {"RiskVariantName": "Position sizing", "RiskVariantParams": {"type": "position_sizing", "method": "max_cap", "max_position": 0.50}},
                {"RiskVariantName": "Position sizing", "RiskVariantParams": {"type": "position_sizing", "method": "max_cap", "max_position": 0.75}},
                {"RiskVariantName": "Position sizing", "RiskVariantParams": {"type": "position_sizing", "method": "confidence_scaled", "max_position": 1.00}},
                {"RiskVariantName": "Position sizing", "RiskVariantParams": {"type": "position_sizing", "method": "volatility_scaled", "max_position": 1.00, "lookback": 20}},
            ]
        )
    return pd.DataFrame(variants)


def _historical_safe_volatility(log_returns: np.ndarray, horizon: int, lookback: int = 20) -> np.ndarray:
    """Trailing volatility proxy using only returns whose horizon has already completed."""
    ret = np.expm1(np.asarray(log_returns, dtype=float).flatten())
    h = int(max(1, horizon))
    safe = np.full(len(ret), np.nan, dtype=float)
    for i in range(len(ret)):
        known_end = i - h
        if known_end < 0:
            continue
        start = max(0, known_end - int(max(2, lookback)) + 1)
        hist = ret[start : known_end + 1]
        hist = hist[np.isfinite(hist)]
        if len(hist) >= 2:
            safe[i] = float(np.std(hist, ddof=1))
    return safe


def _derive_risk_variant_state(
    *,
    variant_params: Dict[str, Any],
    probabilities_up: Iterable[float],
    future_returns: Iterable[float],
    horizon: int,
) -> Dict[str, Any]:
    params = dict(variant_params or {})
    kind = str(params.get("type", "baseline"))
    log_returns = np.asarray(future_returns, dtype=float).flatten()
    vols = _historical_safe_volatility(log_returns, horizon, int(params.get("lookback", 20)))
    finite_vols = vols[np.isfinite(vols)]
    state: Dict[str, Any] = {}
    if kind == "volatility_filter":
        percentile = float(params.get("max_vol_percentile", 80))
        state["VolatilityCutoff"] = float(np.percentile(finite_vols, percentile)) if len(finite_vols) else np.inf
    if kind == "position_sizing" and params.get("method") == "volatility_scaled":
        state["TargetVolatility"] = float(np.median(finite_vols)) if len(finite_vols) else np.nan
    return state


def _variant_param_text(params: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> str:
    merged = dict(params or {})
    for key, value in (state or {}).items():
        if isinstance(value, float):
            merged[key] = round(value, 6) if np.isfinite(value) else value
        else:
            merged[key] = value
    return str(merged)


def _profit_factor_from_returns(returns: np.ndarray) -> float:
    wins = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    gross_profit = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) else 0.0
    if gross_loss > 0.0:
        return gross_profit / gross_loss
    return np.inf if gross_profit > 0.0 else 0.0


def apply_risk_control_variant(
    *,
    probabilities_up: Iterable[float],
    future_returns: Iterable[float],
    actual_direction: Iterable[int],
    test_index: Optional[Iterable[Any]] = None,
    asset: str = "",
    horizon: int = 1,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    mode: str = "long_only",
    cooldown: int = 0,
    transaction_cost: float = 0.001,
    risk_variant: Optional[Dict[str, Any]] = None,
    learned_state: Optional[Dict[str, Any]] = None,
    baseline_direction_accuracy: float = np.nan,
) -> SignalBacktestResult:
    """Apply one risk-control variant to a non-overlapping trade simulation."""
    variant = risk_variant or {"RiskVariantName": "Baseline signal", "RiskVariantParams": {"type": "baseline"}}
    params = dict(variant.get("RiskVariantParams", variant.get("params", {})) or {})
    kind = str(params.get("type", "baseline"))
    state = dict(learned_state or {})

    p_up, log_returns, y_dir, index = _aligned_signal_arrays(
        probabilities_up=probabilities_up,
        future_returns=future_returns,
        actual_direction=actual_direction,
        test_index=test_index,
    )
    h = int(max(1, horizon))
    cooldown_rows = int(max(0, cooldown))
    signals = generate_signals(p_up, long_threshold=long_threshold, short_threshold=short_threshold, mode=mode)
    simple_returns = np.expm1(log_returns)
    safe_vol = _historical_safe_volatility(log_returns, h, int(params.get("lookback", 20)))

    potential_signals = int(np.sum((signals != 0) & (np.arange(len(signals)) + h < len(signals))))
    trades: List[Dict[str, Any]] = []
    equity_value = 1.0
    peak_equity = 1.0
    loss_streak = 0
    pause_until_row = -1
    i = 0

    while i < len(signals):
        signal = int(signals[i])
        if signal == 0:
            i += 1
            continue
        exit_i = i + h
        if exit_i >= len(signals):
            break

        if i < pause_until_row:
            i += 1
            continue

        probability = float(p_up[i])
        variant_note = ""
        position_size = 1.0

        if kind == "volatility_filter":
            cutoff = _safe_float(state.get("VolatilityCutoff"), default=np.inf)
            current_vol = _safe_float(safe_vol[i], default=np.nan)
            if np.isfinite(current_vol) and current_vol > cutoff:
                i += 1
                continue
        elif kind == "probability_band_filter":
            max_probability = _safe_float(params.get("max_probability"), default=1.0)
            if probability > max_probability:
                i += 1
                continue
        elif kind == "position_sizing":
            method = str(params.get("method", "max_cap"))
            cap = float(params.get("max_position", 1.0))
            if method == "max_cap":
                position_size = cap
            elif method == "confidence_scaled":
                if signal == 1:
                    raw_size = (probability - float(long_threshold)) / max(1.0 - float(long_threshold), 1e-9)
                else:
                    raw_size = (float(short_threshold) - probability) / max(float(short_threshold), 1e-9)
                position_size = min(cap, max(0.10, raw_size))
            elif method == "volatility_scaled":
                target_vol = _safe_float(state.get("TargetVolatility"), default=np.nan)
                current_vol = _safe_float(safe_vol[i], default=np.nan)
                if np.isfinite(target_vol) and np.isfinite(current_vol) and current_vol > 0.0:
                    position_size = min(cap, max(0.10, target_vol / current_vol))
                else:
                    position_size = cap
            variant_note = f"position_size={position_size:.4f}"

        realized = float(simple_returns[i])
        gross_strategy = float(signal * realized * position_size)
        strategy_after_cost = gross_strategy - float(transaction_cost) * 2.0 * abs(position_size)
        equity_value *= 1.0 + strategy_after_cost
        peak_equity = max(peak_equity, equity_value)
        drawdown = equity_value / peak_equity - 1.0
        win = strategy_after_cost > 0.0
        loss_streak = 0 if win else loss_streak + 1

        trades.append(
            {
                "EntryRow": int(i),
                "ExitRow": int(exit_i),
                "EntryDate": index[i],
                "ExitDate": index[exit_i],
                "Asset": asset,
                "Horizon": h,
                "Signal": signal,
                "ProbabilityUp": probability,
                "EntryReturnTarget": float(log_returns[i]),
                "RealizedReturn": realized,
                "PositionSize": float(position_size),
                "StrategyReturnAfterCost": strategy_after_cost,
                "HoldingDays": h,
                "Win/Loss": "Win" if win else "Loss",
                "LongThreshold": float(long_threshold),
                "ShortThreshold": float(short_threshold),
                "Mode": _normalize_mode(mode),
                "BacktestStyle": "non_overlapping_realistic",
                "RiskVariantName": variant.get("RiskVariantName", kind),
                "RiskVariantParams": _variant_param_text(params, state),
                "VariantNote": variant_note,
                "EquityAfterTrade": equity_value,
                "DrawdownAfterTrade_%": drawdown * 100.0,
            }
        )

        if kind == "drawdown_stop":
            threshold = float(params.get("max_drawdown_pct", 15.0)) / 100.0
            if drawdown <= -threshold:
                pause_until_row = exit_i + int(params.get("resume_after_rows", 10))
        elif kind == "loss_streak_stop":
            if loss_streak >= int(params.get("max_losses", 2)):
                pause_until_row = exit_i + int(params.get("resume_after_rows", 10))
                loss_streak = 0

        i = exit_i + 1 + cooldown_rows

    trade_log = pd.DataFrame(trades)
    trade_returns = trade_log["StrategyReturnAfterCost"].astype(float).to_numpy() if not trade_log.empty else np.array([], dtype=float)
    position_sizes = trade_log["PositionSize"].astype(float).to_numpy() if not trade_log.empty else np.array([], dtype=float)
    if len(trade_returns):
        equity = pd.Series((1.0 + trade_returns).cumprod(), index=pd.to_datetime(trade_log["ExitDate"]))
        strategy_total = float(equity.iloc[-1] - 1.0)
        win_rate = float(np.mean(trade_returns > 0.0) * 100.0)
        max_dd = _max_drawdown_pct(equity)
        exposure = float(min(100.0, np.sum(trade_log["HoldingDays"].astype(float).to_numpy() * np.abs(position_sizes)) / max(len(p_up), 1) * 100.0))
        avg_trade = float(np.mean(trade_returns) * 100.0)
        median_trade = float(np.median(trade_returns) * 100.0)
        best_trade = float(np.max(trade_returns) * 100.0)
        worst_trade = float(np.min(trade_returns) * 100.0)
        entry_positions = trade_log["EntryRow"].astype(int).to_numpy()
        trade_signals = trade_log["Signal"].astype(int).to_numpy()
        trade_dirs = y_dir[entry_positions]
        correct = ((trade_signals == 1) & (trade_dirs == 1)) | ((trade_signals == -1) & (trade_dirs == 0))
        active_dir_acc = float(np.mean(correct) * 100.0)
    else:
        strategy_total = win_rate = max_dd = exposure = avg_trade = median_trade = best_trade = worst_trade = active_dir_acc = 0.0

    possible_bh_entries = np.arange(0, max(len(simple_returns) - h, 0), h)
    buy_hold_returns = simple_returns[possible_bh_entries] if len(possible_bh_entries) else np.array([], dtype=float)
    buy_hold_total = float(np.prod(1.0 + buy_hold_returns) - 1.0) if len(buy_hold_returns) else 0.0
    strategy_vs_bh = (strategy_total - buy_hold_total) * 100.0
    baseline_acc = _safe_float(baseline_direction_accuracy)
    verdict, warnings = _threshold_verdict(
        signal_count=int(len(trade_log)),
        total_rows=len(p_up),
        strategy_vs_buy_hold_pct=strategy_vs_bh,
        active_direction_accuracy=active_dir_acc,
        baseline_direction_accuracy=baseline_acc,
        win_rate=win_rate,
    )

    metrics: Dict[str, Any] = {
        "Mode": _normalize_mode(mode),
        "BacktestStyle": "non_overlapping_realistic",
        "RiskVariantName": variant.get("RiskVariantName", kind),
        "RiskVariantParams": _variant_param_text(params, state),
        "LongThreshold": round(float(long_threshold), 4),
        "ShortThreshold": round(float(short_threshold), 4),
        "TransactionCost_%": round(float(transaction_cost) * 100.0, 4),
        "CooldownRows": cooldown_rows,
        "Rows": int(len(p_up)),
        "PotentialSignals": potential_signals,
        "NumberOfTrades": int(len(trade_log)),
        "SignalCount": int(len(trade_log)),
        "TradesRemoved": int(max(potential_signals - len(trade_log), 0)),
        "TradeFrequency_%": round(len(trade_log) / max(len(p_up), 1) * 100.0, 2),
        "SignalFrequency_%": round(len(trade_log) / max(len(p_up), 1) * 100.0, 2),
        "WinRate_%": round(win_rate, 2),
        "WinRateActive_%": round(win_rate, 2),
        "AverageTradeReturn_%": round(avg_trade, 4),
        "MedianTradeReturn_%": round(median_trade, 4),
        "TotalCompoundedReturn_%": round(strategy_total * 100.0, 4),
        "StrategyTotalReturn_%": round(strategy_total * 100.0, 4),
        "BuyHoldReturn_%": round(buy_hold_total * 100.0, 4),
        "StrategyMinusBuyHold_%": round(strategy_vs_bh, 4),
        "Sharpe": round(_sharpe_ratio(trade_returns, h), 4),
        "MaxDrawdown_%": round(max_dd, 4),
        "BestTrade_%": round(best_trade, 4),
        "WorstTrade_%": round(worst_trade, 4),
        "ProfitFactor": round(float(_profit_factor_from_returns(trade_returns)), 4) if len(trade_returns) else 0.0,
        "Exposure_%": round(exposure, 2),
        "DirectionAccuracyActive_%": round(active_dir_acc, 2),
        "BaselineDirectionAccuracy_%": round(baseline_acc, 2) if np.isfinite(baseline_acc) else np.nan,
        "ThresholdVerdict": verdict,
        "Verdict": verdict,
        "Warnings": warnings,
    }
    return SignalBacktestResult(metrics=metrics, signal_frame=trade_log)


def score_risk_variant_on_validation(
    validation_metrics: Dict[str, Any],
    baseline_validation_metrics: Dict[str, Any],
) -> Dict[str, float]:
    """Validation-only score for selecting risk-control variants."""
    val_vs = _metric_value(validation_metrics, ["StrategyMinusBuyHold_%"], default=-100.0)
    val_return = _metric_value(validation_metrics, ["TotalCompoundedReturn_%", "StrategyTotalReturn_%"], default=-100.0)
    val_dd = _metric_value(validation_metrics, ["MaxDrawdown_%"], default=0.0)
    val_sharpe = _metric_value(validation_metrics, ["Sharpe"], default=0.0)
    trades = _metric_value(validation_metrics, ["NumberOfTrades", "SignalCount"], default=0.0)
    rows = int(_metric_value(validation_metrics, ["Rows"], default=0.0))
    trades_removed = _metric_value(validation_metrics, ["TradesRemoved"], default=0.0)
    potential = _metric_value(validation_metrics, ["PotentialSignals"], default=max(trades, 1.0))

    base_return = _metric_value(baseline_validation_metrics, ["TotalCompoundedReturn_%", "StrategyTotalReturn_%"], default=0.0)
    base_dd = _metric_value(baseline_validation_metrics, ["MaxDrawdown_%"], default=0.0)
    base_sharpe = _metric_value(baseline_validation_metrics, ["Sharpe"], default=0.0)
    drawdown_improvement = abs(min(base_dd, 0.0)) - abs(min(val_dd, 0.0))
    return_change = val_return - base_return
    sharpe_change = val_sharpe - base_sharpe
    min_trades = _minimum_trade_count(rows)
    low_trade_penalty = max(0.0, min_trades - trades) * 8.0
    overfiltered_penalty = 30.0 if potential > 0.0 and trades_removed / max(potential, 1.0) > 0.70 else 0.0
    edge_penalty = 50.0 if val_vs <= 0.0 else 0.0
    destroyed_penalty = 25.0 if drawdown_improvement > 2.0 and val_vs <= 0.0 else 0.0
    score = val_vs + drawdown_improvement * 2.0 + sharpe_change * 5.0 + min(return_change, 0.0) * 0.35
    score -= low_trade_penalty + overfiltered_penalty + edge_penalty + destroyed_penalty
    return {
        "RiskValidationScore": round(float(score), 4),
        "ValidationDrawdownImprovement_%": round(float(drawdown_improvement), 4),
        "ValidationReturnChange_%": round(float(return_change), 4),
        "ValidationSharpeChange": round(float(sharpe_change), 4),
        "ValidationLowTradePenalty": round(float(low_trade_penalty), 4),
        "ValidationOverFilteredPenalty": round(float(overfiltered_penalty), 4),
    }


def evaluate_locked_risk_variant(
    *,
    signal_output: Any,
    selected_long_threshold: float,
    selected_short_threshold: float,
    selected_cooldown: int,
    validation_fraction: float,
    risk_variant: Dict[str, Any],
    learned_state: Optional[Dict[str, Any]] = None,
    mode: str = "long_only",
    transaction_cost: float = 0.001,
) -> SignalBacktestResult:
    split = _chronological_validation_test_split(
        probabilities_up=signal_output.probabilities_up_test,
        future_returns=signal_output.actual_return_test,
        actual_direction=signal_output.actual_direction_test,
        test_index=signal_output.test_index,
        validation_fraction=validation_fraction,
    )
    locked = split["locked_test"]
    locked_baseline = _segment_direction_baseline_accuracy(locked["actual_direction"])
    return apply_risk_control_variant(
        probabilities_up=locked["probabilities_up"],
        future_returns=locked["future_returns"],
        actual_direction=locked["actual_direction"],
        test_index=locked["index"],
        asset=getattr(signal_output, "asset", ""),
        horizon=int(signal_output.horizon),
        long_threshold=selected_long_threshold,
        short_threshold=selected_short_threshold,
        mode=mode,
        cooldown=selected_cooldown,
        transaction_cost=transaction_cost,
        risk_variant=risk_variant,
        learned_state=learned_state,
        baseline_direction_accuracy=locked_baseline,
    )


def _risk_metric(metrics: Dict[str, Any], names: Iterable[str], default: float = np.nan) -> float:
    return _metric_value(metrics, names, default=default)


def risk_control_verdict(
    *,
    locked_metrics: Dict[str, Any],
    baseline_locked_metrics: Dict[str, Any],
) -> Dict[str, str]:
    locked_vs = _risk_metric(locked_metrics, ["StrategyMinusBuyHold_%"], default=np.nan)
    locked_dd = _risk_metric(locked_metrics, ["MaxDrawdown_%"], default=0.0)
    locked_return = _risk_metric(locked_metrics, ["TotalCompoundedReturn_%", "StrategyTotalReturn_%"], default=np.nan)
    locked_trades = int(_risk_metric(locked_metrics, ["NumberOfTrades", "SignalCount"], default=0.0))
    locked_rows = int(_risk_metric(locked_metrics, ["Rows"], default=0.0))
    baseline_vs = _risk_metric(baseline_locked_metrics, ["StrategyMinusBuyHold_%"], default=np.nan)
    baseline_dd = _risk_metric(baseline_locked_metrics, ["MaxDrawdown_%"], default=0.0)
    baseline_return = _risk_metric(baseline_locked_metrics, ["TotalCompoundedReturn_%", "StrategyTotalReturn_%"], default=np.nan)
    baseline_trades = int(_risk_metric(baseline_locked_metrics, ["NumberOfTrades", "SignalCount"], default=0.0))

    dd_improvement = abs(min(baseline_dd, 0.0)) - abs(min(locked_dd, 0.0))
    return_change = locked_return - baseline_return if np.isfinite(locked_return) and np.isfinite(baseline_return) else np.nan
    reasons: List[str] = []
    min_trades = _minimum_trade_count(locked_rows)
    if locked_trades < min_trades:
        reasons.append(f"LowTradeCount: locked test has {locked_trades} trades; minimum is {min_trades}")
    if np.isfinite(locked_vs) and locked_vs <= 0.0:
        reasons.append("locked test fails buy-and-hold")
    if baseline_trades > 0 and locked_trades / max(baseline_trades, 1) < 0.30:
        reasons.append("OverFiltered: risk control removed more than 70% of baseline trades")
    if dd_improvement > 3.0 and np.isfinite(locked_vs) and locked_vs <= 1.0:
        reasons.append("ReturnDestroyed: drawdown improved but edge disappeared")
    if dd_improvement <= 1.0:
        reasons.append("NoImprovement: locked-test drawdown did not improve meaningfully")
    if locked_dd <= -20.0:
        reasons.append(f"DrawdownRisk: locked-test max drawdown is {locked_dd:.2f}%")

    if locked_trades < min_trades or (np.isfinite(locked_vs) and locked_vs <= 0.0):
        verdict = "Do not trust"
    elif dd_improvement >= 5.0 and np.isfinite(locked_vs) and locked_vs > 0.0:
        verdict = "Improved research candidate"
    elif dd_improvement >= 3.0 and np.isfinite(locked_vs) and locked_vs > 0.0 and np.isfinite(baseline_vs) and locked_vs < max(1.0, baseline_vs * 0.35):
        verdict = "Safer but weaker"
    elif dd_improvement >= 3.0 and np.isfinite(locked_vs) and locked_vs > 0.0:
        verdict = "Safer but weaker"
    else:
        verdict = "No improvement"

    return {
        "RiskControlVerdict": verdict,
        "FailureReason": "; ".join(dict.fromkeys(reasons)),
    }


def _risk_control_row(
    *,
    asset: str,
    horizon: int,
    model_depth: str,
    use_phase5_features: bool,
    mode: str,
    base_threshold: float,
    base_cooldown: int,
    variant: Dict[str, Any],
    validation_metrics: Dict[str, Any],
    baseline_validation_metrics: Dict[str, Any],
    locked_metrics: Optional[Dict[str, Any]] = None,
    baseline_locked_metrics: Optional[Dict[str, Any]] = None,
    learned_state: Optional[Dict[str, Any]] = None,
    cost_fragility_flag: str = "",
    locked_status: str = "Not evaluated on locked test",
) -> Dict[str, Any]:
    locked_metrics = locked_metrics or {}
    baseline_locked_metrics = baseline_locked_metrics or {}
    score = score_risk_variant_on_validation(validation_metrics, baseline_validation_metrics)
    verdict = {"RiskControlVerdict": locked_status, "FailureReason": ""}
    return_change = np.nan
    dd_improvement = np.nan
    sharpe_change = np.nan
    if locked_metrics and baseline_locked_metrics:
        verdict = risk_control_verdict(locked_metrics=locked_metrics, baseline_locked_metrics=baseline_locked_metrics)
        return_change = _risk_metric(locked_metrics, ["TotalCompoundedReturn_%", "StrategyTotalReturn_%"]) - _risk_metric(
            baseline_locked_metrics, ["TotalCompoundedReturn_%", "StrategyTotalReturn_%"]
        )
        dd_improvement = abs(min(_risk_metric(baseline_locked_metrics, ["MaxDrawdown_%"], default=0.0), 0.0)) - abs(
            min(_risk_metric(locked_metrics, ["MaxDrawdown_%"], default=0.0), 0.0)
        )
        sharpe_change = _risk_metric(locked_metrics, ["Sharpe"], default=0.0) - _risk_metric(baseline_locked_metrics, ["Sharpe"], default=0.0)

    row = {
        "Asset": asset,
        "Horizon": int(horizon),
        "ModelDepth": model_depth,
        "Phase5Enabled": bool(use_phase5_features),
        "SignalMode": _normalize_mode(mode),
        "BaseThreshold": round(float(base_threshold), 4),
        "BaseCooldown": int(base_cooldown),
        "RiskVariantName": variant.get("RiskVariantName", ""),
        "RiskVariantParams": _variant_param_text(variant.get("RiskVariantParams", {}), learned_state),
        "ValidationStrategyReturn_%": _risk_metric(validation_metrics, ["TotalCompoundedReturn_%", "StrategyTotalReturn_%"]),
        "ValidationBuyHoldReturn_%": _risk_metric(validation_metrics, ["BuyHoldReturn_%"]),
        "ValidationVsBuyHold_%": _risk_metric(validation_metrics, ["StrategyMinusBuyHold_%"]),
        "ValidationMaxDrawdown_%": _risk_metric(validation_metrics, ["MaxDrawdown_%"]),
        "ValidationSharpe": _risk_metric(validation_metrics, ["Sharpe"]),
        "ValidationTrades": int(_risk_metric(validation_metrics, ["NumberOfTrades", "SignalCount"], default=0.0)),
        "ValidationExposure_%": _risk_metric(validation_metrics, ["Exposure_%"]),
        "LockedStrategyReturn_%": _risk_metric(locked_metrics, ["TotalCompoundedReturn_%", "StrategyTotalReturn_%"]),
        "LockedBuyHoldReturn_%": _risk_metric(locked_metrics, ["BuyHoldReturn_%"]),
        "LockedVsBuyHold_%": _risk_metric(locked_metrics, ["StrategyMinusBuyHold_%"]),
        "LockedMaxDrawdown_%": _risk_metric(locked_metrics, ["MaxDrawdown_%"]),
        "LockedSharpe": _risk_metric(locked_metrics, ["Sharpe"]),
        "LockedTrades": int(_risk_metric(locked_metrics, ["NumberOfTrades", "SignalCount"], default=0.0)) if locked_metrics else np.nan,
        "LockedExposure_%": _risk_metric(locked_metrics, ["Exposure_%"]),
        "ReturnChangeVsBaseline_%": round(float(return_change), 4) if np.isfinite(return_change) else np.nan,
        "DrawdownImprovementVsBaseline_%": round(float(dd_improvement), 4) if np.isfinite(dd_improvement) else np.nan,
        "SharpeChangeVsBaseline": round(float(sharpe_change), 4) if np.isfinite(sharpe_change) else np.nan,
        "TradesRemoved": int(_risk_metric(validation_metrics, ["TradesRemoved"], default=0.0)),
        "CostFragilityFlag": cost_fragility_flag,
        "RiskControlVerdict": verdict["RiskControlVerdict"],
        "FailureReason": verdict["FailureReason"],
        "RiskValidationScore": score["RiskValidationScore"],
        "SelectedByValidation": False,
        "LockedEvaluationStatus": locked_status,
    }
    return row


def _cost_fragility_from_table(cost_table: pd.DataFrame, variant_name: str) -> str:
    if cost_table is None or cost_table.empty:
        return ""
    df = cost_table[cost_table["RiskVariantName"].eq(variant_name)] if "RiskVariantName" in cost_table.columns else cost_table
    if df.empty:
        return ""
    low = df[df["TransactionCost_%"].astype(float).le(0.0001)]
    realistic = df[df["TransactionCost_%"].astype(float).ge(0.10)]
    low_ok = bool(not low.empty and _safe_float(low.iloc[0].get("VsBuyHold_%"), default=-np.inf) > 0.0)
    realistic_fails = bool(not realistic.empty and (realistic["VsBuyHold_%"].astype(float) <= 0.0).any())
    return "CostFragile" if low_ok and realistic_fails else ""


def _build_risk_cost_stress_table(
    *,
    signal_output: Any,
    selected_long_threshold: float,
    selected_short_threshold: float,
    selected_cooldown: int,
    validation_fraction: float,
    mode: str,
    baseline_variant: Dict[str, Any],
    selected_variant: Dict[str, Any],
    selected_state: Dict[str, Any],
    costs: Iterable[float],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for variant, state in [(baseline_variant, {}), (selected_variant, selected_state)]:
        variant_name = str(variant.get("RiskVariantName", ""))
        for cost in costs:
            result = evaluate_locked_risk_variant(
                signal_output=signal_output,
                selected_long_threshold=selected_long_threshold,
                selected_short_threshold=selected_short_threshold,
                selected_cooldown=selected_cooldown,
                validation_fraction=validation_fraction,
                risk_variant=variant,
                learned_state=state,
                mode=mode,
                transaction_cost=float(cost),
            )
            metrics = result.metrics
            rows.append(
                {
                    "RiskVariantName": variant_name,
                    "RiskVariantParams": _variant_param_text(variant.get("RiskVariantParams", {}), state),
                    "TransactionCost_%": round(float(cost) * 100.0, 4),
                    "Trades": int(metrics.get("NumberOfTrades", metrics.get("SignalCount", 0))),
                    "StrategyReturn_%": metrics.get("TotalCompoundedReturn_%", metrics.get("StrategyTotalReturn_%", np.nan)),
                    "BuyHoldReturn_%": metrics.get("BuyHoldReturn_%", np.nan),
                    "VsBuyHold_%": metrics.get("StrategyMinusBuyHold_%", np.nan),
                    "MaxDrawdown_%": metrics.get("MaxDrawdown_%", np.nan),
                    "Sharpe": metrics.get("Sharpe", np.nan),
                    "Verdict": metrics.get("ThresholdVerdict", metrics.get("Verdict", "")),
                }
            )
    return pd.DataFrame(rows)


def summarize_risk_control_results(results: pd.DataFrame) -> Dict[str, Any]:
    if results is None or results.empty:
        empty = pd.DataFrame(columns=RISK_CONTROL_COLUMNS)
        return {"baseline_vs_best": empty, "warnings": ["No risk-control results were produced."]}
    baseline = results[results["RiskVariantName"].eq("Baseline signal")]
    selected = results[results["SelectedByValidation"].eq(True)]
    baseline_vs_best = pd.concat([baseline.head(1), selected.head(1)]).drop_duplicates()
    warnings: List[str] = []
    if selected.empty:
        warnings.append("NoImprovement: no risk-control variant was selected.")
    else:
        row = selected.iloc[0]
        failure = str(row.get("FailureReason", "") or "")
        if "CostFragile" in str(row.get("CostFragilityFlag", "")):
            warnings.append("CostFragile: selected variant only survives at unrealistically low cost.")
        for key in ("DrawdownRisk", "LowTradeCount", "OverFiltered", "ReturnDestroyed", "NoImprovement"):
            if key in failure:
                warnings.append(failure if failure.startswith(key) else f"{key}: {failure}")
    return {"baseline_vs_best": baseline_vs_best, "warnings": list(dict.fromkeys(warnings))}


def run_risk_controlled_candidate_upgrade(
    *,
    raw_df: Optional[pd.DataFrame] = None,
    asset_name: str = "Silver",
    horizon: int = 5,
    model_depth: str = "core",
    use_phase5_features: bool = True,
    signal_mode: str = "long_only",
    threshold_candidates: Iterable[float] = (0.50, 0.55, 0.60, 0.65, 0.70),
    cooldown_candidates: Iterable[int] = (0, 2, 5),
    validation_fraction: float = 0.5,
    transaction_cost: float = 0.001,
    risk_variant_names: Optional[Iterable[str]] = None,
    cost_values: Iterable[float] = (0.0, 0.0005, 0.001, 0.002, 0.005),
    signal_output: Optional[Any] = None,
    signal_output_factory: Optional[Any] = None,
) -> RiskControlUpgradeReport:
    """Run Phase 7F validation-only risk-control selection and locked-test evaluation."""
    if signal_output is None:
        signal_output = _build_signal_output(
            raw_df=raw_df,
            asset=asset_name,
            horizon=int(horizon),
            model_depth=model_depth,
            use_phase5_features=use_phase5_features,
            signal_outputs=None,
            signal_output_factory=signal_output_factory,
        )

    asset = str(asset_name or getattr(signal_output, "asset", ""))
    h = int(horizon or getattr(signal_output, "horizon", 1))
    thresholds = [float(t) for t in threshold_candidates]
    cooldowns = [int(max(0, c)) for c in cooldown_candidates]
    mode = _normalize_mode(signal_mode)
    short_thresholds = [max(0.0, min(0.49, 1.0 - t)) for t in thresholds] if mode != "long_only" else [min(0.45, t - 0.05) for t in thresholds]

    scan_report = run_signal_research_scan(
        asset_names=[asset],
        horizons=[h],
        model_depth=model_depth,
        use_phase5_features=use_phase5_features,
        signal_mode=mode,
        threshold_candidates=thresholds,
        cooldown_candidates=cooldowns,
        validation_fraction=validation_fraction,
        transaction_cost=transaction_cost,
        signal_outputs={(asset, h): signal_output},
    )
    if scan_report.full_results.empty:
        raise ValueError("Baseline signal scanner produced no candidate")

    baseline_scan_row = scan_report.full_results.iloc[0].to_dict()
    selected_cooldown = int(_safe_float(baseline_scan_row.get("CooldownRows"), default=0.0))
    baseline_result = run_validation_locked_signal_engine(
        signal_output=signal_output,
        mode=mode,
        transaction_cost=transaction_cost,
        backtest_style="non_overlapping_realistic",
        cooldown=selected_cooldown,
        validation_fraction=validation_fraction,
        long_thresholds=thresholds,
        short_thresholds=short_thresholds,
    )
    selected = baseline_result.selected_threshold or {}
    selected_long = _safe_float(selected.get("SelectedLongThreshold"), default=np.nan)
    selected_short = _safe_float(selected.get("SelectedShortThreshold"), default=min(0.45, selected_long - 0.05))

    split = _chronological_validation_test_split(
        probabilities_up=signal_output.probabilities_up_test,
        future_returns=signal_output.actual_return_test,
        actual_direction=signal_output.actual_direction_test,
        test_index=signal_output.test_index,
        validation_fraction=validation_fraction,
    )
    validation = split["validation"]
    validation_baseline = _segment_direction_baseline_accuracy(validation["actual_direction"])

    baseline_variant = {"RiskVariantName": "Baseline signal", "RiskVariantParams": {"type": "baseline"}}
    baseline_validation = apply_risk_control_variant(
        probabilities_up=validation["probabilities_up"],
        future_returns=validation["future_returns"],
        actual_direction=validation["actual_direction"],
        test_index=validation["index"],
        asset=asset,
        horizon=h,
        long_threshold=selected_long,
        short_threshold=selected_short,
        mode=mode,
        cooldown=selected_cooldown,
        transaction_cost=transaction_cost,
        risk_variant=baseline_variant,
        learned_state={},
        baseline_direction_accuracy=validation_baseline,
    )

    variant_grid = build_risk_variant_grid(risk_variant_names)
    if variant_grid.empty:
        variant_grid = build_risk_variant_grid(["baseline"])
    if "Baseline signal" not in set(variant_grid["RiskVariantName"]):
        variant_grid = pd.concat([build_risk_variant_grid(["baseline"]), variant_grid], ignore_index=True)

    validation_rows: List[Dict[str, Any]] = []
    variant_payloads: List[Dict[str, Any]] = []
    for _, variant_row in variant_grid.iterrows():
        variant = {
            "RiskVariantName": variant_row["RiskVariantName"],
            "RiskVariantParams": dict(variant_row["RiskVariantParams"]),
        }
        state = _derive_risk_variant_state(
            variant_params=variant["RiskVariantParams"],
            probabilities_up=validation["probabilities_up"],
            future_returns=validation["future_returns"],
            horizon=h,
        )
        validation_result = apply_risk_control_variant(
            probabilities_up=validation["probabilities_up"],
            future_returns=validation["future_returns"],
            actual_direction=validation["actual_direction"],
            test_index=validation["index"],
            asset=asset,
            horizon=h,
            long_threshold=selected_long,
            short_threshold=selected_short,
            mode=mode,
            cooldown=selected_cooldown,
            transaction_cost=transaction_cost,
            risk_variant=variant,
            learned_state=state,
            baseline_direction_accuracy=validation_baseline,
        )
        row = _risk_control_row(
            asset=asset,
            horizon=h,
            model_depth=model_depth,
            use_phase5_features=use_phase5_features,
            mode=mode,
            base_threshold=selected_long,
            base_cooldown=selected_cooldown,
            variant=variant,
            validation_metrics=validation_result.metrics,
            baseline_validation_metrics=baseline_validation.metrics,
            locked_metrics={},
            baseline_locked_metrics={},
            learned_state=state,
        )
        validation_rows.append(row)
        variant_payloads.append({"variant": variant, "state": state, "validation_result": validation_result, "row": row})

    ordered_payloads = sorted(
        variant_payloads,
        key=lambda item: (
            -_safe_float(item["row"].get("RiskValidationScore"), default=-np.inf),
            -_safe_float(item["row"].get("ValidationTrades"), default=-np.inf),
        ),
    )
    selected_payload = ordered_payloads[0]
    selected_variant = selected_payload["variant"]
    selected_state = selected_payload["state"]

    selected_locked_result = evaluate_locked_risk_variant(
        signal_output=signal_output,
        selected_long_threshold=selected_long,
        selected_short_threshold=selected_short,
        selected_cooldown=selected_cooldown,
        validation_fraction=validation_fraction,
        risk_variant=selected_variant,
        learned_state=selected_state,
        mode=mode,
        transaction_cost=transaction_cost,
    )

    cost_stress = _build_risk_cost_stress_table(
        signal_output=signal_output,
        selected_long_threshold=selected_long,
        selected_short_threshold=selected_short,
        selected_cooldown=selected_cooldown,
        validation_fraction=validation_fraction,
        mode=mode,
        baseline_variant=baseline_variant,
        selected_variant=selected_variant,
        selected_state=selected_state,
        costs=cost_values,
    )
    selected_cost_fragility = _cost_fragility_from_table(cost_stress, str(selected_variant.get("RiskVariantName", "")))

    final_rows: List[Dict[str, Any]] = []
    for payload in variant_payloads:
        variant = payload["variant"]
        state = payload["state"]
        validation_metrics = payload["validation_result"].metrics
        is_baseline = variant.get("RiskVariantName") == "Baseline signal"
        is_selected = payload is selected_payload
        locked_metrics = baseline_result.metrics if is_baseline else selected_locked_result.metrics if is_selected else {}
        baseline_locked_metrics = baseline_result.metrics if (is_baseline or is_selected) else {}
        status = "Baseline locked reference" if is_baseline else "Selected by validation; locked test evaluated once" if is_selected else "Not evaluated on locked test"
        row = _risk_control_row(
            asset=asset,
            horizon=h,
            model_depth=model_depth,
            use_phase5_features=use_phase5_features,
            mode=mode,
            base_threshold=selected_long,
            base_cooldown=selected_cooldown,
            variant=variant,
            validation_metrics=validation_metrics,
            baseline_validation_metrics=baseline_validation.metrics,
            locked_metrics=locked_metrics,
            baseline_locked_metrics=baseline_locked_metrics,
            learned_state=state,
            cost_fragility_flag=selected_cost_fragility if is_selected else "",
            locked_status=status,
        )
        row["SelectedByValidation"] = bool(is_selected)
        final_rows.append(row)

    full_table = pd.DataFrame(final_rows)
    for col in RISK_CONTROL_COLUMNS:
        if col not in full_table.columns:
            full_table[col] = np.nan
    leading = RISK_CONTROL_COLUMNS
    full_table = full_table[leading + [c for c in full_table.columns if c not in leading]]
    summary = summarize_risk_control_results(full_table)
    warnings = summary["warnings"]
    if selected_cost_fragility:
        warnings.append("CostFragile: selected variant only survives at unrealistically low cost.")
    try:
        split_sensitivity = build_validation_split_sensitivity_table(
            signal_output=signal_output,
            asset_name=asset,
            horizon=h,
            model_depth=model_depth,
            use_phase5_features=use_phase5_features,
            mode=mode,
            threshold_candidates=thresholds,
            cooldown_candidates=cooldowns,
            split_fractions=(0.40, 0.50, 0.60),
            transaction_cost=transaction_cost,
        )
        if not split_sensitivity.empty:
            split_thresholds = split_sensitivity["SelectedThreshold"].dropna().astype(float)
            split_cooldowns = split_sensitivity["SelectedCooldown"].dropna().astype(float)
            split_vs = split_sensitivity["LockedTestVsBuyHold_%"].dropna().astype(float)
            split_reasons: List[str] = []
            if len(split_thresholds) and split_thresholds.max() - split_thresholds.min() >= 0.10:
                split_reasons.append("threshold changes materially")
            if split_cooldowns.nunique() > 1:
                split_reasons.append("cooldown changes")
            if len(split_vs) and (split_vs.max() - split_vs.min() >= 15.0 or (split_vs.gt(0).any() and split_vs.le(0).any())):
                split_reasons.append("locked-test edge changes heavily")
            if split_reasons:
                warnings.append(f"SplitUnstable: {', '.join(split_reasons)}.")
    except Exception as exc:
        warnings.append(f"SplitUnstable check unavailable: {exc}")
    warnings = list(dict.fromkeys([w for w in warnings if w]))
    settings = {
        "asset": asset,
        "horizon": h,
        "model_depth": model_depth,
        "use_phase5_features": bool(use_phase5_features),
        "signal_mode": mode,
        "threshold_candidates": thresholds,
        "cooldown_candidates": cooldowns,
        "validation_fraction": float(validation_fraction),
        "transaction_cost": float(transaction_cost),
        "risk_variant_names": list(risk_variant_names) if risk_variant_names is not None else "all",
        "cost_values": [float(c) for c in cost_values],
        "selection_basis": "validation_only_risk_variant_score",
    }
    selected_info = {
        "RiskVariantName": selected_variant.get("RiskVariantName", ""),
        "RiskVariantParams": _variant_param_text(selected_variant.get("RiskVariantParams", {}), selected_state),
        "BaseThreshold": selected_long,
        "BaseCooldown": selected_cooldown,
        "RiskValidationScore": _safe_float(selected_payload["row"].get("RiskValidationScore"), default=np.nan),
    }
    return RiskControlUpgradeReport(
        baseline_vs_best=summary["baseline_vs_best"],
        full_variant_table=full_table,
        cost_stress_table=cost_stress,
        warnings=warnings,
        selected_variant=selected_info,
        baseline_result=baseline_result,
        selected_result=selected_locked_result,
        settings=settings,
    )


def build_walk_forward_windows(
    *,
    total_rows: int,
    validation_window: int = 180,
    test_window: int = 90,
    step_size: int = 60,
    mode: str = "rolling",
) -> pd.DataFrame:
    """Build chronological validation -> locked-test walk-forward windows."""
    n = int(max(0, total_rows))
    val_n = int(max(1, validation_window))
    test_n = int(max(1, test_window))
    step_n = int(max(1, step_size))
    window_mode = str(mode).lower().strip()
    rows: List[Dict[str, Any]] = []

    if n < val_n + test_n:
        return pd.DataFrame(columns=["WindowId", "ValidationStartRow", "ValidationEndRow", "LockedTestStartRow", "LockedTestEndRow", "WindowMode"])

    start = 0
    window_id = 1
    while True:
        if window_mode == "expanding":
            validation_start = 0
            validation_end = val_n + start
        else:
            validation_start = start
            validation_end = start + val_n
        locked_start = validation_end
        locked_end = locked_start + test_n
        if locked_end > n:
            break
        rows.append(
            {
                "WindowId": int(window_id),
                "ValidationStartRow": int(validation_start),
                "ValidationEndRow": int(validation_end),
                "LockedTestStartRow": int(locked_start),
                "LockedTestEndRow": int(locked_end),
                "WindowMode": "expanding" if window_mode == "expanding" else "rolling",
            }
        )
        start += step_n
        window_id += 1
    return pd.DataFrame(rows)


def _slice_signal_segment(aligned: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    return {
        "probabilities_up": aligned["probabilities_up"][start:end],
        "future_returns": aligned["future_returns"][start:end],
        "actual_direction": aligned["actual_direction"][start:end],
        "index": aligned["index"][start:end],
    }


def _index_label(index: Iterable[Any], fallback: str = "") -> str:
    idx = pd.Index(index)
    if len(idx) == 0:
        return fallback
    return str(idx[0]) if fallback == "start" else str(idx[-1])


def evaluate_walk_forward_window(
    *,
    signal_output: Any,
    window: Dict[str, Any],
    asset_name: str,
    horizon: int,
    model_depth: str = "core",
    use_phase5_features: bool = True,
    signal_mode: str = "long_only",
    threshold_candidates: Iterable[float] = (0.50, 0.55, 0.60, 0.65, 0.70),
    cooldown_candidates: Iterable[int] = (0, 2, 5),
    transaction_cost: float = 0.001,
    min_trades_per_window: int = 3,
) -> Dict[str, Any]:
    """
    Evaluate one walk-forward window.

    Threshold and cooldown are selected only on the validation slice. The
    locked-test slice is evaluated once after settings are fixed.
    """
    p_up, log_returns, y_dir, index = _aligned_signal_arrays(
        probabilities_up=signal_output.probabilities_up_test,
        future_returns=signal_output.actual_return_test,
        actual_direction=signal_output.actual_direction_test,
        test_index=signal_output.test_index,
    )
    aligned = {
        "probabilities_up": p_up,
        "future_returns": log_returns,
        "actual_direction": y_dir,
        "index": index,
    }
    validation = _slice_signal_segment(aligned, int(window["ValidationStartRow"]), int(window["ValidationEndRow"]))
    locked = _slice_signal_segment(aligned, int(window["LockedTestStartRow"]), int(window["LockedTestEndRow"]))
    thresholds = [float(t) for t in threshold_candidates]
    cooldowns = [int(max(0, c)) for c in cooldown_candidates]
    mode = _normalize_mode(signal_mode)
    short_thresholds = [max(0.0, min(0.49, 1.0 - t)) for t in thresholds] if mode != "long_only" else [min(0.45, t - 0.05) for t in thresholds]

    validation_baseline = _segment_direction_baseline_accuracy(validation["actual_direction"])
    validation_candidates: List[Dict[str, Any]] = []
    for cooldown in cooldowns:
        sweep = run_threshold_sweep(
            probabilities_up=validation["probabilities_up"],
            future_returns=validation["future_returns"],
            actual_direction=validation["actual_direction"],
            test_index=validation["index"],
            baseline_direction_accuracy=validation_baseline,
            mode=mode,
            long_thresholds=thresholds,
            short_thresholds=short_thresholds,
            transaction_cost=transaction_cost,
            horizon=int(horizon),
            backtest_style="non_overlapping_realistic",
            cooldown=int(cooldown),
            asset=asset_name,
        )
        scored = _score_validation_sweep(sweep)
        if scored.empty:
            continue
        selected_rows = scored[scored["SelectedLockedThreshold"].eq(True)]
        selected = selected_rows.iloc[0] if not selected_rows.empty else scored.iloc[0]
        candidate = dict(selected)
        candidate["SelectedCooldown"] = int(cooldown)
        validation_candidates.append(candidate)

    if not validation_candidates:
        raise ValueError("No validation candidates were produced for walk-forward window")

    selected_candidate = sorted(
        validation_candidates,
        key=lambda r: (
            -_safe_float(r.get("ValidationSelectionScore"), default=-np.inf),
            -_metric_value(r, ["NumberOfTrades", "SignalCount"], default=0.0),
            _safe_float(r.get("SelectedCooldown"), default=np.inf),
        ),
    )[0]
    selected_threshold = _safe_float(selected_candidate.get("LongThreshold"), default=np.nan)
    selected_short = _safe_float(selected_candidate.get("ShortThreshold"), default=min(0.45, selected_threshold - 0.05))
    selected_cooldown = int(selected_candidate["SelectedCooldown"])

    locked_baseline = _segment_direction_baseline_accuracy(locked["actual_direction"])
    locked_result = run_realistic_trade_backtest(
        probabilities_up=locked["probabilities_up"],
        future_returns=locked["future_returns"],
        actual_direction=locked["actual_direction"],
        test_index=locked["index"],
        asset=asset_name,
        baseline_direction_accuracy=locked_baseline,
        long_threshold=selected_threshold,
        short_threshold=selected_short,
        mode=mode,
        transaction_cost=transaction_cost,
        horizon=int(horizon),
        cooldown=selected_cooldown,
    )

    locked_metrics = locked_result.metrics
    validation_trades = int(_metric_value(selected_candidate, ["NumberOfTrades", "SignalCount"], default=0.0))
    locked_trades = int(locked_metrics.get("NumberOfTrades", locked_metrics.get("SignalCount", 0)))
    locked_vs_bh = _safe_float(locked_metrics.get("StrategyMinusBuyHold_%"), default=np.nan)
    locked_return = _safe_float(locked_metrics.get("TotalCompoundedReturn_%", locked_metrics.get("StrategyTotalReturn_%")), default=np.nan)
    locked_dd = _safe_float(locked_metrics.get("MaxDrawdown_%"), default=np.nan)
    beat_bh = bool(np.isfinite(locked_vs_bh) and locked_vs_bh > 0.0)
    positive_return = bool(np.isfinite(locked_return) and locked_return > 0.0)

    failures: List[str] = []
    if locked_trades < int(min_trades_per_window):
        failures.append(f"low locked-test trades ({locked_trades} < {int(min_trades_per_window)})")
    if not beat_bh:
        failures.append("locked test fails buy-and-hold")
    if not positive_return:
        failures.append("locked strategy return is not positive")
    if np.isfinite(locked_dd) and locked_dd <= -25.0:
        failures.append("locked drawdown is severe")

    if failures:
        window_verdict = "Do not trust"
    elif beat_bh and positive_return and locked_trades >= max(int(min_trades_per_window), 5):
        window_verdict = "Window passed"
    else:
        window_verdict = "Weak / unstable research only"

    return {
        "WindowId": int(window["WindowId"]),
        "Asset": asset_name,
        "Horizon": int(horizon),
        "ModelDepth": model_depth,
        "Phase5Enabled": bool(use_phase5_features),
        "SignalMode": mode,
        "ValidationStart": _index_label(validation["index"], "start"),
        "ValidationEnd": _index_label(validation["index"]),
        "LockedTestStart": _index_label(locked["index"], "start"),
        "LockedTestEnd": _index_label(locked["index"]),
        "SelectedThreshold": round(float(selected_threshold), 4),
        "SelectedCooldown": selected_cooldown,
        "ValidationTrades": validation_trades,
        "ValidationStrategyReturn_%": _metric_value(selected_candidate, ["TotalCompoundedReturn_%", "StrategyTotalReturn_%"], default=np.nan),
        "ValidationBuyHoldReturn_%": _metric_value(selected_candidate, ["BuyHoldReturn_%"], default=np.nan),
        "ValidationVsBuyHold_%": _metric_value(selected_candidate, ["StrategyMinusBuyHold_%"], default=np.nan),
        "ValidationMaxDrawdown_%": _metric_value(selected_candidate, ["MaxDrawdown_%"], default=np.nan),
        "LockedTrades": locked_trades,
        "LockedWinRate_%": locked_metrics.get("WinRate_%", locked_metrics.get("WinRateActive_%", np.nan)),
        "LockedStrategyReturn_%": locked_metrics.get("TotalCompoundedReturn_%", locked_metrics.get("StrategyTotalReturn_%", np.nan)),
        "LockedBuyHoldReturn_%": locked_metrics.get("BuyHoldReturn_%", np.nan),
        "LockedVsBuyHold_%": locked_metrics.get("StrategyMinusBuyHold_%", np.nan),
        "LockedMaxDrawdown_%": locked_metrics.get("MaxDrawdown_%", np.nan),
        "LockedSharpe": locked_metrics.get("Sharpe", np.nan),
        "LockedExposure_%": locked_metrics.get("Exposure_%", np.nan),
        "BeatBuyHold": beat_bh,
        "PositiveStrategyReturn": positive_return,
        "WindowVerdict": window_verdict,
        "FailureReason": "; ".join(failures),
        "ValidationSelectionScore": _safe_float(selected_candidate.get("ValidationSelectionScore"), default=np.nan),
    }


def threshold_cooldown_stability(window_results: pd.DataFrame) -> Dict[str, Any]:
    if window_results is None or window_results.empty:
        return {"ThresholdStability": "No windows", "CooldownStability": "No windows", "SettingsStabilityScore": 0.0}
    thresholds = window_results["SelectedThreshold"].dropna().astype(float)
    cooldowns = window_results["SelectedCooldown"].dropna().astype(float)
    threshold_range = float(thresholds.max() - thresholds.min()) if len(thresholds) else np.nan
    cooldown_unique = int(cooldowns.nunique()) if len(cooldowns) else 0

    if len(thresholds) == 0:
        threshold_stability = "No threshold"
        threshold_score = 0.0
    elif threshold_range <= 0.05:
        threshold_stability = "Stable"
        threshold_score = 100.0
    elif threshold_range <= 0.10:
        threshold_stability = "Moderate"
        threshold_score = 65.0
    else:
        threshold_stability = "Unstable"
        threshold_score = 25.0

    if cooldown_unique <= 1 and len(cooldowns):
        cooldown_stability = "Stable"
        cooldown_score = 100.0
    elif cooldown_unique <= 2:
        cooldown_stability = "Moderate"
        cooldown_score = 65.0
    else:
        cooldown_stability = "Unstable"
        cooldown_score = 25.0

    return {
        "ThresholdStability": threshold_stability,
        "CooldownStability": cooldown_stability,
        "SettingsStabilityScore": round(float((threshold_score + cooldown_score) / 2.0), 2),
        "ThresholdRange": round(float(threshold_range), 4) if np.isfinite(threshold_range) else np.nan,
        "CooldownUniqueCount": cooldown_unique,
    }


def walk_forward_verdict(summary_row: Dict[str, Any]) -> Dict[str, str]:
    windows = int(_metric_value(summary_row, ["NumberOfWindows"], default=0.0))
    beat_rate = _metric_value(summary_row, ["BeatBuyHoldRate_%"], default=0.0)
    positive_rate = _metric_value(summary_row, ["PositiveReturnRate_%"], default=0.0)
    avg_vs = _metric_value(summary_row, ["AvgLockedVsBuyHold_%"], default=np.nan)
    median_vs = _metric_value(summary_row, ["MedianLockedVsBuyHold_%"], default=np.nan)
    worst_dd = _metric_value(summary_row, ["WorstLockedMaxDrawdown_%"], default=0.0)
    avg_trades = _metric_value(summary_row, ["AvgTradesPerWindow"], default=0.0)
    low_trade_windows = int(_metric_value(summary_row, ["LowTradeWindowCount"], default=0.0))
    stability_score = _metric_value(summary_row, ["WalkForwardStabilityScore"], default=0.0)
    threshold_stability = str(summary_row.get("ThresholdStability", ""))
    cooldown_stability = str(summary_row.get("CooldownStability", ""))

    failures: List[str] = []
    if windows < 2:
        failures.append("too few walk-forward windows")
    if beat_rate < 50.0:
        failures.append("fails buy-and-hold in most windows")
    if np.isfinite(median_vs) and median_vs <= 0.0:
        failures.append("negative median edge")
    if low_trade_windows > 0:
        failures.append(f"{low_trade_windows} low-trade windows")
    if np.isfinite(worst_dd) and worst_dd <= -25.0:
        failures.append("severe worst drawdown")
    if "Unstable" in threshold_stability or "Unstable" in cooldown_stability:
        failures.append("threshold/cooldown unstable")

    if windows == 0 or beat_rate < 40.0 or (np.isfinite(median_vs) and median_vs <= 0.0) or avg_trades < 1.0:
        verdict = "Do not trust"
    elif (
        beat_rate >= 70.0
        and positive_rate >= 70.0
        and np.isfinite(avg_vs)
        and avg_vs > 0.0
        and np.isfinite(median_vs)
        and median_vs > 0.0
        and worst_dd > -20.0
        and low_trade_windows == 0
        and stability_score >= 65.0
    ):
        verdict = "Strong walk-forward research candidate"
    elif beat_rate >= 50.0 and np.isfinite(median_vs) and median_vs > 0.0 and avg_trades >= 2.0:
        verdict = "Research candidate"
    else:
        verdict = "Weak / unstable research only"

    return {"WalkForwardVerdict": verdict, "FailureReason": "; ".join(dict.fromkeys(failures))}


def summarize_walk_forward_results(window_results: pd.DataFrame, *, min_trades_per_window: int = 3) -> pd.DataFrame:
    if window_results is None or window_results.empty:
        return pd.DataFrame(columns=WALK_FORWARD_AGG_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for (asset, horizon), group in window_results.groupby(["Asset", "Horizon"], dropna=False):
        g = group.copy()
        n = int(len(g))
        beat_count = int(g["BeatBuyHold"].fillna(False).astype(bool).sum())
        positive_count = int(g["PositiveStrategyReturn"].fillna(False).astype(bool).sum())
        low_trade_count = int((g["LockedTrades"].fillna(0).astype(float) < int(min_trades_per_window)).sum())
        stability = threshold_cooldown_stability(g)
        beat_rate = beat_count / max(n, 1) * 100.0
        positive_rate = positive_count / max(n, 1) * 100.0
        avg_vs = float(g["LockedVsBuyHold_%"].astype(float).mean()) if n else np.nan
        median_vs = float(g["LockedVsBuyHold_%"].astype(float).median()) if n else np.nan
        worst_dd = float(g["LockedMaxDrawdown_%"].astype(float).min()) if n else np.nan
        avg_sharpe = float(g["LockedSharpe"].astype(float).mean()) if n else np.nan
        stability_score = float(
            np.clip(
                beat_rate * 0.35
                + positive_rate * 0.20
                + max(min(avg_vs, 20.0), -20.0) * 1.0
                + stability["SettingsStabilityScore"] * 0.25
                - max(0.0, abs(min(worst_dd, 0.0)) - 20.0),
                0.0,
                100.0,
            )
        )
        row = {
            "Asset": asset,
            "Horizon": int(horizon),
            "NumberOfWindows": n,
            "WindowsBeatingBuyHold": beat_count,
            "BeatBuyHoldRate_%": round(float(beat_rate), 2),
            "PositiveReturnWindows": positive_count,
            "PositiveReturnRate_%": round(float(positive_rate), 2),
            "AvgLockedStrategyReturn_%": round(float(g["LockedStrategyReturn_%"].astype(float).mean()), 4),
            "MedianLockedStrategyReturn_%": round(float(g["LockedStrategyReturn_%"].astype(float).median()), 4),
            "AvgLockedVsBuyHold_%": round(float(avg_vs), 4),
            "MedianLockedVsBuyHold_%": round(float(median_vs), 4),
            "WorstLockedVsBuyHold_%": round(float(g["LockedVsBuyHold_%"].astype(float).min()), 4),
            "BestLockedVsBuyHold_%": round(float(g["LockedVsBuyHold_%"].astype(float).max()), 4),
            "AvgLockedMaxDrawdown_%": round(float(g["LockedMaxDrawdown_%"].astype(float).mean()), 4),
            "WorstLockedMaxDrawdown_%": round(float(worst_dd), 4),
            "AvgLockedSharpe": round(float(avg_sharpe), 4),
            "MedianLockedSharpe": round(float(g["LockedSharpe"].astype(float).median()), 4),
            "AvgTradesPerWindow": round(float(g["LockedTrades"].astype(float).mean()), 2),
            "LowTradeWindowCount": low_trade_count,
            "ThresholdStability": stability["ThresholdStability"],
            "CooldownStability": stability["CooldownStability"],
            "WalkForwardStabilityScore": round(stability_score, 2),
        }
        row.update(walk_forward_verdict(row))
        rows.append(row)

    out = pd.DataFrame(rows)
    for col in WALK_FORWARD_AGG_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[WALK_FORWARD_AGG_COLUMNS]


def run_walk_forward_risk_validation(
    *,
    raw_df: Optional[pd.DataFrame] = None,
    asset_names: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    model_depth: str = "core",
    use_phase5_features: bool = True,
    signal_mode: str = "long_only",
    threshold_candidates: Iterable[float] = (0.50, 0.55, 0.60, 0.65, 0.70),
    cooldown_candidates: Iterable[int] = (0, 2, 5),
    transaction_cost: float = 0.001,
    validation_window: int = 180,
    test_window: int = 90,
    step_size: int = 60,
    min_trades_per_window: int = 3,
    window_mode: str = "rolling",
    signal_outputs: Optional[Dict[Any, Any]] = None,
    signal_output_factory: Optional[Any] = None,
    progress_callback: Optional[Any] = None,
) -> WalkForwardValidationReport:
    """Run walk-forward validation across selected assets and horizons."""
    if asset_names is None:
        from src.asset_config import get_asset_names

        assets = list(get_asset_names())
    else:
        assets = list(asset_names)

    if horizons is None:
        try:
            from src.direct_forecast_models import DIRECT_FORECAST_HORIZONS

            scan_horizons = list(DIRECT_FORECAST_HORIZONS)
        except Exception:
            scan_horizons = [1, 5, 10, 20, 30]
    else:
        scan_horizons = [int(h) for h in horizons]

    if not assets:
        raise ValueError("Select at least one asset")
    if not scan_horizons:
        raise ValueError("Select at least one horizon")

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    warnings: List[str] = []
    total = len(assets) * len(scan_horizons)
    done = 0

    for asset in assets:
        for horizon in scan_horizons:
            try:
                if progress_callback is not None:
                    progress_callback(done, total, f"Preparing {asset} {horizon}D")
                signal_output = _build_signal_output(
                    raw_df=raw_df,
                    asset=asset,
                    horizon=int(horizon),
                    model_depth=model_depth,
                    use_phase5_features=use_phase5_features,
                    signal_outputs=signal_outputs,
                    signal_output_factory=signal_output_factory,
                )
                p_up, log_returns, y_dir, index = _aligned_signal_arrays(
                    probabilities_up=signal_output.probabilities_up_test,
                    future_returns=signal_output.actual_return_test,
                    actual_direction=signal_output.actual_direction_test,
                    test_index=signal_output.test_index,
                )
                windows = build_walk_forward_windows(
                    total_rows=len(p_up),
                    validation_window=validation_window,
                    test_window=test_window,
                    step_size=step_size,
                    mode=window_mode,
                )
                if windows.empty:
                    msg = f"not enough rows for walk-forward windows ({len(p_up)} rows)"
                    errors.append({"Asset": asset, "Horizon": int(horizon), "Error": msg})
                    warnings.append(f"{asset} {horizon}D: {msg}")
                    continue
                for _, window in windows.iterrows():
                    try:
                        row = evaluate_walk_forward_window(
                            signal_output=signal_output,
                            window=window.to_dict(),
                            asset_name=asset,
                            horizon=int(horizon),
                            model_depth=model_depth,
                            use_phase5_features=use_phase5_features,
                            signal_mode=signal_mode,
                            threshold_candidates=threshold_candidates,
                            cooldown_candidates=cooldown_candidates,
                            transaction_cost=transaction_cost,
                            min_trades_per_window=min_trades_per_window,
                        )
                    except Exception as exc:
                        row = {
                            "WindowId": int(window["WindowId"]),
                            "Asset": asset,
                            "Horizon": int(horizon),
                            "ValidationStart": "",
                            "ValidationEnd": "",
                            "LockedTestStart": "",
                            "LockedTestEnd": "",
                            "SelectedThreshold": np.nan,
                            "SelectedCooldown": np.nan,
                            "ValidationTrades": 0,
                            "LockedTrades": 0,
                            "BeatBuyHold": False,
                            "PositiveStrategyReturn": False,
                            "WindowVerdict": "Do not trust",
                            "FailureReason": str(exc),
                        }
                        errors.append({"Asset": asset, "Horizon": int(horizon), "WindowId": int(window["WindowId"]), "Error": str(exc)})
                    rows.append(row)
            finally:
                done += 1
                if progress_callback is not None:
                    progress_callback(done, total, f"Completed {asset} {horizon}D")

    window_results = pd.DataFrame(rows)
    if not window_results.empty:
        for col in WALK_FORWARD_WINDOW_COLUMNS:
            if col not in window_results.columns:
                window_results[col] = np.nan
        leading_cols = WALK_FORWARD_WINDOW_COLUMNS
        extra_cols = [c for c in window_results.columns if c not in leading_cols]
        window_results = window_results[leading_cols + extra_cols]
    else:
        window_results = pd.DataFrame(columns=WALK_FORWARD_WINDOW_COLUMNS)

    aggregate = summarize_walk_forward_results(window_results, min_trades_per_window=min_trades_per_window)
    verdict_counts = aggregate["WalkForwardVerdict"].fillna("Unknown").value_counts().to_dict() if not aggregate.empty else {}
    if not aggregate.empty:
        weak = aggregate[aggregate["WalkForwardVerdict"].astype(str).ne("Strong walk-forward research candidate")]
        for _, row in weak.iterrows():
            warnings.append(f"{row['Asset']} {int(row['Horizon'])}D: {row['WalkForwardVerdict']} ({row.get('FailureReason', '')})")
    settings = {
        "assets": assets,
        "horizons": scan_horizons,
        "model_depth": model_depth,
        "use_phase5_features": bool(use_phase5_features),
        "signal_mode": _normalize_mode(signal_mode),
        "threshold_candidates": [float(t) for t in threshold_candidates],
        "cooldown_candidates": [int(c) for c in cooldown_candidates],
        "transaction_cost": float(transaction_cost),
        "validation_window": int(validation_window),
        "test_window": int(test_window),
        "step_size": int(step_size),
        "min_trades_per_window": int(min_trades_per_window),
        "window_mode": str(window_mode),
        "selection_basis": "validation_only_per_walk_forward_window",
    }
    return WalkForwardValidationReport(
        aggregate_summary=aggregate,
        window_results=window_results,
        verdict_counts=verdict_counts,
        warnings=list(dict.fromkeys([w for w in warnings if w])),
        errors=pd.DataFrame(errors),
        settings=settings,
    )
