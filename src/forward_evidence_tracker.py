"""Phase 9 forward paper-trading evidence tracker.

This module records research-only paper signals, keeps immature outcomes
pending, and evaluates matured rows only from price history available at the
requested as-of date. It does not trade, promote candidates, or label anything
production-ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.asset_config import get_asset_names, get_target_column


DEFAULT_FORWARD_HORIZONS: Tuple[int, ...] = (1, 5, 10, 20, 30)

FORWARD_SIGNAL_LOG_COLUMNS: Tuple[str, ...] = (
    "SignalId",
    "CreatedAt",
    "Asset",
    "Horizon",
    "SignalDate",
    "TargetOutcomeDate",
    "ModelName",
    "ProbabilityUp",
    "PredictedDirection",
    "SignalStrength",
    "EntryPrice",
    "Status",
    "ActualOutcomeDate",
    "ExitPrice",
    "ActualDirection",
    "RealizedReturn",
    "BenchmarkReturn",
    "VsBuyHold",
    "WinLoss",
    "BeatBenchmark",
    "EvidenceMode",
    "Warnings",
)

FORWARD_ACCURACY_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "TotalSignals",
    "PendingSignals",
    "MaturedSignals",
    "InvalidSignals",
    "DirectionalAccuracy_%",
    "WinRate_%",
    "BeatBenchmarkRate_%",
    "AvgRealizedReturn_%",
    "MedianRealizedReturn_%",
    "AvgVsBuyHold_%",
    "MedianVsBuyHold_%",
    "WorstRealizedReturn_%",
    "EvidenceVerdict",
    "Warnings",
)

FORWARD_PROBABILITY_CALIBRATION_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "Rows",
    "BrierScore",
    "ECE",
    "MeanProbabilityUp",
    "ActualUpRate_%",
    "HighConfidenceRows",
    "HighConfidenceWinRate_%",
    "CalibrationVerdict",
    "Warnings",
)

FORWARD_COVERAGE_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "TotalSignals",
    "PendingSignals",
    "MaturedSignals",
    "InvalidSignals",
    "CoverageStatus",
    "CoverageScore",
    "Warnings",
)

FORWARD_WARNING_COLUMNS: Tuple[str, ...] = (
    "SignalId",
    "Asset",
    "Horizon",
    "WarningType",
    "Severity",
    "Message",
)

FORWARD_NEXT_ACTION_COLUMNS: Tuple[str, ...] = (
    "Asset",
    "Horizon",
    "NextResearchAction",
    "ActionPriority",
)


@dataclass
class ForwardPaperEvidenceReport:
    forward_signal_log: pd.DataFrame
    pending_outcome_table: pd.DataFrame
    matured_outcome_table: pd.DataFrame
    forward_accuracy_summary: pd.DataFrame
    forward_probability_calibration_summary: pd.DataFrame
    asset_horizon_forward_coverage: pd.DataFrame
    warning_table: pd.DataFrame
    next_research_action_table: pd.DataFrame
    settings: Dict[str, Any] = field(default_factory=dict)


def _empty_signal_log() -> pd.DataFrame:
    return pd.DataFrame(columns=list(FORWARD_SIGNAL_LOG_COLUMNS))


def _empty_report(settings: Optional[Dict[str, Any]] = None) -> ForwardPaperEvidenceReport:
    return ForwardPaperEvidenceReport(
        forward_signal_log=_empty_signal_log(),
        pending_outcome_table=_empty_signal_log(),
        matured_outcome_table=_empty_signal_log(),
        forward_accuracy_summary=pd.DataFrame(columns=list(FORWARD_ACCURACY_SUMMARY_COLUMNS)),
        forward_probability_calibration_summary=pd.DataFrame(columns=list(FORWARD_PROBABILITY_CALIBRATION_COLUMNS)),
        asset_horizon_forward_coverage=pd.DataFrame(columns=list(FORWARD_COVERAGE_COLUMNS)),
        warning_table=pd.DataFrame(columns=list(FORWARD_WARNING_COLUMNS)),
        next_research_action_table=pd.DataFrame(columns=list(FORWARD_NEXT_ACTION_COLUMNS)),
        settings=settings or {},
    )


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _to_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    return pd.Timestamp(ts) if pd.notna(ts) else pd.NaT


def _as_of_timestamp(raw_df: Optional[pd.DataFrame], as_of_date: Optional[Any]) -> pd.Timestamp:
    if as_of_date is not None:
        return _to_timestamp(as_of_date).normalize()
    if raw_df is not None and not raw_df.empty:
        idx = pd.DatetimeIndex(pd.to_datetime(raw_df.index, errors="coerce")).dropna()
        if len(idx):
            return pd.Timestamp(idx.max()).normalize()
    return pd.Timestamp.utcnow().tz_localize(None).normalize()


def _normalise_horizon(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Horizon" not in out.columns:
        return out
    out["Horizon"] = out["Horizon"].astype(str).str.replace("D", "", regex=False)
    out["Horizon"] = pd.to_numeric(out["Horizon"], errors="coerce").astype("Int64")
    return out


def _join_warnings(warnings: Iterable[Any]) -> str:
    clean: List[str] = []
    for warning in warnings:
        text = str(warning or "").strip()
        if not text or text.lower() == "nan":
            continue
        for part in text.split(";"):
            item = part.strip()
            if item and item not in clean:
                clean.append(item)
    return "; ".join(clean)


def _normalise_direction(value: Any, probability: Any = np.nan) -> str:
    text = str(value or "").strip().lower()
    if text in {"up", "long", "buy", "1", "true", "yes"}:
        return "Up"
    if text in {"down", "short", "sell", "0", "false", "no"}:
        return "Down"
    probability_value = _safe_float(probability, default=np.nan)
    if np.isfinite(probability_value):
        return "Up" if probability_value >= 0.5 else "Down"
    return ""


def _signal_strength(probability: Any) -> str:
    p = _safe_float(probability, default=np.nan)
    if not np.isfinite(p):
        return "Unknown"
    distance = abs(p - 0.5)
    if distance >= 0.20:
        return "High"
    if distance >= 0.10:
        return "Medium"
    return "Low"


def _signal_id(asset: Any, horizon: Any, signal_date: Any, model_name: Any) -> str:
    raw = f"{asset}|{_safe_int(horizon)}|{pd.Timestamp(signal_date).date() if pd.notna(signal_date) else ''}|{model_name}"
    return sha1(raw.encode("utf-8")).hexdigest()[:16]


def _price_frame(raw_df: Optional[pd.DataFrame], as_of: pd.Timestamp) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    out = raw_df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    return out[out.index <= as_of].copy()


def _entry_price(price_df: pd.DataFrame, asset: str, signal_date: pd.Timestamp) -> Tuple[float, Optional[int], Optional[pd.Timestamp]]:
    if price_df.empty or not pd.notna(signal_date):
        return np.nan, None, None
    target_col = get_target_column(asset)
    if target_col not in price_df.columns:
        return np.nan, None, None
    idx = pd.DatetimeIndex(price_df.index)
    candidates = np.where(idx >= signal_date)[0]
    if len(candidates) == 0:
        return np.nan, None, None
    pos = int(candidates[0])
    price = _safe_float(price_df.iloc[pos][target_col], default=np.nan)
    return price, pos, pd.Timestamp(idx[pos])


def _target_date_from_calendar(signal_date: pd.Timestamp, horizon: int, calendar: Optional[pd.DatetimeIndex]) -> pd.Timestamp:
    if calendar is not None and len(calendar) and pd.notna(signal_date):
        candidates = np.where(calendar >= signal_date)[0]
        if len(candidates):
            pos = int(candidates[0]) + int(horizon)
            if pos < len(calendar):
                return pd.Timestamp(calendar[pos]).normalize()
    return (signal_date + pd.offsets.BDay(int(horizon))).normalize()


def _calibration_warning_lookup(
    probability_calibration_summary: Optional[pd.DataFrame],
    probability_calibration_warnings: Optional[pd.DataFrame],
) -> Dict[Tuple[str, int], List[str]]:
    lookup: Dict[Tuple[str, int], List[str]] = {}
    for table in [probability_calibration_summary, probability_calibration_warnings]:
        if table is None or table.empty or not {"Asset", "Horizon"}.issubset(table.columns):
            continue
        table = _normalise_horizon(table)
        for _, row in table.iterrows():
            key = (str(row.get("Asset", "")), _safe_int(row.get("Horizon")))
            warnings: List[str] = []
            grade = str(row.get("CalibrationGrade", "") or "")
            warning_type = str(row.get("WarningType", "") or "")
            raw_available = row.get("RawProbabilityOutcomesAvailable", True)
            if warning_type:
                warnings.append(warning_type)
            if grade and grade not in {"WellCalibrated", "UsefulButNoisy"}:
                warnings.append("ProbabilityStillUnreliable")
            if raw_available is False:
                warnings.append("ProbabilityStillUnreliable")
            lookup.setdefault(key, [])
            lookup[key] = _join_warnings(lookup[key] + warnings).split("; ") if warnings or lookup[key] else lookup[key]
    return lookup


def _normalise_existing_log(existing_forward_signal_log: Optional[pd.DataFrame]) -> pd.DataFrame:
    if existing_forward_signal_log is None or existing_forward_signal_log.empty:
        return _empty_signal_log()
    out = existing_forward_signal_log.copy()
    for col in FORWARD_SIGNAL_LOG_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[list(FORWARD_SIGNAL_LOG_COLUMNS)]
    out = _normalise_horizon(out)
    for col in ["SignalDate", "TargetOutcomeDate", "ActualOutcomeDate", "CreatedAt"]:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    out["Asset"] = out["Asset"].astype(str)
    out["Horizon"] = pd.to_numeric(out["Horizon"], errors="coerce").fillna(0).astype(int)
    out["EvidenceMode"] = out["EvidenceMode"].fillna("ForwardPaperSignal").replace("", "ForwardPaperSignal")
    missing_id = out["SignalId"].isna() | out["SignalId"].astype(str).eq("")
    for idx in out[missing_id].index:
        out.at[idx, "SignalId"] = _signal_id(out.at[idx, "Asset"], out.at[idx, "Horizon"], out.at[idx, "SignalDate"], out.at[idx, "ModelName"])
    return out.reset_index(drop=True)


def _prediction_rows_to_signal_log(
    predictions_table: Optional[pd.DataFrame],
    raw_df: Optional[pd.DataFrame],
    assets: Iterable[str],
    horizons: Iterable[int],
    as_of: pd.Timestamp,
    calibration_warnings: Dict[Tuple[str, int], List[str]],
) -> pd.DataFrame:
    if predictions_table is None or predictions_table.empty:
        return _empty_signal_log()
    df = _normalise_horizon(predictions_table)
    if "Asset" not in df.columns or "Horizon" not in df.columns:
        return _empty_signal_log()
    df["Asset"] = df["Asset"].astype(str)
    df["Horizon"] = pd.to_numeric(df["Horizon"], errors="coerce").astype("Int64")
    asset_set = set(str(asset) for asset in assets)
    horizon_set = set(int(h) for h in horizons)
    df = df[df["Asset"].isin(asset_set) & df["Horizon"].isin(horizon_set)].copy()
    if df.empty:
        return _empty_signal_log()

    price_df = _price_frame(raw_df, as_of)
    calendar = pd.DatetimeIndex(price_df.index) if not price_df.empty else None
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        asset = str(row.get("Asset", ""))
        horizon = _safe_int(row.get("Horizon"))
        probability = _safe_float(row.get("ProbabilityUp", row.get("PredictedProbabilityUp", np.nan)), default=np.nan)
        signal_date = _to_timestamp(row.get("SignalDate", row.get("Date", as_of))).normalize()
        target_date = _to_timestamp(row.get("TargetOutcomeDate", pd.NaT))
        if not pd.notna(target_date):
            target_date = _target_date_from_calendar(signal_date, horizon, calendar)
        model_name = str(row.get("ModelName", row.get("Model", "UploadedPrediction")) or "UploadedPrediction")
        entry_price = _safe_float(row.get("EntryPrice"), default=np.nan)
        if not np.isfinite(entry_price):
            entry_price, _, _ = _entry_price(price_df, asset, signal_date)
        warnings = list(calibration_warnings.get((asset, horizon), []))
        warnings.extend([w.strip() for w in str(row.get("Warnings", row.get("Warning", ""))).split(";") if w.strip()])
        if not np.isfinite(probability):
            warnings.append("MissingProbability")
        if not np.isfinite(entry_price):
            warnings.append("MissingEntryPrice")
        warnings.append("PendingOutcome")
        status = "Invalid" if "MissingProbability" in warnings or "MissingEntryPrice" in warnings else "Pending"
        rows.append(
            {
                "SignalId": _signal_id(asset, horizon, signal_date, model_name),
                "CreatedAt": pd.Timestamp.utcnow().tz_localize(None),
                "Asset": asset,
                "Horizon": int(horizon),
                "SignalDate": signal_date,
                "TargetOutcomeDate": target_date,
                "ModelName": model_name,
                "ProbabilityUp": probability,
                "PredictedDirection": _normalise_direction(row.get("PredictedDirection", ""), probability),
                "SignalStrength": str(row.get("SignalStrength", _signal_strength(probability)) or _signal_strength(probability)),
                "EntryPrice": entry_price,
                "Status": status,
                "ActualOutcomeDate": pd.NaT,
                "ExitPrice": np.nan,
                "ActualDirection": np.nan,
                "RealizedReturn": np.nan,
                "BenchmarkReturn": np.nan,
                "VsBuyHold": np.nan,
                "WinLoss": "",
                "BeatBenchmark": np.nan,
                "EvidenceMode": "ForwardPaperSignal",
                "Warnings": _join_warnings(warnings),
            }
        )
    return pd.DataFrame(rows, columns=list(FORWARD_SIGNAL_LOG_COLUMNS))


def generate_forward_model_prediction_rows(
    *,
    raw_df: pd.DataFrame,
    assets: Iterable[str],
    horizons: Iterable[int],
    model_depth: str = "fast",
    use_phase5_features: bool = True,
    model_name: Optional[str] = None,
    as_of_date: Optional[Any] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate current direct-model probability rows for paper logging.

    The model is fit only on direct-horizon rows whose future labels are known
    as of the supplied data. The scored row is the latest feature row available
    at ``as_of_date`` and remains paper evidence only.
    """
    from src.direct_forecast_models import build_direct_feature_frame, run_direct_forecast_report, _model_specs

    as_of = _as_of_timestamp(raw_df, as_of_date)
    raw_as_of = _price_frame(raw_df, as_of)
    rows: List[Dict[str, Any]] = []
    for asset in assets:
        for horizon in horizons:
            try:
                report = run_direct_forecast_report(
                    raw_df=raw_as_of,
                    asset_name=str(asset),
                    horizon=int(horizon),
                    model_depth=model_depth,
                    use_phase5_features=use_phase5_features,
                    random_state=random_state,
                )
                if report.dataset is None or report.leaderboard is None or report.leaderboard.empty:
                    raise ValueError("No direct forecast dataset or leaderboard")
                dataset = report.dataset
                selected_model = str(model_name or report.leaderboard.iloc[0]["Model"])
                direction_models = {name: direction_model for name, _, direction_model in _model_specs(model_depth, random_state=random_state)}
                if selected_model not in direction_models:
                    raise ValueError(f"Unknown direct direction model: {selected_model}")
                feature_df = build_direct_feature_frame(
                    raw_as_of,
                    target_col=get_target_column(str(asset)),
                    use_phase5_features=use_phase5_features,
                ).replace([np.inf, -np.inf], np.nan)
                feature_df = feature_df[feature_df.index <= as_of].dropna(subset=dataset.feature_cols)
                if feature_df.empty:
                    raise ValueError("No current feature row available for paper signal")
                latest = feature_df.iloc[[-1]]
                signal_date = pd.Timestamp(latest.index[-1]).normalize()
                X_current = dataset.feature_scaler.transform(latest[dataset.feature_cols].astype(float).to_numpy())
                X_known_raw = dataset.df_model[dataset.feature_cols].astype(float).to_numpy()
                X_known = dataset.feature_scaler.transform(X_known_raw)
                y_known = dataset.df_model[dataset.direction_target_col].astype(bool).astype(int).to_numpy()
                unique = np.unique(y_known)
                if len(unique) < 2:
                    probability = float(unique[0])
                else:
                    model = direction_models[selected_model]
                    model.fit(X_known, y_known)
                    if hasattr(model, "predict_proba"):
                        proba_raw = model.predict_proba(X_current)
                        probability = float(proba_raw[:, 1][0] if getattr(proba_raw, "ndim", 1) == 2 and proba_raw.shape[1] >= 2 else np.asarray(proba_raw).flatten()[0])
                    else:
                        probability = float(np.asarray(model.predict(X_current)).flatten()[0])
                probability = float(np.clip(probability, 0.0, 1.0))
                rows.append(
                    {
                        "Asset": str(asset),
                        "Horizon": int(horizon),
                        "SignalDate": signal_date,
                        "ModelName": selected_model,
                        "ProbabilityUp": probability,
                        "PredictedDirection": "Up" if probability >= 0.5 else "Down",
                        "SignalStrength": _signal_strength(probability),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "Asset": str(asset),
                        "Horizon": int(horizon),
                        "SignalDate": as_of,
                        "ModelName": str(model_name or "DirectForecastModel"),
                        "ProbabilityUp": np.nan,
                        "PredictedDirection": "",
                        "SignalStrength": "Unknown",
                        "Warnings": str(exc),
                    }
                )
    return pd.DataFrame(rows)


def _invalid_generation_prediction_rows(
    assets: Iterable[str],
    horizons: Iterable[int],
    as_of: pd.Timestamp,
    reason: str,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for asset in assets:
        for horizon in horizons:
            rows.append(
                {
                    "Asset": str(asset),
                    "Horizon": int(horizon),
                    "SignalDate": as_of,
                    "ModelName": str(model_name or "DirectForecastModel"),
                    "ProbabilityUp": np.nan,
                    "PredictedDirection": "",
                    "SignalStrength": "Unknown",
                    "Warnings": str(reason),
                }
            )
    return pd.DataFrame(rows)


def _append_without_deleting(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        out = new_rows.copy()
    elif new_rows.empty:
        out = existing.copy()
    else:
        out = pd.concat([existing, new_rows], ignore_index=True)
    if out.empty:
        return _empty_signal_log()
    for col in FORWARD_SIGNAL_LOG_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[list(FORWARD_SIGNAL_LOG_COLUMNS)]
    out = out.drop_duplicates(subset=["SignalId"], keep="first")
    return out.reset_index(drop=True)


def _update_one_outcome(row: pd.Series, price_df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    out = row.copy()
    warnings = [w.strip() for w in str(out.get("Warnings", "")).split(";") if w.strip()]
    if "MissingProbability" in warnings:
        out["Status"] = "Invalid"
        out["Warnings"] = _join_warnings(warnings)
        return out
    if str(out.get("Status", "")).lower() == "matured" and pd.notna(out.get("ActualDirection")):
        return out
    asset = str(out.get("Asset", ""))
    horizon = _safe_int(out.get("Horizon"))
    signal_date = _to_timestamp(out.get("SignalDate")).normalize()
    if not pd.notna(signal_date) or signal_date > as_of:
        out["Status"] = "Pending"
        warnings.extend(["PendingOutcome", "OutcomeNotMatured"])
        out["Warnings"] = _join_warnings(warnings)
        return out
    try:
        target_col = get_target_column(asset)
    except Exception:
        out["Status"] = "Invalid"
        warnings.append("MissingEntryPrice")
        out["Warnings"] = _join_warnings(warnings)
        return out
    if price_df.empty or target_col not in price_df.columns:
        out["Status"] = "Invalid"
        warnings.append("MissingEntryPrice")
        out["Warnings"] = _join_warnings(warnings)
        return out
    entry_price, entry_pos, entry_date = _entry_price(price_df, asset, signal_date)
    stored_entry = _safe_float(out.get("EntryPrice"), default=np.nan)
    if not np.isfinite(entry_price) and np.isfinite(stored_entry):
        entry_price = stored_entry
    if entry_pos is None or not np.isfinite(entry_price) or entry_price <= 0:
        out["Status"] = "Invalid"
        warnings.append("MissingEntryPrice")
        out["Warnings"] = _join_warnings(warnings)
        return out
    exit_pos = int(entry_pos) + int(horizon)
    idx = pd.DatetimeIndex(price_df.index)
    if exit_pos >= len(idx):
        out["Status"] = "Pending"
        warnings.extend(["PendingOutcome", "OutcomeNotMatured"])
        out["TargetOutcomeDate"] = _target_date_from_calendar(signal_date, horizon, idx)
        out["Warnings"] = _join_warnings(warnings)
        return out
    exit_date = pd.Timestamp(idx[exit_pos]).normalize()
    if exit_date > as_of:
        out["Status"] = "Pending"
        warnings.extend(["PendingOutcome", "OutcomeNotMatured"])
        out["TargetOutcomeDate"] = exit_date
        out["Warnings"] = _join_warnings(warnings)
        return out
    exit_price = _safe_float(price_df.iloc[exit_pos][target_col], default=np.nan)
    if not np.isfinite(exit_price) or exit_price <= 0:
        out["Status"] = "Invalid"
        warnings.append("MissingExitPrice")
        out["Warnings"] = _join_warnings(warnings)
        return out

    asset_return = float(exit_price / entry_price - 1.0)
    predicted = _normalise_direction(out.get("PredictedDirection", ""), out.get("ProbabilityUp"))
    signal_return = asset_return if predicted != "Down" else -asset_return
    actual_up = bool(asset_return > 0.0)
    out["EntryPrice"] = float(entry_price)
    out["ActualOutcomeDate"] = exit_date
    out["TargetOutcomeDate"] = exit_date
    out["ExitPrice"] = float(exit_price)
    out["ActualDirection"] = int(actual_up)
    out["RealizedReturn"] = float(signal_return)
    out["BenchmarkReturn"] = float(asset_return)
    out["VsBuyHold"] = float(signal_return - asset_return)
    out["WinLoss"] = "Win" if signal_return > 0 else "Loss"
    out["BeatBenchmark"] = bool(signal_return > asset_return)
    out["Status"] = "Matured"
    warnings = [w for w in warnings if w not in {"PendingOutcome", "OutcomeNotMatured"}]
    out["Warnings"] = _join_warnings(warnings)
    return out


def update_matured_forward_outcomes(
    forward_signal_log: pd.DataFrame,
    raw_df: Optional[pd.DataFrame],
    *,
    as_of_date: Optional[Any] = None,
) -> pd.DataFrame:
    log = _normalise_existing_log(forward_signal_log)
    if log.empty:
        return log
    as_of = _as_of_timestamp(raw_df, as_of_date)
    price_df = _price_frame(raw_df, as_of)
    updated_rows = [_update_one_outcome(row, price_df, as_of) for _, row in log.iterrows()]
    out = pd.DataFrame(updated_rows, columns=list(FORWARD_SIGNAL_LOG_COLUMNS))
    return out.reset_index(drop=True)


def _drawdown_from_returns(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    equity = (1.0 + clean).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def _accuracy_summary(log: pd.DataFrame, assets: Iterable[str], horizons: Iterable[int], min_forward_evidence: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for asset in assets:
        for horizon in horizons:
            subset = log[log["Asset"].astype(str).eq(str(asset)) & log["Horizon"].astype(int).eq(int(horizon))].copy() if not log.empty else pd.DataFrame()
            matured = subset[subset["Status"].astype(str).eq("Matured")].copy() if not subset.empty else pd.DataFrame()
            pending = subset[subset["Status"].astype(str).eq("Pending")] if not subset.empty else pd.DataFrame()
            invalid = subset[subset["Status"].astype(str).eq("Invalid")] if not subset.empty else pd.DataFrame()
            warnings: List[str] = []
            if len(matured) < int(min_forward_evidence):
                warnings.extend(["NotEnoughForwardEvidence", "LowTradeCount"])
            realized = pd.to_numeric(matured.get("RealizedReturn", pd.Series(dtype=float)), errors="coerce")
            vs = pd.to_numeric(matured.get("VsBuyHold", pd.Series(dtype=float)), errors="coerce")
            actual = pd.to_numeric(matured.get("ActualDirection", pd.Series(dtype=float)), errors="coerce")
            pred = matured.get("PredictedDirection", pd.Series(dtype=str)).astype(str).str.lower().map({"up": 1, "down": 0})
            direction_acc = float((pred == actual).mean() * 100.0) if len(matured) and actual.notna().any() else np.nan
            win_rate = float(matured["WinLoss"].astype(str).eq("Win").mean() * 100.0) if len(matured) else np.nan
            beat_rate = float(matured["BeatBenchmark"].astype(bool).mean() * 100.0) if len(matured) and "BeatBenchmark" in matured.columns else np.nan
            if len(matured) and pd.to_numeric(vs, errors="coerce").mean() < 0:
                warnings.append("BenchmarkDominated")
            drawdown = _drawdown_from_returns(realized)
            if np.isfinite(drawdown) and drawdown <= -0.15:
                warnings.append("DrawdownRisk")
            verdict = "Not enough forward evidence yet"
            if len(matured) >= int(min_forward_evidence):
                if np.isfinite(direction_acc) and direction_acc >= 55.0 and np.isfinite(vs.mean()) and vs.mean() > 0:
                    verdict = "Forward research evidence improving"
                elif np.isfinite(vs.mean()) and vs.mean() < 0:
                    verdict = "Forward evidence benchmark dominated"
                else:
                    verdict = "Forward research evidence mixed"
            rows.append(
                {
                    "Asset": str(asset),
                    "Horizon": int(horizon),
                    "TotalSignals": int(len(subset)),
                    "PendingSignals": int(len(pending)),
                    "MaturedSignals": int(len(matured)),
                    "InvalidSignals": int(len(invalid)),
                    "DirectionalAccuracy_%": round(direction_acc, 4) if np.isfinite(direction_acc) else np.nan,
                    "WinRate_%": round(win_rate, 4) if np.isfinite(win_rate) else np.nan,
                    "BeatBenchmarkRate_%": round(beat_rate, 4) if np.isfinite(beat_rate) else np.nan,
                    "AvgRealizedReturn_%": round(float(realized.mean() * 100.0), 4) if not realized.dropna().empty else np.nan,
                    "MedianRealizedReturn_%": round(float(realized.median() * 100.0), 4) if not realized.dropna().empty else np.nan,
                    "AvgVsBuyHold_%": round(float(vs.mean() * 100.0), 4) if not vs.dropna().empty else np.nan,
                    "MedianVsBuyHold_%": round(float(vs.median() * 100.0), 4) if not vs.dropna().empty else np.nan,
                    "WorstRealizedReturn_%": round(float(realized.min() * 100.0), 4) if not realized.dropna().empty else np.nan,
                    "EvidenceVerdict": verdict,
                    "Warnings": _join_warnings(warnings),
                }
            )
    return pd.DataFrame(rows, columns=list(FORWARD_ACCURACY_SUMMARY_COLUMNS))


def _ece(probabilities: pd.Series, actuals: pd.Series) -> float:
    frame = pd.DataFrame({"p": pd.to_numeric(probabilities, errors="coerce"), "y": pd.to_numeric(actuals, errors="coerce")}).dropna()
    if frame.empty:
        return np.nan
    bins = pd.cut(frame["p"], bins=[0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.0], include_lowest=True)
    total = len(frame)
    ece = 0.0
    for _, group in frame.groupby(bins, observed=False):
        if group.empty:
            continue
        ece += len(group) / total * abs(float(group["p"].mean()) - float(group["y"].mean()))
    return float(ece)


def _probability_summary(log: pd.DataFrame, assets: Iterable[str], horizons: Iterable[int], min_forward_evidence: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    matured = log[log["Status"].astype(str).eq("Matured")].copy() if not log.empty else pd.DataFrame()
    for asset in assets:
        for horizon in horizons:
            subset = matured[matured["Asset"].astype(str).eq(str(asset)) & matured["Horizon"].astype(int).eq(int(horizon))].copy() if not matured.empty else pd.DataFrame()
            p = pd.to_numeric(subset.get("ProbabilityUp", pd.Series(dtype=float)), errors="coerce")
            y = pd.to_numeric(subset.get("ActualDirection", pd.Series(dtype=float)), errors="coerce")
            frame = pd.DataFrame({"p": p, "y": y}).dropna()
            warnings: List[str] = []
            if len(frame) < int(min_forward_evidence):
                warnings.extend(["NotEnoughForwardEvidence", "LowTradeCount"])
            brier = float(((frame["p"] - frame["y"]) ** 2).mean()) if not frame.empty else np.nan
            ece = _ece(frame["p"], frame["y"]) if not frame.empty else np.nan
            high = frame[frame["p"].ge(0.70)] if not frame.empty else pd.DataFrame()
            high_win = float(high["y"].mean() * 100.0) if not high.empty else np.nan
            verdict = "Not enough forward evidence yet"
            if len(frame) >= int(min_forward_evidence):
                if np.isfinite(brier) and brier <= 0.22 and np.isfinite(ece) and ece <= 0.10:
                    verdict = "Forward probability evidence useful"
                else:
                    verdict = "ProbabilityStillUnreliable"
                    warnings.append("ProbabilityStillUnreliable")
            rows.append(
                {
                    "Asset": str(asset),
                    "Horizon": int(horizon),
                    "Rows": int(len(frame)),
                    "BrierScore": round(brier, 6) if np.isfinite(brier) else np.nan,
                    "ECE": round(ece, 6) if np.isfinite(ece) else np.nan,
                    "MeanProbabilityUp": round(float(frame["p"].mean()), 6) if not frame.empty else np.nan,
                    "ActualUpRate_%": round(float(frame["y"].mean() * 100.0), 4) if not frame.empty else np.nan,
                    "HighConfidenceRows": int(len(high)),
                    "HighConfidenceWinRate_%": round(high_win, 4) if np.isfinite(high_win) else np.nan,
                    "CalibrationVerdict": verdict,
                    "Warnings": _join_warnings(warnings),
                }
            )
    return pd.DataFrame(rows, columns=list(FORWARD_PROBABILITY_CALIBRATION_COLUMNS))


def _coverage_table(log: pd.DataFrame, assets: Iterable[str], horizons: Iterable[int], min_forward_evidence: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for asset in assets:
        for horizon in horizons:
            subset = log[log["Asset"].astype(str).eq(str(asset)) & log["Horizon"].astype(int).eq(int(horizon))].copy() if not log.empty else pd.DataFrame()
            pending = int(subset["Status"].astype(str).eq("Pending").sum()) if not subset.empty else 0
            matured = int(subset["Status"].astype(str).eq("Matured").sum()) if not subset.empty else 0
            invalid = int(subset["Status"].astype(str).eq("Invalid").sum()) if not subset.empty else 0
            warnings: List[str] = []
            if matured < int(min_forward_evidence):
                warnings.extend(["NotEnoughForwardEvidence", "LowTradeCount"])
            if pending > 0:
                warnings.append("PendingOutcome")
            score = min(matured / max(int(min_forward_evidence), 1) * 100.0, 100.0)
            status = "EnoughForwardEvidence" if matured >= int(min_forward_evidence) else "Not enough forward evidence yet"
            rows.append(
                {
                    "Asset": str(asset),
                    "Horizon": int(horizon),
                    "TotalSignals": int(len(subset)),
                    "PendingSignals": pending,
                    "MaturedSignals": matured,
                    "InvalidSignals": invalid,
                    "CoverageStatus": status,
                    "CoverageScore": round(float(score), 4),
                    "Warnings": _join_warnings(warnings),
                }
            )
    return pd.DataFrame(rows, columns=list(FORWARD_COVERAGE_COLUMNS))


def _warning_table(log: pd.DataFrame, summaries: Iterable[pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not log.empty:
        for _, row in log.iterrows():
            for warning in [w.strip() for w in str(row.get("Warnings", "")).split(";") if w.strip()]:
                severity = "High" if warning in {"MissingExitPrice", "MissingEntryPrice", "ProbabilityStillUnreliable", "BenchmarkDominated", "DrawdownRisk"} else "Medium"
                rows.append(
                    {
                        "SignalId": row.get("SignalId", ""),
                        "Asset": row.get("Asset", ""),
                        "Horizon": row.get("Horizon", np.nan),
                        "WarningType": warning,
                        "Severity": severity,
                        "Message": f"{warning} in forward paper evidence tracker.",
                    }
                )
    for table in summaries:
        if table is None or table.empty or not {"Asset", "Horizon", "Warnings"}.issubset(table.columns):
            continue
        for _, row in table.iterrows():
            for warning in [w.strip() for w in str(row.get("Warnings", "")).split(";") if w.strip()]:
                severity = "High" if warning in {"ProbabilityStillUnreliable", "BenchmarkDominated", "DrawdownRisk"} else "Medium"
                rows.append(
                    {
                        "SignalId": "SUMMARY",
                        "Asset": row.get("Asset", ""),
                        "Horizon": row.get("Horizon", np.nan),
                        "WarningType": warning,
                        "Severity": severity,
                        "Message": f"{warning} in forward paper summary.",
                    }
                )
    return pd.DataFrame(rows, columns=list(FORWARD_WARNING_COLUMNS)).drop_duplicates().reset_index(drop=True) if rows else pd.DataFrame(columns=list(FORWARD_WARNING_COLUMNS))


def _next_actions(coverage: pd.DataFrame, accuracy: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in coverage.iterrows():
        asset = row.get("Asset", "")
        horizon = _safe_int(row.get("Horizon"))
        warnings = str(row.get("Warnings", ""))
        accuracy_row = accuracy[accuracy["Asset"].astype(str).eq(str(asset)) & accuracy["Horizon"].astype(int).eq(horizon)]
        calibration_row = calibration[calibration["Asset"].astype(str).eq(str(asset)) & calibration["Horizon"].astype(int).eq(horizon)]
        action = "Keep collecting forward paper evidence; do not promote or trade live."
        priority = "Medium"
        if "NotEnoughForwardEvidence" in warnings:
            action = "Continue paper logging until enough outcomes mature."
            priority = "High"
        if not accuracy_row.empty and "BenchmarkDominated" in str(accuracy_row.iloc[0].get("Warnings", "")):
            action = "Review benchmark dependence before any further candidate upgrade."
            priority = "High"
        if not calibration_row.empty and "ProbabilityStillUnreliable" in str(calibration_row.iloc[0].get("Warnings", "")):
            action = "Continue probability calibration diagnostics before using confidence filters."
            priority = "High"
        rows.append({"Asset": asset, "Horizon": horizon, "NextResearchAction": action, "ActionPriority": priority})
    return pd.DataFrame(rows, columns=list(FORWARD_NEXT_ACTION_COLUMNS))


def run_forward_paper_evidence_tracker(
    *,
    raw_df: Optional[pd.DataFrame] = None,
    existing_forward_signal_log: Optional[pd.DataFrame] = None,
    prediction_table: Optional[pd.DataFrame] = None,
    true_raw_trade_log_table: Optional[pd.DataFrame] = None,
    probability_calibration_summary: Optional[pd.DataFrame] = None,
    probability_calibration_warnings: Optional[pd.DataFrame] = None,
    assets: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    generate_new_signals: bool = False,
    update_matured_outcomes: bool = True,
    as_of_date: Optional[Any] = None,
    model_depth: str = "fast",
    use_phase5_features: bool = True,
    model_name: Optional[str] = None,
    min_forward_evidence: int = 10,
    random_state: int = 42,
) -> ForwardPaperEvidenceReport:
    """Run the Phase 9 forward paper evidence tracker.

    No outcome is written until the required horizon exists inside price data
    available at ``as_of_date``. If no prediction rows are supplied, model
    signal generation is attempted only when ``generate_new_signals`` is true.
    """
    asset_list = list(assets or get_asset_names())
    horizon_list = [int(h) for h in (horizons or DEFAULT_FORWARD_HORIZONS)]
    as_of = _as_of_timestamp(raw_df, as_of_date)
    settings = {
        "phase": "9",
        "purpose": "forward_paper_trading_evidence_collection_only",
        "production_ready_label_allowed": False,
        "candidate_promotion_allowed": False,
        "as_of_date": str(as_of.date()),
        "assets": asset_list,
        "horizons": horizon_list,
        "generate_new_signals": bool(generate_new_signals),
        "update_matured_outcomes": bool(update_matured_outcomes),
    }
    existing = _normalise_existing_log(existing_forward_signal_log)
    calibration_lookup = _calibration_warning_lookup(probability_calibration_summary, probability_calibration_warnings)

    prediction_input = prediction_table
    if generate_new_signals and prediction_input is None:
        if raw_df is not None and not raw_df.empty:
            prediction_input = generate_forward_model_prediction_rows(
                raw_df=raw_df,
                assets=asset_list,
                horizons=horizon_list,
                model_depth=model_depth,
                use_phase5_features=use_phase5_features,
                model_name=model_name,
                as_of_date=as_of,
                random_state=random_state,
            )
        else:
            prediction_input = _invalid_generation_prediction_rows(
                asset_list,
                horizon_list,
                as_of,
                "FreshPredictionUnavailable: no raw dataset or uploaded prediction source was supplied.",
                model_name=model_name,
            )
    elif prediction_input is None and true_raw_trade_log_table is not None and not true_raw_trade_log_table.empty:
        prediction_input = true_raw_trade_log_table

    new_signal_rows = _prediction_rows_to_signal_log(
        prediction_input,
        raw_df,
        asset_list,
        horizon_list,
        as_of,
        calibration_lookup,
    ) if generate_new_signals and prediction_input is not None else _empty_signal_log()
    combined = _append_without_deleting(existing, new_signal_rows)
    if update_matured_outcomes and not combined.empty:
        combined = update_matured_forward_outcomes(combined, raw_df, as_of_date=as_of)

    if combined.empty:
        report = _empty_report(settings)
        report.asset_horizon_forward_coverage = _coverage_table(combined, asset_list, horizon_list, min_forward_evidence)
        report.forward_accuracy_summary = _accuracy_summary(combined, asset_list, horizon_list, min_forward_evidence)
        report.forward_probability_calibration_summary = _probability_summary(combined, asset_list, horizon_list, min_forward_evidence)
        report.warning_table = _warning_table(combined, [report.asset_horizon_forward_coverage, report.forward_accuracy_summary, report.forward_probability_calibration_summary])
        report.next_research_action_table = _next_actions(report.asset_horizon_forward_coverage, report.forward_accuracy_summary, report.forward_probability_calibration_summary)
        return report

    forward_log = combined[list(FORWARD_SIGNAL_LOG_COLUMNS)].reset_index(drop=True)
    pending = forward_log[forward_log["Status"].astype(str).eq("Pending")].copy().reset_index(drop=True)
    matured = forward_log[forward_log["Status"].astype(str).eq("Matured")].copy().reset_index(drop=True)
    accuracy = _accuracy_summary(forward_log, asset_list, horizon_list, min_forward_evidence)
    probability = _probability_summary(forward_log, asset_list, horizon_list, min_forward_evidence)
    coverage = _coverage_table(forward_log, asset_list, horizon_list, min_forward_evidence)
    warnings = _warning_table(forward_log, [coverage, accuracy, probability])
    next_actions = _next_actions(coverage, accuracy, probability)
    return ForwardPaperEvidenceReport(
        forward_signal_log=forward_log,
        pending_outcome_table=pending,
        matured_outcome_table=matured,
        forward_accuracy_summary=accuracy,
        forward_probability_calibration_summary=probability,
        asset_horizon_forward_coverage=coverage,
        warning_table=warnings,
        next_research_action_table=next_actions,
        settings=settings,
    )


__all__ = [
    "DEFAULT_FORWARD_HORIZONS",
    "FORWARD_ACCURACY_SUMMARY_COLUMNS",
    "FORWARD_COVERAGE_COLUMNS",
    "FORWARD_NEXT_ACTION_COLUMNS",
    "FORWARD_PROBABILITY_CALIBRATION_COLUMNS",
    "FORWARD_SIGNAL_LOG_COLUMNS",
    "FORWARD_WARNING_COLUMNS",
    "ForwardPaperEvidenceReport",
    "generate_forward_model_prediction_rows",
    "run_forward_paper_evidence_tracker",
    "update_matured_forward_outcomes",
]
