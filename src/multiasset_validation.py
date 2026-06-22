# src/multiasset_validation.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import time

import numpy as np
import pandas as pd

from src.asset_config import get_asset_names, get_target_column
from src.data_loader import DataLoader
from src.indicators import TechnicalIndicators
from src.feature_engineering import FeatureEngineer
from src.feature_intelligence import add_phase5_feature_intelligence
from src.preprocessing import Preprocessor
from src.train import ModelTrainer
from src.baselines import price_baseline_leaderboard
from src.research_validation import build_validation_report, walk_forward_validate_model


@dataclass
class MultiAssetValidationReport:
    """Container for the multi-asset research validation matrix."""

    asset_summary: pd.DataFrame
    model_leaderboard: pd.DataFrame
    baseline_leaderboard: pd.DataFrame
    leakage_matrix: pd.DataFrame
    walk_forward_summary: pd.DataFrame
    errors: pd.DataFrame


ProgressCallback = Optional[Callable[[int, int, str], None]]


def _target_prefix(target_col: str) -> str:
    return str(target_col).replace("_Close", "")


def _build_features_for_asset(raw_df: pd.DataFrame, target_col: str, use_phase5_features: bool = True) -> pd.DataFrame:
    """
    Build target-specific indicators/features for one asset.

    This mirrors the Streamlit pipeline but lives in src/ so the multi-asset
    validation can run outside the app and inside tests.
    """
    prefix = _target_prefix(target_col)
    df = TechnicalIndicators(prefix=prefix).add_all(raw_df.copy())

    # Keep latest target rows even when supporting assets update later.
    # This is time-series safe: only carries past observations forward.
    df = df.sort_index().ffill()

    fe = FeatureEngineer(target_col=target_col)
    df_features = fe.build_features(df)

    if use_phase5_features:
        df_features = add_phase5_feature_intelligence(df_features, target_col=target_col)

    return df_features


def _selected_training_functions(trainer: ModelTrainer, model_set: str) -> List[Tuple[str, Callable]]:
    """
    Return model training functions for a requested validation depth.

    The purpose is to support both quick smoke tests and serious all-asset
    research runs without forcing the user to wait for the full suite every time.
    """
    key = str(model_set).lower().strip()

    if key in {"fast", "fast audit", "quick", "smoke"}:
        return [
            ("Linear Regression", trainer.train_linear_regression),
            ("Decision Tree", trainer.train_decision_tree),
        ]

    if key in {"core", "core research", "balanced", "research"}:
        return [
            ("Linear Regression", trainer.train_linear_regression),
            ("Random Forest", trainer.train_random_forest),
            ("XGBoost", trainer.train_xgboost),
            ("LightGBM", trainer.train_lightgbm),
            ("CatBoost", trainer.train_catboost),
        ]

    if key in {"full", "full ml", "full suite"}:
        return [
            ("Linear Regression", trainer.train_linear_regression),
            ("Decision Tree", trainer.train_decision_tree),
            ("Random Forest", trainer.train_random_forest),
            ("XGBoost", trainer.train_xgboost),
            ("LightGBM", trainer.train_lightgbm),
            ("CatBoost", trainer.train_catboost),
            ("SVR", trainer.train_svr),
        ]

    raise ValueError(f"Unknown model_set: {model_set}. Use fast, core, or full.")


def _train_requested_models(data, pp: Preprocessor, model_set: str) -> ModelTrainer:
    trainer = ModelTrainer(use_optuna=False, target_scaler=data.target_scaler, preprocessor=pp)

    for name, fn in _selected_training_functions(trainer, model_set):
        try:
            result = fn(data)
            trainer.results[result.name] = result
        except Exception as exc:
            # Keep the all-asset matrix robust: one missing optional library or
            # one failed model should not stop every asset's validation.
            trainer.results[f"{name} FAILED"] = None
            print(f"[multiasset] {name} failed: {exc}")

    trainer.results = {k: v for k, v in trainer.results.items() if v is not None}
    return trainer


def _extract_asset_summary(
    *,
    asset: str,
    target_col: str,
    data,
    validation_report,
    elapsed_sec: float,
) -> Dict:
    trust = validation_report.trust_scores.copy()
    baselines = validation_report.baseline_board.copy()

    best = trust.iloc[0].to_dict() if not trust.empty else {}
    naive = baselines[baselines["Model"].astype(str).str.contains("Naive", case=False, na=False)]
    naive_rmse = float(naive.iloc[0]["RMSE"]) if not naive.empty else np.nan

    latest_date = None
    try:
        latest_date = pd.Timestamp(data.test_index[-1]).date().isoformat()
    except Exception:
        latest_date = ""

    score = float(best.get("TrustScore", np.nan)) if best else np.nan
    if np.isfinite(score) and score >= 75:
        asset_status = "High-trust candidate"
    elif np.isfinite(score) and score >= 55:
        asset_status = "Medium-trust candidate"
    elif np.isfinite(score) and score >= 35:
        asset_status = "Low-trust / research only"
    else:
        asset_status = "Do not trust for signals"

    return {
        "Asset": asset,
        "Target": target_col,
        "Rows": int(len(data.df_clean)) if hasattr(data, "df_clean") else int(len(data.prices_train) + len(data.prices_val) + len(data.prices_test)),
        "TrainRows": int(len(data.X_train)),
        "ValRows": int(len(data.X_val)),
        "TestRows": int(len(data.X_test)),
        "LatestTargetDate": latest_date,
        "BestModel": best.get("Model", ""),
        "TrustScore": round(score, 2) if np.isfinite(score) else np.nan,
        "AssetVerdict": asset_status,
        "ModelVerdict": best.get("Verdict", ""),
        "BestRMSE": best.get("RMSE", np.nan),
        "NaiveRMSE": round(naive_rmse, 4) if np.isfinite(naive_rmse) else np.nan,
        "RMSE_vs_Naive_%": best.get("RMSE_vs_Naive_%", np.nan),
        "DirectionalAccuracy": best.get("DirectionalAccuracy", np.nan),
        "Sharpe_LongOnly": best.get("Sharpe_LongOnly", np.nan),
        "MaxDD_LongOnly_%": best.get("MaxDD_LongOnly_%", np.nan),
        "Strategy_vs_BuyHold_%": best.get("Strategy_vs_BuyHold_%", np.nan),
        "ElapsedSec": round(float(elapsed_sec), 2),
    }


def run_multiasset_validation(
    *,
    raw_df: Optional[pd.DataFrame] = None,
    start_date: str = "2015-01-01",
    asset_names: Optional[Iterable[str]] = None,
    model_set: str = "fast",
    use_cache: bool = True,
    include_walk_forward: bool = False,
    walk_forward_splits: int = 3,
    progress_callback: ProgressCallback = None,
    use_phase5_features: bool = True,
) -> MultiAssetValidationReport:
    """
    Run the Phase 4A research validation matrix across all configured assets.

    Parameters
    ----------
    raw_df:
        Optional already-loaded master market dataset. If omitted, DataLoader is used.
    asset_names:
        Assets to validate. Defaults to all assets from asset_config.
    model_set:
        fast  -> Linear Regression + Decision Tree
        core  -> Linear Regression + Random Forest + XGBoost + LightGBM + CatBoost
        full  -> all ML models including SVR
    include_walk_forward:
        If True, run a small walk-forward check for the best model per asset.
        This is slower, so it is optional.
    progress_callback:
        Optional callback receiving (current_asset_number, total_assets, message).
    use_phase5_features:
        If True, add Phase 5 FI_* feature intelligence before preprocessing/training.
    """
    if raw_df is None:
        raw_df = DataLoader(start_date=start_date, end_date=None).load_all(use_cache=use_cache)

    assets = list(asset_names) if asset_names is not None else get_asset_names()
    total = len(assets)

    asset_rows: List[Dict] = []
    model_frames: List[pd.DataFrame] = []
    baseline_frames: List[pd.DataFrame] = []
    leakage_frames: List[pd.DataFrame] = []
    wf_rows: List[Dict] = []
    error_rows: List[Dict] = []

    for i, asset in enumerate(assets, start=1):
        t0 = time.perf_counter()
        target_col = get_target_column(asset)

        if progress_callback:
            progress_callback(i, total, f"Validating {asset} ({target_col})...")

        try:
            if target_col not in raw_df.columns:
                raise ValueError(f"Target column {target_col} not found in master dataset")

            df_features = _build_features_for_asset(raw_df, target_col=target_col, use_phase5_features=use_phase5_features)
            pp = Preprocessor(target_col=target_col)
            data = pp.run(df_features)
            trainer = _train_requested_models(data, pp, model_set=model_set)

            if not trainer.results:
                raise RuntimeError("No models trained successfully for this asset")

            report = build_validation_report(trainer, data, df_features)
            elapsed = time.perf_counter() - t0

            asset_rows.append(
                _extract_asset_summary(
                    asset=asset,
                    target_col=target_col,
                    data=data,
                    validation_report=report,
                    elapsed_sec=elapsed,
                )
            )

            trust = report.trust_scores.copy()
            if not trust.empty:
                trust.insert(0, "Asset", asset)
                trust.insert(1, "Target", target_col)
                model_frames.append(trust)

            baseline = report.baseline_board.copy()
            if not baseline.empty:
                baseline.insert(0, "Asset", asset)
                baseline.insert(1, "Target", target_col)
                baseline_frames.append(baseline)

            leakage = report.leakage_report.copy()
            if not leakage.empty:
                leakage.insert(0, "Asset", asset)
                leakage.insert(1, "Target", target_col)
                leakage_frames.append(leakage)

            if include_walk_forward:
                best_model_name = str(report.trust_scores.iloc[0]["Model"]) if not report.trust_scores.empty else None
                if best_model_name and best_model_name in trainer.results:
                    try:
                        folds, summary = walk_forward_validate_model(
                            trainer.results[best_model_name].model,
                            best_model_name,
                            data,
                            pp,
                            n_splits=int(walk_forward_splits),
                            max_train_size=None,
                        )
                        wf_rows.append(
                            {
                                "Asset": asset,
                                "Target": target_col,
                                "Model": best_model_name,
                                **summary,
                            }
                        )
                    except Exception as exc:
                        wf_rows.append(
                            {
                                "Asset": asset,
                                "Target": target_col,
                                "Model": best_model_name,
                                "Error": str(exc),
                            }
                        )

        except Exception as exc:
            error_rows.append(
                {
                    "Asset": asset,
                    "Target": target_col,
                    "Error": str(exc),
                }
            )

    asset_summary = pd.DataFrame(asset_rows)
    if not asset_summary.empty:
        asset_summary = asset_summary.sort_values(["TrustScore", "RMSE_vs_Naive_%"], ascending=False).reset_index(drop=True)
        asset_summary.insert(0, "Rank", range(1, len(asset_summary) + 1))

    model_leaderboard = pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame()
    if not model_leaderboard.empty:
        model_leaderboard = model_leaderboard.sort_values(["TrustScore", "RMSE_vs_Naive_%"], ascending=False).reset_index(drop=True)

    baseline_leaderboard = pd.concat(baseline_frames, ignore_index=True) if baseline_frames else pd.DataFrame()
    leakage_matrix = pd.concat(leakage_frames, ignore_index=True) if leakage_frames else pd.DataFrame()
    walk_forward_summary = pd.DataFrame(wf_rows)
    errors = pd.DataFrame(error_rows)

    return MultiAssetValidationReport(
        asset_summary=asset_summary,
        model_leaderboard=model_leaderboard,
        baseline_leaderboard=baseline_leaderboard,
        leakage_matrix=leakage_matrix,
        walk_forward_summary=walk_forward_summary,
        errors=errors,
    )


def summarize_asset_status(asset_summary: pd.DataFrame) -> Dict[str, int]:
    """Small helper for dashboard metric cards."""
    if asset_summary is None or asset_summary.empty or "AssetVerdict" not in asset_summary.columns:
        return {"High": 0, "Medium": 0, "Low": 0, "DoNotTrust": 0}

    verdicts = asset_summary["AssetVerdict"].astype(str)
    return {
        "High": int(verdicts.str.contains("High", case=False, na=False).sum()),
        "Medium": int(verdicts.str.contains("Medium", case=False, na=False).sum()),
        "Low": int(verdicts.str.contains("Low", case=False, na=False).sum()),
        "DoNotTrust": int(verdicts.str.contains("Do not trust", case=False, na=False).sum()),
    }
