from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data_loader import DataLoader
from src.indicators import TechnicalIndicators
from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor
from src.train import ModelTrainer
from src.research_validation import (
    build_validation_report,
    regime_performance,
    walk_forward_validate_model,
)


def run_for_asset(asset_name: str, target_col: str):
    df = DataLoader(start_date="2015-01-01", end_date=None).load_all(use_cache=True)
    prefix = target_col.replace("_Close", "")
    df = TechnicalIndicators(prefix=prefix).add_all(df)
    df = df.sort_index().ffill()
    df = FeatureEngineer(target_col=target_col).build_features(df)

    pp = Preprocessor(target_col=target_col)
    data = pp.run(df)

    # Use Linear Regression for a fast smoke test. The app can run all models.
    trainer = ModelTrainer(use_optuna=False, target_scaler=data.target_scaler, preprocessor=pp)
    trainer.train_linear_regression(data)
    trainer.results["Linear Regression"] = trainer.train_linear_regression(data)

    report = build_validation_report(trainer, data, df)
    assert not report.trust_scores.empty
    assert not report.baseline_board.empty
    assert not report.leakage_report.empty
    assert "TrustScore" in report.trust_scores.columns

    result = trainer.results["Linear Regression"]
    regime = regime_performance(data, result.predictions_test)
    assert not regime.empty

    folds, summary = walk_forward_validate_model(result.model, "Linear Regression", data, pp, n_splits=3)
    assert len(folds) == 3
    assert np.isfinite(summary["MeanRMSE"])

    print(f"OK {asset_name}: trust rows={len(report.trust_scores)}, leakage checks={len(report.leakage_report)}, wf_folds={len(folds)}")


if __name__ == "__main__":
    run_for_asset("Gold", "Gold_Close")
    print("Phase 4 research validation smoke test passed.")
