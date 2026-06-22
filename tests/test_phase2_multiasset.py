# tests/test_phase2_multiasset.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.asset_config import ASSETS, get_target_column
from src.data_loader import DataLoader
from src.indicators import TechnicalIndicators
from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor


def main():
    df_raw = DataLoader(start_date="2018-01-01", end_date=None).load_all(use_cache=True)

    for asset_name in ASSETS:
        target_col = get_target_column(asset_name)
        prefix = target_col.replace("_Close", "")

        if target_col not in df_raw.columns:
            raise AssertionError(f"{asset_name}: missing {target_col}")

        df_ind = TechnicalIndicators(prefix=prefix).add_all(df_raw.copy())
        df_ind = df_ind.sort_index().ffill()
        df_features = FeatureEngineer(target_col=target_col).build_features(df_ind)

        pp = Preprocessor(target_col=target_col)
        data = pp.run(df_features)

        assert data.target_col == target_col
        assert len(data.feature_cols) > 10
        assert len(data.X_train) > 0
        assert len(data.X_test) > 0

        print(
            f"OK {asset_name:10s} -> {target_col:12s} | "
            f"rows={len(df_features)} | features={len(data.feature_cols)} | "
            f"last={df_features.index[-1].date()}"
        )

    print("Phase 2 multi-asset smoke test passed.")


if __name__ == "__main__":
    main()
