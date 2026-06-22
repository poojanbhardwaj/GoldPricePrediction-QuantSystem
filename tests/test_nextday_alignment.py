from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DataLoader
from src.indicators import TechnicalIndicators
from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor
from src.asset_config import get_target_column


def target_prefix(target_col: str) -> str:
    return target_col.replace("_Close", "")


def check_asset(asset_name: str):
    target_col = get_target_column(asset_name)
    df_raw = DataLoader(start_date="2015-01-01", end_date=None).load_all(use_cache=True)
    ti = TechnicalIndicators(prefix=target_prefix(target_col))
    df = ti.add_all(df_raw)
    df = df.sort_index().ffill()
    fe = FeatureEngineer(target_col=target_col)
    df = fe.build_features(df)

    pp = Preprocessor(target_col=target_col)
    data = pp.run(df)

    assert len(data.X_test) == len(data.y_test) == len(data.prices_test) == len(data.test_index)
    assert len(data.X_train) == len(data.y_train) == len(data.prices_train)
    assert np.isfinite(data.prices_test).all()
    assert np.isfinite(data.X_test).all()

    # After next-day alignment, the last labelled target date should be the
    # latest feature-engineered date. The latest feature row itself is used by
    # the app for the future prediction.
    assert data.test_index[-1] == df.index[-1], (data.test_index[-1], df.index[-1])

    anchors = np.concatenate([[data.last_price_before_test], data.prices_test[:-1]])
    actual_next_returns = data.prices_test / anchors - 1.0
    assert np.isfinite(actual_next_returns).all()

    print(
        f"OK {asset_name:10s} target={target_col:12s} "
        f"rows={len(data.X_train)+len(data.X_val)+len(data.X_test)} "
        f"test_last_target_date={data.test_index[-1].date()}"
    )


if __name__ == "__main__":
    for asset in ["Gold", "Bitcoin"]:
        check_asset(asset)
    print("Next-day alignment test passed.")
