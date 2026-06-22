from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DataLoader
from src.indicators import TechnicalIndicators
from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor
from src.baselines import price_baseline_leaderboard
from src.directional_models import (
    train_directional_models,
    directional_leaderboard,
    directional_baseline_leaderboard,
    run_directional_probability_backtest,
)


def build_gold_data():
    df = DataLoader(start_date="2015-01-01", end_date=None).load_all(use_cache=True)
    df = TechnicalIndicators(prefix="Gold").add_all(df)
    df = df.sort_index().ffill()
    df = FeatureEngineer(target_col="Gold_Close").build_features(df)
    pp = Preprocessor(target_col="Gold_Close")
    data = pp.run(df)
    return pp, data


def main():
    pp, data = build_gold_data()

    baseline_board = price_baseline_leaderboard(data)
    assert not baseline_board.empty
    assert "Naive: tomorrow = today" in baseline_board["Model"].values
    print("Price baselines OK")
    print(baseline_board)

    direction_base = directional_baseline_leaderboard(data, pp)
    assert not direction_base.empty
    print("Directional baselines OK")
    print(direction_base)

    # Fast smoke test: sklearn models only.
    results = train_directional_models(data, pp, include_heavy=False)
    board = directional_leaderboard(results)
    assert not board.empty
    print("Directional models OK")
    print(board)

    first_name = board.iloc[0]["Model"]
    proba = results[first_name].probabilities_test
    metrics, equity = run_directional_probability_backtest(
        data,
        proba,
        probability_threshold=0.55,
        transaction_cost=0.001,
        allow_short=False,
    )
    assert len(equity) == len(data.prices_test)
    print("Probability backtest OK")
    print(metrics)

    print("Phase 3 baseline + directional smoke test passed.")


if __name__ == "__main__":
    main()
