# tests/test_phase1_modules.py
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd

from src.prediction_ranges import calculate_prediction_range
from src.signals import generate_trading_signal
from src.backtesting import run_backtest_from_predictions
from src.risk import generate_risk_report
from src.benchmarks import add_baseline_predictions, compare_against_baselines
from src.walk_forward import expanding_window_splits
from src.direct_forecast import add_direct_horizon_targets
from src.portfolio import simple_score_allocation
from src.data_quality import check_data_quality


def main():
    print("Testing prediction range...")
    pred_range = calculate_prediction_range(
        last_price=4215,
        predicted_price=4296.65,
        rmse=61.39,
        model_used="CatBoost",
    )
    print(pred_range.to_dict())

    print("\nTesting signals...")
    signal = generate_trading_signal(
        predicted_return_pct=pred_range.predicted_return_pct,
        lower_return_pct=pred_range.lower_return_pct,
        upper_return_pct=pred_range.upper_return_pct,
    )
    print(signal.to_dict())

    print("\nTesting backtesting...")
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.01, size=len(dates))
    prices = 2000 * np.cumprod(1 + returns)
    predicted_returns = returns + rng.normal(0, 0.008, size=len(dates))

    demo = pd.DataFrame(
        {
            "Gold_Close": prices,
            "Predicted_Return": predicted_returns,
        },
        index=dates,
    )

    bt = run_backtest_from_predictions(
        demo,
        price_col="Gold_Close",
        predicted_return_col="Predicted_Return",
    )
    print(bt.metrics)

    print("\nTesting risk...")
    print(generate_risk_report(demo["Gold_Close"]).to_dict())

    print("\nTesting benchmarks...")
    bench = add_baseline_predictions(demo, price_col="Gold_Close")
    bench["CatBoost_Prediction"] = bench["Gold_Close"].shift(1) * 1.001
    print(compare_against_baselines(bench, actual_col="Gold_Close", model_prediction_col="CatBoost_Prediction"))

    print("\nTesting walk-forward...")
    print(len(expanding_window_splits(1000, initial_train_size=500, test_size=100, max_folds=5)))

    print("\nTesting direct forecasts...")
    print(add_direct_horizon_targets(demo, price_col="Gold_Close").tail())

    print("\nTesting portfolio allocation...")
    print(simple_score_allocation(
        {"Gold": 0.012, "Silver": 0.008, "Bitcoin": 0.025, "S&P 500": 0.006},
        {"Gold": 0.16, "Silver": 0.25, "Bitcoin": 0.65, "S&P 500": 0.18},
    ))

    print("\nTesting data quality...")
    print(check_data_quality(demo, asset_columns=["Gold_Close", "Silver_Close"]).to_dict())

    print("\nAll module tests completed.")


if __name__ == "__main__":
    main()
