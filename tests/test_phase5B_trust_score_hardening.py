from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.research_validation import model_trust_score


def test_terrible_model_gets_very_low_trust_score():
    score, verdict = model_trust_score(
        rmse_improvement_pct=-500.0,
        directional_accuracy=48.0,
        sharpe_ratio=2.0,
        max_drawdown_pct=-3.0,
        strategy_vs_buy_hold_pct=-30.0,
        overfit_gap_pct=150.0,
    )

    assert score <= 10.0
    assert verdict == "Do not trust for signals"


def test_extreme_rmse_underperformance_is_near_zero():
    for rmse_improvement in [-500.0, -1000.0, -2500.0]:
        score, verdict = model_trust_score(
            rmse_improvement_pct=rmse_improvement,
            directional_accuracy=62.0,
            sharpe_ratio=2.0,
            max_drawdown_pct=-3.0,
            strategy_vs_buy_hold_pct=10.0,
            overfit_gap_pct=20.0,
        )

        assert score <= 2.0
        assert verdict == "Do not trust for signals"


def test_medium_trust_requires_quality_gates():
    score, verdict = model_trust_score(
        rmse_improvement_pct=8.0,
        directional_accuracy=49.0,
        sharpe_ratio=2.0,
        max_drawdown_pct=-3.0,
        strategy_vs_buy_hold_pct=8.0,
        overfit_gap_pct=20.0,
    )

    assert score < 55.0
    assert verdict != "Medium trust / needs monitoring"
    assert verdict != "High trust candidate"


if __name__ == "__main__":
    test_terrible_model_gets_very_low_trust_score()
    test_extreme_rmse_underperformance_is_near_zero()
    test_medium_trust_requires_quality_gates()
    print("Phase 5B trust score hardening tests passed.")
