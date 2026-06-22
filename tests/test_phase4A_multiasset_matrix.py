from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.multiasset_validation import run_multiasset_validation, summarize_asset_status


def main():
    # Keep smoke test fast but genuinely multi-asset.
    report = run_multiasset_validation(
        asset_names=["Gold", "Bitcoin"],
        model_set="fast",
        include_walk_forward=False,
        use_cache=True,
    )

    assert report.asset_summary is not None and not report.asset_summary.empty
    assert set(report.asset_summary["Asset"]) == {"Gold", "Bitcoin"}
    assert "TrustScore" in report.asset_summary.columns
    assert "AssetVerdict" in report.asset_summary.columns
    assert report.model_leaderboard is not None and not report.model_leaderboard.empty
    assert set(report.model_leaderboard["Asset"]).issubset({"Gold", "Bitcoin"})
    assert report.baseline_leaderboard is not None and not report.baseline_leaderboard.empty
    assert report.leakage_matrix is not None and not report.leakage_matrix.empty

    status = summarize_asset_status(report.asset_summary)
    assert isinstance(status, dict)

    print("Multi-asset asset summary:")
    print(report.asset_summary)
    print("\nStatus counts:")
    print(status)
    print("\nPhase 4A multi-asset validation matrix smoke test passed.")


if __name__ == "__main__":
    main()
