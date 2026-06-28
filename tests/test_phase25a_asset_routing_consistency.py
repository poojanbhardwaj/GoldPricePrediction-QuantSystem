from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_30_day_forecast_passes_selected_asset_target_to_visualizer():
    app_source = _source("app.py")
    page = app_source.split('elif page == "📅 30-Day Forecast":', 1)[1]
    assert "active_target_col = getattr(pp" in page
    assert "target_col=active_target_col" in page
    assert "asset_label=selected_asset" in page
    assert "plot_forecast_plotly(forecast_df, df_features, n_history_days=90)" not in page


def test_visualizer_forecast_does_not_silently_default_history_to_gold():
    viz_source = _source("src/visualization.py")
    assert 'target_col: Optional[str] = None' in viz_source
    assert 'target_col must be provided when plotting forecast history.' in viz_source
    assert 'Day Gold Price Forecast' not in viz_source


def test_dataset_download_filename_is_multi_asset_not_gold_only():
    app_source = _source("app.py")
    assert 'gold_master_dataset.csv' not in app_source
    assert 'multi_asset_master_dataset.csv' in app_source


def test_single_asset_research_pages_default_to_sidebar_asset_not_hardcoded_smoke_assets():
    app_source = _source("app.py")
    assert 'diag_default_asset = diag_assets.index("Silver")' not in app_source
    assert 'rc_default_asset = rc_assets.index("Silver")' not in app_source
    assert 'index=get_asset_names().index("Crude Oil")' not in app_source
    assert 'diag_default_asset = diag_assets.index(selected_asset)' in app_source
    assert 'rc_default_asset = rc_assets.index(selected_asset)' in app_source
    assert 'index=get_asset_names().index(selected_asset)' in app_source
