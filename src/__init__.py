"""
src — Gold Price Prediction Source Package
==========================================
Modular ML/DL pipeline for gold price forecasting.

Submodules
----------
config_loader   : YAML config singleton
logger          : Centralized rotating logger
data_loader     : yfinance / FRED data ingestion
preprocessing   : Cleaning, scaling, train-test split
indicators      : Technical indicator generation (TA-Lib style)
feature_engineering : Lag, rolling, ratio & calendar features
train           : ML & DL model training orchestrator
predict         : Inference & 30-day rolling forecast
visualization   : Professional Matplotlib / Plotly charts
utils           : Shared helper functions
"""

__version__ = "1.0.0"
__author__  = "B.Tech Final Year Project"
