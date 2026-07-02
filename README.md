# Multi-Asset Quant Research Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)
[![CI](https://github.com/poojanbhardwaj/GoldPricePrediction-QuantSystem/actions/workflows/ci.yml/badge.svg)](https://github.com/poojanbhardwaj/GoldPricePrediction-QuantSystem/actions)
[![Research Only](https://img.shields.io/badge/use-research--only-lightgrey)](#safety)

A portfolio-grade Streamlit system for honest, multi-asset market research. It combines saved research snapshots, direct-horizon forecasts, candidate ranking, evidence-of-edge checks, cost and risk diagnostics, personalized plans, and user-owned research history.

The platform supports **Gold, Silver, Crude Oil, Bitcoin, S&P 500, and GLD** across 1D, 5D, 10D, 20D, and 30D research horizons.

## Product Highlights

- Multi-asset research snapshots with explicit source and freshness labels
- Leakage-aware forecasting and validation against naive/passive baselines
- Candidate Watchlist and Evidence of Edge views that keep weak results visible
- Cost-aware, risk-aware personalized research plans
- Paper-signal tracking, walk-forward validation, replay, and benchmark audits
- User-owned profiles, saved plans, preferences, and research-history comparisons
- Optional Supabase verified-email auth with a clearly labeled local SQLite fallback
- Premium public preview with the full workspace protected by application login
- Extensive automated regression and quality-gate coverage

## Architecture

```text
Market data and checked-in demo snapshots
                  |
                  v
Data loading -> features -> direct forecasts -> signal validation
                  |                              |
                  v                              v
          source/freshness audit        costs, risk, benchmarks
                  \______________________________/
                                 |
                                 v
               watchlist -> evidence -> personalized plans
                                 |
                                 v
            user-owned plans, preferences, and research history
                                 |
                                 v
                         Streamlit product shell
```

Primary modules include `data_loader`, feature engineering, forecasting, signal and replay engines, risk intelligence, `auth_manager`, `user_platform`, and `research_history`. See [Architecture](docs/ARCHITECTURE.md) for the complete data and authentication flows.

## Screenshots

Screenshot targets for the deployed portfolio:

- `docs/screenshots/public_preview.png`
- `docs/screenshots/login.png`
- `docs/screenshots/watchlist.png`
- `docs/screenshots/research_history.png`
- `docs/screenshots/account_settings.png`

These paths are intentional placeholders until reviewed deployment screenshots are captured.

See the [Screenshot Checklist](docs/SCREENSHOTS.md) for capture guidance and privacy checks.

## Local Setup

```powershell
git clone https://github.com/poojanbhardwaj/GoldPricePrediction-QuantSystem.git
cd GoldPricePrediction-QuantSystem
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The application works without Supabase in **Local development auth mode**. Local passwords use salted PBKDF2 hashes and email ownership is not claimed as verified.

### Optional Verified Email

Copy `.streamlit/secrets.example.toml` to the untracked `.streamlit/secrets.toml` and provide only:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your-anon-key"
```

Use the public anon key, never a service-role key. With Supabase configured, new users must verify their email before the research workspace unlocks.

## Testing

```powershell
python -m compileall app.py src
python -m pytest tests -q
```

GitHub Actions runs both commands on Python 3.11 for every push and pull request. Supabase tests use mocks and do not require deployment secrets.

## Deployment

The app is designed for Streamlit Cloud. Configure secrets through the deployment settings, not in Git. Local SQLite is a development fallback; production user data can move to Postgres while retaining the stable `user_id`, `auth_provider`, and `auth_user_id` ownership model.

Never commit `.streamlit/secrets.toml`, `.env*`, `data/app.db`, generated model outputs, or private user exports.

## Safety

This project is **research-only**, **not financial advice**, and has no real-money or broker execution path. Cached or saved values are labeled as such, weak and rejected candidates remain visible, and forecasts are uncertain research evidence rather than promises. Returns are never assured.

## Roadmap

- Move user-owned persistence from local SQLite to managed Postgres
- Expand Supabase production-auth operations and account recovery
- Add deeper portfolio attribution and monitoring
- Add research alerts without execution capability
- Capture and maintain deployment screenshots

## Recruiter Demo

Use the [Demo Script](docs/DEMO_SCRIPT.md) for a five-minute tour of the public preview, verified/auth-ready workspace, multi-asset evidence, research history, and account settings.
