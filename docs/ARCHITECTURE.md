# Architecture

## System Map

```text
Public market sources / cached master dataset / checked-in demo artifacts
                              |
                              v
                   Data loader and alignment
                              |
                              v
          Feature engineering and feature intelligence
                              |
                              v
         Direct-horizon forecasting and probability output
                              |
             +----------------+----------------+
             |                                 |
             v                                 v
   Signal and walk-forward validation    Source/freshness audit
             |                                 |
             v                                 |
  Risk, cost, regime, replay, benchmark <------+
             |
             v
 Watchlist -> evidence grading -> personalized research plans
             |
             v
 User-owned SQLite/Postgres-ready persistence -> Streamlit UI
```

## Data Flow

Market rows are aligned chronologically before feature generation. Direct targets use future shifts only in target construction; target columns are excluded from features. Scalers and model selection remain train/validation scoped. The UI reads session results or explicitly labeled saved/cached artifacts and never silently calls them live data.

## UI Flow

Public visitors see the Research Dashboard preview, login, and methodology only. Authenticated users receive grouped Dashboard, Research, Planning, Account, Info, and Advanced navigation. Premium page labels are normalized to stable internal routes so old sessions and saved page preferences remain compatible. Asset selection is page-scoped and appears only where an asset context is required.

## Authentication Flow

1. `auth_manager` checks Streamlit secrets for `SUPABASE_URL` and `SUPABASE_ANON_KEY`.
2. When configured, signup and password sign-in use Supabase Auth. Workspace access requires a verified provider user.
3. The provider identity maps to a stable application row through `auth_provider` plus `auth_user_id`.
4. Without secrets, the app announces Local development auth mode and uses salted PBKDF2 password hashes in SQLite. It does not claim email verification.
5. Session logout clears application identity state. No broker, bank, execution, or trading API credentials are accepted.

## User-Owned Data Flow

Profiles, preferences, plan runs, plan rows, and research history carry the application `user_id`. Every read is filtered by that ID. The schema is normalized and migration-safe, making it straightforward to move from SQLite to Postgres without changing ownership semantics.

## Research History Flow

A generated personalized plan combines the current prediction snapshot, watchlist classification, edge evidence, and plan output into one row per asset. Placeholder-only data is rejected. Immutable runs are saved per user, and the latest two runs are compared without rerunning or changing historic scores.

## Data Provenance

Displayed market and forecast evidence carries a source label, latest available date, and freshness state where available. The product distinguishes latest refreshed snapshots, saved research snapshots, cached dataset prices, stale evidence, and unavailable estimates. Fallbacks remain visible in Snapshot Source Diagnostics instead of silently presenting older data as live.

## Deployment

Streamlit Cloud runs `app.py` on Python 3.11. Checked-in demo snapshots provide an honest public preview. Private secrets are supplied through Streamlit deployment settings. CI compiles the application and runs all tests without requiring external auth credentials.

## Security Notes

- Only Supabase URL and anon key are used by the client integration.
- Service-role keys and real secrets are never stored in the repository.
- Local passwords are salted PBKDF2 hashes, never plaintext.
- User-facing diagnostics redact common secret/token patterns.
- The product has no broker execution path and remains research-only.
