# Storage Layout — Aion Analytics

This file documents the agreed-on storage layout for nightly & intraday ML, logs, and analytics.

- `data/`           : raw inputs (daily/intraday bars, fundamentals, news, social)
- `data_cache/`     : rolling cache snapshots (daily + intraday)
- `ml_data/`        : nightly ML datasets, models, predictions (180-day retention)
- `ml_data_dt/`     : intraday ML datasets, models, predictions, and replay support (180-day retention)
- `analytics/`      : accuracy, PnL, bot performance metrics
- `logs/`           : structured log files for backend, scheduler, bots, intraday, nightly

Nightly retention: 180 days
Intraday retention: 180 days

This file is the source of truth for backend upgrades and future refactors.
