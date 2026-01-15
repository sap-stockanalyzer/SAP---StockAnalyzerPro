<<<<<<< HEAD
# Aion Analytics

Aion Analytics is a full-stack, multi-layer trading research and execution platform.

It runs **nightly EOD intelligence** and **intraday day-trading**, with:

- Human-like market context & regime awareness
- Multi-model ML stack (LightGBM + future LSTM/Transformer)
- Policy & execution layers (EOD bots + intraday bot)
- Continuous learning and drift monitoring
- Frontend dashboard (Next.js) for live, insights, bots, and system status

Core pipeline:

> **Context → Features → Models → Ensemble → Policy → Execution → Learning**

---

## 1. High-Level Architecture

The repo is structured around two main Python backends plus a frontend:

- `backend/` – **Nightly + swing / EOD brain**
- `dt_backend/` – **Intraday + day-trading brain**
- `frontend/` – Next.js UI (dashboard, insights, live, bots, system)
- `utils/` – Shared utilities (logging, progress bars, JSON, time utils)

They share a common storage layout under `data/`, `ml_data/`, `ml_data_dt/`, `analytics/`, and `logs/`.

### 1.1 Nightly vs Intraday

- **Nightly (backend)**:
  - Runs once per day (or on schedule) after market close / pre-market
  - Refreshes fundamentals, macro, sentiment
  - Builds EOD features and datasets
  - Trains / evaluates models
  - Produces nightly predictions & insights
  - Drives swing bots (1w, 2w, 4w)

- **Intraday (dt_backend)**:
  - Runs during market hours
  - Pulls 1m/5m bars (Alpaca, with fallbacks)
  - Maintains intraday rolling state
  - Computes technical + contextual intraday features
  - Runs intraday ML models (LightGBM today, LSTM/Transformer later)
  - Executes intraday trades against broker (paper/live)
  - Logs PnL and bot actions

---

## 2. Project Layout

From `Aion_Analytics_Folder_Map.txt`:

```text
Aion_Analytics/
│
├── backend/
│   ├── routers/
│   │   ├── live_prices_router.py
│   │   ├── insights_router.py
│   │   ├── system_router.py
│   │   └── __init__.py
│   │
│   ├── services/
│   │   ├── news_fetcher.py
│   │   ├── news_fetcher_loop.py
│   │   ├── fundamentals_fetcher.py
│   │   ├── macro_fetcher.py
│   │   ├── insights_builder.py
│   │   ├── prediction_logger.py
│   │   ├── social_sentiment_fetcher.py
│   │   ├── backfill_history.py
│   │   ├── scheduler_runner.py
│   │   ├── nightly_job.py
│   │   └── __init__.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── data_pipeline.py
│   │   ├── context_state.py
│   │   ├── regime_detector.py
│   │   ├── policy_engine.py
│   │   ├── ai_model.py
│   │   ├── continuous_learning.py
│   │   ├── supervisor_agent.py
│   │   └── __init__.py
│   │
│   ├── bots/
│   │   ├── base_swing_bot.py
│   │   ├── strategy_1w.py
│   │   ├── strategy_2w.py
│   │   ├── strategy_4w.py
│   │   ├── runner_1w.py
│   │   ├── runner_2w.py
│   │   ├── runner_4w.py
│   │   └── __init__.py
│   │
│   ├── cache/
│   │   ├── dashboard/
│   │   └── ml/
│   │
│   ├── analytics/
│   │   └── (EOD analytics / PnL)
│   │
│   ├── backend_service.py
│   ├── scheduler_config.py
│   ├── run_backend.py
│   └── __init__.py
│
├── dt_backend/
│   ├── core/
│   │   ├── config_dt.py
│   │   ├── context_state_dt.py
│   │   ├── regime_detector_dt.py
│   │   ├── policy_engine_dt.py
│   │   └── __init__.py
│   │
│   ├── engines/
│   │   ├── indicators.py
│   │   ├── feature_engineering.py
│   │   ├── execution_dt.py
│   │   ├── trade_executor.py
│   │   ├── broker_api.py
│   │   ├── historical_replay_engine.py
│   │   ├── historical_replay_manager.py
│   │   ├── replay_harness.py
│   │   ├── sequence_builder.py
│   │   └── __init__.py
│   │
│   ├── ml/
│   │   ├── ml_data_builder_intraday.py
│   │   ├── train_lightgbm_intraday.py
│   │   ├── train_lstm_intraday.py
│   │   ├── train_transformer_intraday.py
│   │   ├── intraday_hybrid_ensemble.py
│   │   ├── drift_monitor_dt.py
│   │   ├── continuous_learning_intraday.py
│   │   └── __init__.py
│   │
│   ├── jobs/
│   │   ├── daytrading_job.py
│   │   ├── rank_fetch_scheduler.py
│   │   └── __init__.py
│   │
│   ├── models/
│   │   ├── lightgbm_intraday/
│   │   │   ├── model.txt
│   │   │   ├── feature_map.json
│   │   │   ├── label_map.json
│   │   │   └── __init__.py
│   │   └── __init__.py
│   │
│   ├── rolling/
│   │   ├── intraday/rolling_intraday.json.gz
│   │   ├── longterm/
│   │   └── __init__.py
│   │
│   ├── signals/
│   │   ├── intraday/
│   │   │   ├── predictions/
│   │   │   ├── boards/
│   │   │   ├── ranks/
│   │   │   └── __init__.py
│   │   ├── longterm/
│   │   │   └── __init__.py
│   │   └── __init__.py
│   │
│   ├── universe/
│   │   ├── exchanges.json
│   │   ├── symbol_universe.json
│   │   └── __init__.py
│   │
│   ├── bars/
│   │   ├── intraday/
│   │   ├── daily/
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── analytics/
│   ├── performance/
│   ├── pnl/
│   │   ├── bots/
│   │   └── strategies/
│   └── __init__.py
│
├── utils/
│   ├── progress_bar.py
│   ├── logger.py
│   ├── json_tools.py
│   ├── time_utils.py
│   └── __init__.py
│
├── frontend/
│   (Next.js app: dashboard, live, bots, system, models, etc.)
│
├── .env
├── requirements.txt
├── README.md
└── feature_schema.json
=======
# Aion Analytics

Aion Analytics is a full-stack, multi-layer trading research and execution platform.

It runs **nightly EOD intelligence** and **intraday day-trading**, with:

- Human-like market context & regime awareness
- Multi-model ML stack (LightGBM + future LSTM/Transformer)
- Policy & execution layers (EOD bots + intraday bot)
- Continuous learning and drift monitoring
- Frontend dashboard (Next.js) for live, insights, bots, and system status

Core pipeline:

> **Context → Features → Models → Ensemble → Policy → Execution → Learning**

---

## 1. High-Level Architecture

The repo is structured around two main Python backends plus a frontend:

- `backend/` – **Nightly + swing / EOD brain**
- `dt_backend/` – **Intraday + day-trading brain**
- `frontend/` – Next.js UI (dashboard, insights, live, bots, system)
- `utils/` – Shared utilities (logging, progress bars, JSON, time utils)

They share a common storage layout under `data/`, `ml_data/`, `ml_data_dt/`, `analytics/`, and `logs/`.

### 1.1 Nightly vs Intraday

- **Nightly (backend)**:
  - Runs once per day (or on schedule) after market close / pre-market
  - Refreshes fundamentals, macro, sentiment
  - Builds EOD features and datasets
  - Trains / evaluates models
  - Produces nightly predictions & insights
  - Drives swing bots (1w, 2w, 4w)

- **Intraday (dt_backend)**:
  - Runs during market hours
  - Pulls 1m/5m bars (Alpaca, with fallbacks)
  - Maintains intraday rolling state
  - Computes technical + contextual intraday features
  - Runs intraday ML models (LightGBM today, LSTM/Transformer later)
  - Executes intraday trades against broker (paper/live)
  - Logs PnL and bot actions

---

## 2. Project Layout

From `Aion_Analytics_Folder_Map.txt`:

```text
Aion_Analytics/
│
├── backend/
│   ├── routers/
│   │   ├── live_prices_router.py
│   │   ├── insights_router.py
│   │   ├── system_router.py
│   │   └── __init__.py
│   │
│   ├── services/
│   │   ├── news_fetcher.py
│   │   ├── news_fetcher_loop.py
│   │   ├── fundamentals_fetcher.py
│   │   ├── macro_fetcher.py
│   │   ├── insights_builder.py
│   │   ├── prediction_logger.py
│   │   ├── social_sentiment_fetcher.py
│   │   ├── backfill_history.py
│   │   ├── scheduler_runner.py
│   │   ├── nightly_job.py
│   │   └── __init__.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── data_pipeline.py
│   │   ├── context_state.py
│   │   ├── regime_detector.py
│   │   ├── policy_engine.py
│   │   ├── ai_model.py
│   │   ├── continuous_learning.py
│   │   ├── supervisor_agent.py
│   │   └── __init__.py
│   │
│   ├── bots/
│   │   ├── base_swing_bot.py
│   │   ├── strategy_1w.py
│   │   ├── strategy_2w.py
│   │   ├── strategy_4w.py
│   │   ├── runner_1w.py
│   │   ├── runner_2w.py
│   │   ├── runner_4w.py
│   │   └── __init__.py
│   │
│   ├── cache/
│   │   ├── dashboard/
│   │   └── ml/
│   │
│   ├── analytics/
│   │   └── (EOD analytics / PnL)
│   │
│   ├── backend_service.py
│   ├── scheduler_config.py
│   ├── run_backend.py
│   └── __init__.py
│
├── dt_backend/
│   ├── core/
│   │   ├── config_dt.py
│   │   ├── context_state_dt.py
│   │   ├── regime_detector_dt.py
│   │   ├── policy_engine_dt.py
│   │   └── __init__.py
│   │
│   ├── engines/
│   │   ├── indicators.py
│   │   ├── feature_engineering.py
│   │   ├── execution_dt.py
│   │   ├── trade_executor.py
│   │   ├── broker_api.py
│   │   ├── historical_replay_engine.py
│   │   ├── historical_replay_manager.py
│   │   ├── replay_harness.py
│   │   ├── sequence_builder.py
│   │   └── __init__.py
│   │
│   ├── ml/
│   │   ├── ml_data_builder_intraday.py
│   │   ├── train_lightgbm_intraday.py
│   │   ├── train_lstm_intraday.py
│   │   ├── train_transformer_intraday.py
│   │   ├── intraday_hybrid_ensemble.py
│   │   ├── drift_monitor_dt.py
│   │   ├── continuous_learning_intraday.py
│   │   └── __init__.py
│   │
│   ├── jobs/
│   │   ├── daytrading_job.py
│   │   ├── rank_fetch_scheduler.py
│   │   └── __init__.py
│   │
│   ├── models/
│   │   ├── lightgbm_intraday/
│   │   │   ├── model.txt
│   │   │   ├── feature_map.json
│   │   │   ├── label_map.json
│   │   │   └── __init__.py
│   │   └── __init__.py
│   │
│   ├── rolling/
│   │   ├── intraday/rolling_intraday.json.gz
│   │   ├── longterm/
│   │   └── __init__.py
│   │
│   ├── signals/
│   │   ├── intraday/
│   │   │   ├── predictions/
│   │   │   ├── boards/
│   │   │   ├── ranks/
│   │   │   └── __init__.py
│   │   ├── longterm/
│   │   │   └── __init__.py
│   │   └── __init__.py
│   │
│   ├── universe/
│   │   ├── exchanges.json
│   │   ├── symbol_universe.json
│   │   └── __init__.py
│   │
│   ├── bars/
│   │   ├── intraday/
│   │   ├── daily/
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── analytics/
│   ├── performance/
│   ├── pnl/
│   │   ├── bots/
│   │   └── strategies/
│   └── __init__.py
│
├── utils/
│   ├── progress_bar.py
│   ├── logger.py
│   ├── json_tools.py
│   ├── time_utils.py
│   └── __init__.py
│
├── frontend/
│   (Next.js app: dashboard, live, bots, system, models, etc.)
│
├── .env
├── requirements.txt
├── README.md
└── feature_schema.json
>>>>>>> 972c7287330de9c0fc81e4a310eb974fad5114b9
