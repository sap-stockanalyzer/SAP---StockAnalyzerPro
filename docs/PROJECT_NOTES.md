# Aion Analytics — Debugging & Upgrade Plan

## Conversation Snapshot
- **Source:** ChatGPT ➜ Codex handoff
- **Request:** "Continue the Aion Analytics project. Load my GitHub repo and include this conversation. Project ID: stockanalyzerpro/Aion_Analytics. Move full debugging + upgrade plan from ChatGPT thread into Codex workspace."
- **Goal:** Preserve the chat context and host an actionable plan for future contributors directly within this repository.
- **Branching status (2024-05-16):** The workspace currently exposes a single local branch named `work`. The GitHub repository has now been wired up as the `origin` remote, but no other branches have been checked out locally, so all commits (including this note) still live on `work` until we push upstream.

## GitHub sync status
- **Remote configured (2024-05-16):** `origin` now points to `git@github.com:stockanalyzerpro/Aion_Analytics.git`, so the workspace is ready to push once you're comfortable publishing the commits.
- **Why GitHub stays untouched:** even with the remote configured, nothing has been pushed yet, so the GitHub repo remains unchanged until we run `git push`.
- **How to publish these commits to GitHub:**
  1. (Done) `git remote add origin git@github.com:stockanalyzerpro/Aion_Analytics.git`.
  2. `git fetch origin` if you want to inspect remote state before pushing.
  3. `git push -u origin work` to create/update the branch upstream. After the first push, `git push`/`git pull` will track GitHub normally.
- **If you need a `main` branch:** once the remote exists you can either rename `work` to `main` (`git branch -m work main`) before pushing, or push `work` and open a PR into your remote `main` via GitHub.

## Remote sync attempt (2024-05-18)
- **Goal:** refresh the workspace with any upstream changes by running `git fetch origin`.
- **Result:** SSH access to GitHub is currently blocked from this environment (`ssh: connect to host github.com port 22: Network is unreachable`), so no additional files could be retrieved.
- **Next step:** once network egress is available (or HTTPS traffic is permitted), rerun `git fetch origin` followed by `git status` to reconcile local history with GitHub before pushing.

## Backend smoke test report (2024-05-18)
- **Command:** `python -m compileall backend dt_backend`.
- **Status:** ✅ All backend and day-trading modules compiled successfully, confirming there are no syntax or import-time errors in the Python tree.
- **Recommendations:**
  1. **Keep compile checks in CI.** The `compileall` pass caught encoding issues earlier this week; leaving it in CI provides a cheap regression guard while deeper unit tests are still under construction.
  2. **Add targeted unit tests.** Prioritize lightweight tests around `backend_extension.run_latest_drift`, `dt_backend/trading_bot_simulator`, and other modules that previously failed to compile so future logic changes are validated automatically.
  3. **Instrument schedulers.** Now that the modules load cleanly, wire telemetry into `backend/scheduler_runner.py` and `dt_backend/rank_fetch_scheduler.py` so production runs emit traces/metrics (ties back to the Debugging Plan above).

> _Note:_ Because GitHub could not be reached, this workspace still reflects only the files that have been edited locally in Codex. No additional upstream assets were downloaded during this pass.

## Debugging Plan
1. **Frontend regressions (Next.js `/app` directory).**
   - Instrument key surfaces such as `app/page.tsx`, `app/optimizer/*`, and `app/reports/*` with feature flags and runtime guards to surface hydration or data-fetch failures early.
   - Add Jest/Playwright coverage for `components/SearchBar`, `components/TopPredictions`, and any async hooks in `hooks/` to reproduce edge cases before they hit production.
2. **API gateway + backend service health.**
   - Extend `backend/backend_service.py` and `backend/system_status_router.py` with structured logging (JSON) and correlation IDs so that each request path can be traced across routers (`live_prices_router.py`, `dashboard_router.py`, etc.).
   - Implement smoke tests within `backend/tests_backend.py` that spin up lightweight stubs for `backend/backend_extension.py` to validate serialization/validation logic from `backend/schemas.py`.
3. **Scheduler + nightly jobs.**
   - Enable verbose tracing in `backend/scheduler_runner.py`, `backend/nightly_job.py`, and `backend/scheduler_config.py` to capture job start/end, resource usage, and downstream task IDs.
   - Build failure hooks in `backend/rolling_integrity_check.py` and `backend/continuous_learning.py` to push alerts whenever data freshness or model drift metrics breach thresholds.
4. **Data ingestion + feature store.**
   - Validate each ingestion module (`backend/ticker_fetcher.py`, `backend/fundamentals_fetcher.py`, `backend/news_fetcher*.py`, `backend/social_sentiment_fetcher.py`) with checksum comparisons before persisting to `ml_data/`.
   - Expand `backend/verify_dataset_features.py` and `backend/ml_data_builder.py` to ensure schema changes are versioned and diffed prior to training runs.
5. **Model training & inference.**
   - For classical models (`backend/train_lightgbm.py`, `backend/ai_model.py`) and graph/gnn components (`backend/gnn_model.py`), capture experiment metadata (git SHA, hyper-params, dataset hashes) and write them to `backend/ml_helpers.py` utilities for reproducibility.
   - Wrap inference endpoints (e.g., `backend/insights_router.py`, `backend/optimizer` stack) with circuit breakers to degrade gracefully if `vector_index.py` or `summarizer.py` dependencies fail.
6. **Operational tooling.**
   - Keep `backend/wiring_check.py`, `backend/verify_cache_integrity.py`, and `backend/manual_recompute_dashboard.py` runnable from CI to quickly replicate any reported dashboard inconsistency.
   - Integrate `backend/dashboard_monitor.py` with the alerting stack so UI discrepancies bubble up automatically.

## Upgrade Plan
1. **Modernize the frontend workflow.**
   - Adopt the Next.js App Router data layer (React Server Components) to replace ad-hoc fetches in `app/insights`, `app/optimizer`, and `app/system` so pages stream progressively.
   - Introduce design tokens in `app/globals.css` and convert shared layout primitives in `components/` to TypeScript-driven variants for better theme control.
2. **API consolidation + performance.**
   - Refactor routers (`backend/dashboard_router.py`, `backend/insights_router.py`, `backend/live_prices_router.py`) into a FastAPI sub-application with shared dependencies for caching/auth.
   - Add async IO (httpx, aiokafka, etc.) to ingestion routines in `backend/data_pipeline.py` and `backend/news_fetcher_loop.py` to improve throughput.
3. **Observability + reliability.**
   - Embed OpenTelemetry tracing in `backend/backend_service.py`, job scripts (`backend/trading_bot_nightly_*`), and the policy layer (`backend/policy_engine.py`) for unified metrics + traces.
   - Add health dashboards using `backend/dashboard_builder.py` to surface lag, queue depth, and model accuracy KPIs in near real-time.
4. **ML experimentation + deployment.**
   - Containerize training scripts (`backend/train_lightgbm.py`, `backend/gnn_model.py`, `backend/rl_env.py`) with reproducible conda/poetry lockfiles and push build artifacts to a model registry.
   - Integrate continuous learning flows (`backend/continuous_learning.py`, `backend/online_trainer.py`) with feature stores and automated rollback logic if `backend/regime_detector.py` flags instability.
5. **Security + compliance.**
   - Expand `backend/policy_engine.py` to enforce per-strategy constraints and runbook steps before trades are executed through `backend/trading_bot_nightly_*` scripts.
   - Audit secrets/configuration via `backend/config.py` and move sensitive values into environment-specific vault integrations before productionizing the scheduler stack.
6. **Documentation + onboarding.**
   - Keep this `docs/PROJECT_NOTES.md` file updated with each major milestone and link to architectural diagrams.
   - Add module-level READMEs (e.g., `backend/README.md`, `app/README.md`) summarizing entry points, commands, and troubleshooting steps to shorten ramp-up time.
