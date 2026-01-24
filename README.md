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

## Quick Start (5-minute setup)

```bash
# 1. Clone the repository
git clone https://github.com/stockanalyzerpro/Aion_Analytics.git
cd Aion_Analytics

# 2. Install Python dependencies (Python 3.11+ recommended)
pip install -r requirements.txt

# 3. Copy environment template and configure
cp .env.example .env
# Edit .env with your API keys (Alpaca, Polygon, Alpha Vantage, etc.)

# 4. Create required directories
mkdir -p da_brains/intraday ml_data ml_data_dt data logs

# 5. Run the platform (supervisor launches all services)
python run_backend.py
```

The platform will start:
- **Backend API** (port 8000) - Unified backend for all endpoints
- **DT Backend API** (port 8010) - Intraday trading
- **DT Scheduler** - Live market data loop
- **Replay Service** (port 8020) - Historical replay

Access the dashboard at `http://localhost:8000`

**Note:** Port consolidation - Backend now uses unified port 8000 for all endpoints. The previous dual-port setup (8000/8001) has been deprecated to avoid confusion and routing issues.

---

## System Requirements

### Hardware
- **Minimum**: 8GB RAM, 4 CPU cores, 50GB storage
- **Recommended**: 16GB RAM, 8 CPU cores, 200GB SSD

### Software
- **Python**: 3.11 or 3.12 (3.12.3 tested)
- **OS**: Linux (Ubuntu 20.04+), macOS 12+, or Windows with WSL2
- **Database**: File-based (JSON, Parquet) - no external DB required

### External Services (Required)
- **Alpaca Markets** - Paper/live trading account (free paper trading)
- **Polygon.io** - Market data (optional, Alpaca data used as fallback)
- **Alpha Vantage** - Fundamentals and macro data (optional)

---

## Installation

### 1. Python Environment Setup

```bash
# Create virtual environment (recommended)
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
```

Required variables:
- `ALPACA_API_KEY_ID` - Your Alpaca API key
- `ALPACA_API_SECRET_KEY` - Your Alpaca secret key
- `ALPACA_PAPER_BASE_URL` - `https://paper-api.alpaca.markets` (paper trading)

Optional but recommended:
- `POLYGON_API_KEY` - For enhanced market data
- `ALPHA_VANTAGE_API_KEY` - For fundamentals
- `FINNHUB_API_KEY` - For news and sentiment

### 3. Storage Setup

Create required data directories:

```bash
mkdir -p da_brains/intraday
mkdir -p da_brains/positions
mkdir -p ml_data/models
mkdir -p ml_data_dt/models
mkdir -p data/{bars,fundamentals,macro}
mkdir -p logs
mkdir -p analytics/{performance,pnl}
```

### 4. Initial Data Bootstrap (Optional)

```bash
# Fetch historical data for backtesting (one-time setup)
python -m backend.services.backfill_history_sip --symbols SPY,QQQ,AAPL --days 365

# Build initial ML dataset (optional)
python -m backend.ml.ml_data_builder --rebuild
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      AION ANALYTICS PLATFORM                     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐         ┌──────────────────────┐
│   NIGHTLY BRAIN      │         │   INTRADAY BRAIN     │
│   (backend/)         │         │   (dt_backend/)      │
│                      │         │                      │
│  - EOD Data Fetch    │         │  - Live 1m/5m Bars  │
│  - Fundamentals      │         │  - Real-time Regime │
│  - Macro Context     │         │  - ML Predictions   │
│  - Sentiment         │         │  - Position Mgmt    │
│  - ML Training       │         │  - Order Execution  │
│  - Swing Signals     │         │  - Risk Rails       │
│  (1w, 2w, 4w bots)   │         │  (DT strategy)      │
└──────────────────────┘         └──────────────────────┘
         │                                  │
         └──────────────┬───────────────────┘
                        │
         ┌──────────────▼──────────────┐
         │    Shared Storage Layer     │
         │   (data/, ml_data/, logs/)  │
         └─────────────────────────────┘
                        │
         ┌──────────────▼──────────────┐
         │   External Data Sources     │
         │   (Alpaca, Polygon, etc.)   │
         └─────────────────────────────┘
```

### Component Interaction

1. **Context Layer**: Ingests market data, fundamentals, macro indicators, sentiment
2. **Feature Engineering**: Transforms raw data into ML-ready features
3. **Model Layer**: LightGBM predictions (with LSTM/Transformer planned)
4. **Regime Detection**: Classifies market conditions (bull/bear/range/volatile)
5. **Policy Engine**: Decision logic (entry/exit/sizing based on regime + models)
6. **Execution Layer**: Order management and broker interaction
7. **Continuous Learning**: Drift detection and model retraining

---

## High-Level Architecture

The repo is structured around two main Python backends plus a frontend:

- `backend/` – **Nightly + swing / EOD brain**
- `dt_backend/` – **Intraday + day-trading brain**
- `frontend/` – Next.js UI (dashboard, insights, live, bots, system)
- `utils/` – Shared utilities (logging, progress bars, JSON, time utils)

They share a common storage layout under `data/`, `ml_data/`, `ml_data_dt/`, `analytics/`, and `logs/`.

### Nightly vs Intraday

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

## Project Layout

```text
Aion_Analytics/
│
├── backend/               # Nightly + swing EOD brain
│   ├── routers/          # FastAPI endpoints
│   ├── services/         # Data fetchers (news, fundamentals, macro)
│   ├── core/             # ML pipeline, regime detection, policy
│   ├── bots/             # Swing strategies (1w, 2w, 4w)
│   ├── jobs/             # Nightly job scheduler
│   └── admin/            # Authentication & admin tools
│
├── dt_backend/            # Intraday + day-trading brain
│   ├── core/             # Intraday context, regime, policy
│   ├── engines/          # Feature engineering, execution
│   ├── ml/               # Intraday ML models
│   ├── jobs/             # DT scheduler, live data loop
│   ├── services/         # Position manager, truth store
│   ├── risk/             # Risk rails, emergency stop
│   ├── strategies/       # Intraday strategies
│   └── api/              # FastAPI app
│
├── utils/                 # Shared utilities
│   ├── logger.py
│   ├── json_tools.py
│   └── time_utils.py
│
├── data/                  # Market data storage
│   ├── bars/             # OHLCV data
│   ├── fundamentals/     # Company fundamentals
│   └── macro/            # Economic indicators
│
├── ml_data/              # Nightly ML artifacts
│   ├── models/           # Trained models
│   └── datasets/         # Feature datasets
│
├── ml_data_dt/           # Intraday ML artifacts
│   ├── models/           # Intraday models
│   └── rolling/          # Rolling state cache
│
├── da_brains/            # Truth store (trades, positions, metrics)
│   ├── intraday/         # DT trades log
│   └── positions/        # Position registry
│
├── logs/                 # Application logs
│
├── monitoring/           # Monitoring configs (Prometheus, Grafana)
│
├── tests/                # Unit and integration tests
│
├── .env                  # Environment variables (secrets)
├── .env.example          # Environment template
├── knobs.env             # Backend configuration
├── dt_knobs.env          # Intraday configuration
├── requirements.txt      # Python dependencies
├── requirements-dev.txt  # Development dependencies
├── run_backend.py        # Main entry point (supervisor)
└── README.md             # This file
```

---

## API Documentation

### Backend API (Port 8000)

**Unified Backend** - All endpoints now run on port 8000 (previously split across 8000/8001).

#### System Status
- `GET /health` - System health check
- `GET /health/ready` - Readiness probe (Kubernetes)
- `GET /health/live` - Liveness probe
- `GET /metrics` - Prometheus metrics

#### Dashboard
- `GET /api/dashboard` - Dashboard data
- `GET /api/insights` - Nightly insights
- `GET /api/live-prices` - Real-time market prices

#### Bots
- `GET /api/bots/page` - Unified bots page bundle (swing + intraday)
- `GET /api/bots/overview` - Alias for /api/bots/page
- `GET /api/eod/status` - EOD (swing) bots status with positions & PnL
- `GET /api/eod/configs` - EOD bot configurations
- `GET /api/eod/configs/{bot_key}` - Single bot config
- `PUT /api/eod/configs/{bot_key}` - Update bot config
- `GET /api/eod/logs/days` - List of trading days with logs
- `GET /api/eod/logs/last-day` - Latest day's logs
- `GET /api/eod/logs/{day}` - Logs for specific day
- `GET /api/bots-hub` - Bot management hub

#### Portfolio
- `GET /api/portfolio/holdings/top/{horizon}` - Top holdings by PnL
  - `horizon`: `1w` (7 days) or `1m` (28 days)
  - Query param: `limit` (default: 3)
  - Returns: List of top holdings with current prices, entry prices, PnL, days held

#### Events (Server-Sent Events / SSE)
- `GET /api/events/bots` - Real-time bots page updates (5s interval)
- `GET /api/events/admin/logs` - Live admin logs stream (2s interval)
- `GET /api/events/intraday` - Intraday snapshot updates (5s interval)

#### Models
- `GET /api/models` - Model metrics
- `GET /api/metrics` - System metrics

**Example Responses:**

Portfolio holdings:
```json
GET /api/portfolio/holdings/top/1w?limit=3
{
  "horizon": "1w",
  "count": 3,
  "holdings": [
    {
      "ticker": "AAPL",
      "current_price": 185.50,
      "avg_entry_price": 175.00,
      "pnl_dollars": 105.00,
      "pnl_percent": 6.0,
      "quantity": 10,
      "days_held": 8,
      "entry_date": "2026-01-08"
    }
  ]
}
```

EOD bot status:
```json
GET /api/eod/status
{
  "running": true,
  "last_update": "2026-01-19T03:00:00Z",
  "bots": {
    "eod_1w": {
      "enabled": true,
      "cash": 5000.0,
      "invested": 10000.0,
      "allocated": 2500.0,
      "holdings_count": 4,
      "equity": 15000.0,
      "equity_curve": [
        {"t": "2026-01-01", "value": 14500.0},
        {"t": "2026-01-19", "value": 15000.0}
      ],
      "positions": [...]
    }
  }
}
```

### DT Backend API (Port 8010)

#### Health
- `GET /health` - Intraday system health
- `GET /health/ready` - Readiness probe
- `GET /health/live` - Liveness probe
- `GET /metrics` - Prometheus metrics

#### Jobs
- `POST /jobs/cycle` - Trigger trading cycle
- `GET /jobs/status` - Job status

#### Emergency Controls
- `POST /emergency/stop` - Emergency stop (kill switch)
- `POST /emergency/resume` - Resume trading
- `GET /emergency/status` - Emergency stop status

#### Data
- `GET /data/rolling` - Rolling state cache
- `GET /data/positions` - Current positions
- `GET /data/metrics` - DT metrics

---

## Configuration

### Environment Variables

See `.env.example` for complete list. Key variables:

**Trading Controls:**
- `DT_DRY_RUN=1` - Paper trading mode (no real orders)
- `DT_ENABLE_LIVE_TRADING=0` - Enable live trading (0=disabled)
- `DT_EMERGENCY_STOP_FILE=/tmp/dt_emergency_stop` - Emergency stop file

**Risk Management:**
- `DT_MAX_POSITIONS=3` - Maximum concurrent positions
- `DT_DAILY_LOSS_LIMIT_USD=300` - Daily loss limit
- `DT_MAX_WEEKLY_DRAWDOWN_PCT=8.0` - Weekly drawdown cap
- `DT_MAX_MONTHLY_DRAWDOWN_PCT=15.0` - Monthly drawdown cap
- `DT_VIX_SPIKE_THRESHOLD=35.0` - VIX panic threshold

**Data Sources:**
- `DA_BRAINS_ROOT=/path/to/da_brains` - Truth store location
- `ML_DATA_ROOT=/path/to/ml_data` - ML artifacts
- `DT_ROLLING_PATH=da_brains/intraday/rolling_intraday.json.gz` - Rolling cache

### Configuration Knobs

Edit `dt_knobs.env` for intraday tuning:

```bash
# Execution
DT_EXEC_MIN_CONF=0.25          # Minimum confidence for entry
DT_MAX_ORDERS_PER_CYCLE=3      # Orders per cycle
DT_MIN_TRADE_GAP_MINUTES=15    # Minimum time between trades

# Universe
DT_CANDIDATE_UNIVERSE=SPY,QQQ,AAPL,MSFT,TSLA,NVDA
DT_UNIVERSE_SIZE=150           # Universe size
DT_UNIVERSE_ROTATION=1         # Enable rotation

# Risk
DT_MAX_LOSS_PER_SYMBOL_DAY=500  # Max loss per symbol per day
DT_ALLOW_LUNCH_TRADES=1         # Trade during lunch hour
```

### Configuration Constants

All trading constants are centralized in `dt_backend/core/constants_dt.py` for maintainability and consistency:

**Confidence Thresholds:**
- `CONFIDENCE_MIN` (0.45) - Minimum confidence for trade execution (raised from 0.25)
- `CONFIDENCE_MIN_PROBE` (0.18) - Micro-entry tier for low-confidence probes
- `CONFIDENCE_MAX` (0.99) - Hard cap after adjustments
- `CONFIDENCE_EXIT_BUFFER` (0.05) - Margin above HOLD for SELL signals

**Position Sizing:**
- `POSITION_MAX_FRACTION` (0.15) - Maximum 15% of account per symbol
- `POSITION_PROBE_FRACTION` (0.25) - 25% of full conviction for probes
- `POSITION_PRESS_MULT` (1.35) - Scale-up multiplier for high P(Hit)
- `POSITIONS_MAX_OPEN` (3) - Maximum concurrent positions

**Signal Stability & Hysteresis:**
- `EDGE_MIN_TO_FLIP` (0.06) - Minimum edge to flip direction
- `EDGE_HOLD_BIAS` (0.03) - Extra margin to flip from HOLD
- `CONFIRMATIONS_TO_FLIP` (2) - Consecutive confirmations needed
- `HOLD_MIN_TIME_MINUTES` (10) - Minimum hold before exit allowed
- `COOLDOWN_AFTER_BUY_MINUTES` (10) - Flip cooldown after entry

**Risk Management:**
- `DAILY_LOSS_LIMIT` (300.0) - Daily stop loss in USD
- `WEEKLY_DRAWDOWN_MAX_PCT` (8.0) - Weekly drawdown cap
- `MONTHLY_DRAWDOWN_MAX_PCT` (15.0) - Monthly drawdown cap
- `VIX_SPIKE_THRESHOLD` (35.0) - VIX panic threshold
- `EXPOSURE_MAX` (0.55) - Maximum portfolio exposure

**Regime Exposure Mapping:**
- `bull`: 1.00 (Full exposure)
- `chop`: 0.70 (70% of normal)
- `bear`: 0.45 (45% of normal)
- `panic`: 0.20 (20% - defensive mode)
- `stress`: 0.10 (10% - near shutdown)

To modify a threshold, edit `dt_backend/core/constants_dt.py` and restart the platform.

### Auto-Retraining Triggers

The platform automatically triggers model retraining when performance degrades or sufficient time has passed. See `dt_backend/ml/auto_retrain_trigger.py` for implementation.

**Automatic Retraining Conditions:**

Models automatically retrain when ANY of these conditions are met:

1. **Win Rate Degradation**: Win rate < 45% (`WIN_RATE_RETRAINING_THRESHOLD`)
2. **Sharpe Ratio Decline**: Sharpe ratio < 0.5 (`SHARPE_RETRAINING_THRESHOLD`)
3. **Feature Drift Detection**: Feature drift > 15% (`FEATURE_DRIFT_RETRAINING_THRESHOLD`)
4. **Scheduled Refresh**: 7+ days since last retrain (`MAX_DAYS_WITHOUT_RETRAIN`)

**Usage Example:**

```python
from dt_backend.ml.auto_retrain_trigger import AutoRetrainTrigger

trigger = AutoRetrainTrigger()

# Check metrics
should_retrain, reason = trigger.check_and_trigger({
    "win_rate": 0.42,           # Below 0.45 threshold
    "sharpe_ratio": 0.45,       # Below 0.5 threshold
    "feature_drift": 0.12,      # Within acceptable range
})

if should_retrain:
    print(f"🔄 Triggering retrain: {reason}")
    # Schedule retraining
    trigger.record_retrain()
```

**Integration:**

The auto-retrain trigger is integrated into `dt_backend/ml/continuous_learning_intraday.py` and runs automatically during the continuous learning cycle.



---

## Production Deployment

### Option 1: Systemd Service (Linux)

Create `/etc/systemd/system/aion-analytics.service`:

```ini
[Unit]
Description=AION Analytics Trading Platform
After=network.target

[Service]
Type=simple
User=aion
WorkingDirectory=/home/aion/Aion_Analytics
Environment="PATH=/home/aion/Aion_Analytics/venv/bin"
ExecStart=/home/aion/Aion_Analytics/venv/bin/python run_backend.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable aion-analytics
sudo systemctl start aion-analytics
sudo systemctl status aion-analytics
```

### Option 2: Docker Compose

See `DEPLOYMENT.md` for complete Docker setup.

### Option 3: Kubernetes

Health endpoints support Kubernetes probes:

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8010
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8010
  initialDelaySeconds: 10
  periodSeconds: 5
```

---

## Monitoring

### Prometheus Metrics

Metrics endpoint: `http://localhost:8010/metrics`

Key metrics:
- `dt_trades_total` - Total trades by side and bot
- `dt_cycle_duration_seconds` - Cycle execution time
- `dt_open_positions` - Current open positions
- `dt_equity_dollars` - Account equity
- `dt_daily_pnl_dollars` - Daily P&L

### Grafana Dashboard

Import `monitoring/grafana-dashboard.json` for pre-built dashboard with:
- System health status
- Trade volume by bot
- Daily P&L chart
- Open positions gauge
- Drawdown percentage
- Cycle execution time
- Error rate

### Log Files

Logs are written to:
- `logs/nightly_YYYYMMDD.log` - Nightly job logs
- `logs/dt_backend_YYYYMMDD.log` - Intraday logs
- `da_brains/intraday/dt_trades.jsonl` - Trade event log (append-only)

---

## Troubleshooting Guide

### Common Issues

**1. "No module named 'backend'" or "No module named 'dt_backend'"**

Ensure Python path is set correctly:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Aion_Analytics"
```

**2. "Connection refused" errors**

Check if services are running:
```bash
curl http://localhost:8000/health  # Backend
curl http://localhost:8010/health  # DT Backend
```

**3. "API key not found" errors**

Verify `.env` file exists and contains all required API keys:
```bash
grep ALPACA_API_KEY .env
```

**4. "Permission denied" on emergency stop file**

Ensure write permissions:
```bash
mkdir -p /tmp
touch /tmp/dt_emergency_stop
chmod 666 /tmp/dt_emergency_stop
```

**5. High memory usage**

Reduce universe size in `dt_knobs.env`:
```bash
DT_UNIVERSE_SIZE=50  # Down from 150
```

**6. Trading halted (stand_down=True)**

Check risk rails logs:
```bash
grep "stand_down" logs/dt_backend_*.log
```

Common causes:
- Daily loss limit hit → Wait until next day or adjust `DT_DAILY_LOSS_LIMIT_USD`
- VIX spike → Market volatility high, wait for VIX < threshold
- Weekly/monthly drawdown → Wait for time period reset

**7. No trades being placed**

Check configuration:
```bash
# Ensure dry run is disabled for live trading
grep DT_DRY_RUN .env  # Should be 0 for live
grep DT_ENABLE_LIVE_TRADING .env  # Should be 1

# Check emergency stop
curl http://localhost:8010/emergency/status
```

---

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_risk_rails_dt.py

# Run with coverage
pytest tests/ --cov=dt_backend --cov=backend --cov-report=html
```

### Code Style

The project uses:
- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **Type hints** encouraged

```bash
# Format code
black backend/ dt_backend/

# Lint
ruff check backend/ dt_backend/
```

### Contributing

See `CONTRIBUTING.md` for development setup and contribution guidelines.

---

## Security

### Secrets Management

**Never commit secrets to git!**

- Use `.env` for secrets (already in `.gitignore`)
- Use `.env.example` as template (no real keys)
- Rotate API keys regularly
- Use paper trading for development (`DT_DRY_RUN=1`)

### Emergency Procedures

**Emergency Stop:**
```bash
# Immediate stop (all trading halts)
curl -X POST http://localhost:8010/emergency/stop?reason="manual_intervention"

# Resume trading
curl -X POST http://localhost:8010/emergency/resume
```

**Kill All Processes:**
```bash
pkill -f run_backend.py
pkill -f dt_scheduler
```

---

## Performance

### Benchmarks (Intel i7-10700, 16GB RAM)

- **Nightly job**: 15-30 minutes (500 symbols)
- **DT cycle time**: 2-8 seconds (150 symbol universe)
- **Model inference**: <100ms per symbol
- **Order latency**: 50-200ms (Alpaca API)

### Optimization Tips

1. **Reduce universe size** - Fewer symbols = faster cycles
2. **Enable symbol rotation** - Prevents overtrading same symbols
3. **Use SSD storage** - Faster rolling cache reads
4. **Scale UI workers** - Separate read endpoints from scheduler

---

## License

Proprietary - All rights reserved

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/stockanalyzerpro/Aion_Analytics/issues
- Documentation: See `ARCHITECTURE.md`, `DEPLOYMENT.md`, `CONTRIBUTING.md`

---

## Changelog

### v3.3 (Current)
- Fast/slow lane orchestration
- Candidate universe support
- Weekly/monthly drawdown caps
- VIX spike protection
- Emergency stop mechanism
- Prometheus metrics
- Health check endpoints

### v3.2
- Multi-armed bandit integration
- A/B testing framework
- Improved position management

### v3.1
- LightGBM intraday models
- Rolling cache optimization
- Risk rails v1.1 (broker equity source)

### v3.0
- Initial dual-brain architecture
- Nightly + intraday separation
- FastAPI migration

---

**Built with ❤️ by the AION team**
