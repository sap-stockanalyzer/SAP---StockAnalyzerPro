# AION Analytics Architecture

System architecture and design documentation for AION Analytics trading platform.

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    AION ANALYTICS PLATFORM                      │
│                                                                  │
│  ┌──────────────────────┐       ┌─────────────────────────┐   │
│  │   NIGHTLY BRAIN      │       │   INTRADAY BRAIN        │   │
│  │   (backend/)         │       │   (dt_backend/)         │   │
│  │                      │       │                         │   │
│  │  • EOD Data Fetch    │◄─────►│  • Live 1m/5m Bars     │   │
│  │  • Fundamentals      │       │  • Real-time Regime    │   │
│  │  • Macro Context     │       │  • ML Predictions      │   │
│  │  • Sentiment         │       │  • Position Mgmt       │   │
│  │  • ML Training       │       │  • Order Execution     │   │
│  │  • Swing Signals     │       │  • Risk Rails          │   │
│  │  (1w, 2w, 4w bots)   │       │  (DT strategy)         │   │
│  └──────────────────────┘       └─────────────────────────┘   │
│           │                               │                     │
│           └───────────────┬───────────────┘                     │
│                           │                                     │
│           ┌───────────────▼──────────────┐                     │
│           │   Shared Storage Layer       │                     │
│           │  (data/, ml_data/, logs/)    │                     │
│           └──────────────────────────────┘                     │
└────────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────▼──────────────┐
           │   External Data Sources      │
           │  (Alpaca, Polygon, APIs)     │
           └──────────────────────────────┘
```

## Component Architecture

### 1. Dual-Brain Design

AION uses a **dual-brain architecture** with separate systems for different trading timeframes:

#### Nightly Brain (backend/)
- **Purpose**: EOD intelligence and swing trading
- **Frequency**: Once per day after market close
- **Responsibilities**:
  - Data aggregation (fundamentals, macro, sentiment)
  - Feature engineering for EOD data
  - ML model training (LightGBM)
  - Swing bot signals (1-week, 2-week, 4-week)
  - Insights generation

#### Intraday Brain (dt_backend/)
- **Purpose**: Real-time day trading
- **Frequency**: Continuous during market hours
- **Responsibilities**:
  - Live 1m/5m bar fetching
  - Intraday feature engineering
  - ML model inference
  - Real-time regime detection
  - Order execution
  - Risk management

### 2. Core Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌─────────┐
│   Context   │────►│   Features   │────►│ Models  │
│  (Market)   │     │ (Technical)  │     │  (ML)   │
└─────────────┘     └──────────────┘     └─────────┘
                                               │
                                               ▼
┌─────────────┐     ┌──────────────┐     ┌─────────┐
│  Learning   │◄────│  Execution   │◄────│ Policy  │
│ (Feedback)  │     │  (Trading)   │     │(Decide) │
└─────────────┘     └──────────────┘     └─────────┘
```

### 3. Technology Stack

**Backend:**
- **Framework**: FastAPI (Python 3.11+)
- **ML**: LightGBM, scikit-learn
- **Data**: Pandas, NumPy, Parquet
- **APIs**: Alpaca, Polygon, Alpha Vantage

**Infrastructure:**
- **Storage**: File-based (JSON, JSONL, Parquet)
- **Monitoring**: Prometheus, Grafana
- **Deployment**: Systemd, Docker, Kubernetes

## Data Flow

### Nightly Pipeline

```
1. Data Collection
   ↓
2. Context Building (fundamentals, macro, sentiment)
   ↓
3. Feature Engineering (EOD features)
   ↓
4. Model Training (if needed)
   ↓
5. Predictions (swing signals)
   ↓
6. Bot Execution (1w, 2w, 4w)
   ↓
7. Store Results (insights, trades)
```

### Intraday Pipeline

```
1. Bars Fetch (1m/5m from Alpaca)
   ↓
2. Context Building (intraday stats, VIX)
   ↓
3. Feature Engineering (technical indicators)
   ↓
4. ML Predictions (signal + confidence)
   ↓
5. Regime Detection (bull/bear/range/volatile)
   ↓
6. Policy Decision (side, size, risk)
   ↓
7. Risk Rails Check (drawdowns, VIX, limits)
   ↓
8. Order Execution (if not stand_down)
   ↓
9. Position Management (stops, targets, exits)
   ↓
10. Metrics & Logging
```

## Storage Layout

```
Aion_Analytics/
├── da_brains/              # Truth store
│   ├── intraday/
│   │   ├── dt_state.json          # Current state
│   │   ├── dt_trades.jsonl        # Trade log (append-only)
│   │   ├── dt_metrics.json        # Metrics snapshot
│   │   └── rolling_intraday.json.gz  # Rolling cache
│   └── positions/
│       └── position_registry.json  # Position ownership
│
├── ml_data/                # Nightly ML artifacts
│   └── models/
│       └── lightgbm_*.txt
│
├── ml_data_dt/             # Intraday ML artifacts
│   └── models/
│       └── lightgbm_intraday_*.txt
│
├── data/                   # Market data
│   ├── bars/
│   ├── fundamentals/
│   └── macro/
│
└── logs/                   # Application logs
    ├── nightly_YYYYMMDD.log
    └── dt_backend_YYYYMMDD.log
```

## Risk Management Architecture

### Multi-Layer Risk System

```
Layer 1: Configuration Validation
  └─► Validate knobs at startup
      └─► Ensure safe ranges

Layer 2: Pre-Trade Checks
  └─► Confidence threshold
  └─► Position limits
  └─► Per-symbol loss limits
  └─► Flip cooldown

Layer 3: Risk Rails
  └─► Daily loss limit
  └─► Daily drawdown cap
  └─► Weekly drawdown cap (8%)
  └─► Monthly drawdown cap (15%)
  └─► VIX spike protection
  └─► Max exposure
  └─► Cooldown after losses

Layer 4: Emergency Stop
  └─► File-based kill switch
  └─► Immediate halt
  └─► Manual/API triggered
```

### Risk State Machine

```
┌──────────┐
│  Normal  │
│  Trading │
└────┬─────┘
     │
     ├─► Daily Loss Limit Hit ─────────┐
     ├─► Weekly Drawdown > 8% ─────────┤
     ├─► Monthly Drawdown > 15% ───────┤
     ├─► VIX Spike (>35) ──────────────┤
     ├─► Emergency Stop File ──────────┤
     │                                  ▼
     │                          ┌──────────────┐
     │                          │  Stand Down  │
     │                          │  (No Trade)  │
     │                          └──────┬───────┘
     │                                  │
     │                                  │
     ◄──────── Conditions Clear ────────┘
```

## API Architecture

### Backend API (Ports 8000/8001)

```
/health                    # Health check
/health/ready             # Readiness probe
/health/live              # Liveness probe
/api/dashboard            # Dashboard data
/api/insights             # Nightly insights
/api/live-prices          # Real-time prices
/api/eod-bots             # EOD bot status
/api/models               # Model metrics
```

### DT Backend API (Port 8010)

```
/health                   # Health check with components
/health/ready            # Readiness probe
/health/live             # Liveness probe
/metrics                 # Prometheus metrics
/metrics/summary         # JSON metrics summary
/emergency/stop          # Trigger emergency stop
/emergency/resume        # Resume trading
/emergency/status        # Check emergency status
/emergency/test          # Test mechanism
/jobs/cycle              # Trigger trading cycle
/jobs/status             # Job status
/data/rolling            # Rolling cache
/data/positions          # Current positions
```

## Monitoring Architecture

### Metrics Collection

```
┌────────────────┐
│  AION Backend  │
│  (Prometheus)  │
└───────┬────────┘
        │ Scrape
        │ /metrics
        ▼
┌────────────────┐
│  Prometheus    │
│   (Storage)    │
└───────┬────────┘
        │ Query
        ▼
┌────────────────┐
│    Grafana     │
│  (Dashboard)   │
└────────────────┘
```

### Key Metrics

- `dt_trades_total{side,bot}` - Trade counter
- `dt_cycle_duration_seconds` - Cycle execution time
- `dt_open_positions` - Current positions
- `dt_equity_dollars` - Account equity
- `dt_daily_pnl_dollars` - Daily P&L
- `dt_errors_total` - Error counter

## Design Decisions

### 1. File-Based Storage

**Why**: Simplicity, portability, no external dependencies

**Trade-offs**:
- ✅ No database setup
- ✅ Easy to backup/inspect
- ✅ Git-friendly (state tracking)
- ❌ No ACID transactions
- ❌ Limited query capabilities

### 2. Dual-Brain Architecture

**Why**: Separate concerns for different timeframes

**Benefits**:
- Independent scaling
- Different data requirements
- Failure isolation
- Specialized optimizations

### 3. Hot-Reloadable Configuration

**Why**: Operational flexibility without restarts

**Implementation**:
- File-based config (`dt_knobs.env`)
- Reloaded each cycle
- Allows live tuning

### 4. Emergency Stop File

**Why**: Simplest possible kill switch

**Benefits**:
- No service dependencies
- Shell scriptable
- Works across failures
- Instant activation

### 5. VIX Spike Protection

**Why**: Avoid trading during volatility spikes

**Implementation**:
- Fetch VIX level each cycle
- Compare against threshold
- Stand down if exceeded

## Scalability Considerations

### Current Limitations

- Single-process execution
- File-based state (not distributed)
- Manual scaling only

### Future Scalability

To scale beyond current limits:

1. **Horizontal Scaling**
   - Separate read/write APIs
   - Load balancer for UI endpoints
   - Shared storage (NFS, S3)

2. **Database Migration**
   - PostgreSQL for state
   - TimescaleDB for time-series
   - Redis for cache

3. **Microservices**
   - Separate data fetching service
   - Separate ML inference service
   - Message queue (RabbitMQ, Kafka)

## Security Architecture

### Defense in Depth

```
Layer 1: Network
  └─► Firewall (UFW)
  └─► VPN access
  └─► Rate limiting

Layer 2: Authentication
  └─► Admin token
  └─► API key rotation
  └─► Secret management

Layer 3: Application
  └─► Input validation
  └─► Dry run mode
  └─► Emergency stop

Layer 4: Data
  └─► Encrypted secrets
  └─► Restrictive permissions
  └─► Audit logging
```

## Performance Characteristics

### Typical Performance

- **Nightly job**: 15-30 minutes (500 symbols)
- **DT cycle**: 2-8 seconds (150 symbols)
- **Model inference**: <100ms per symbol
- **Order latency**: 50-200ms (Alpaca API)

### Bottlenecks

1. **Data Fetching**: Network I/O to external APIs
2. **Feature Engineering**: CPU-bound calculations
3. **Model Inference**: CPU-bound (LightGBM is fast)
4. **File I/O**: Rolling cache save/load

### Optimization Strategies

- Reduce universe size
- Enable candidate filtering
- Cache frequently accessed data
- Parallel feature computation (future)

---

For operational procedures, see DEPLOYMENT.md  
For development guidelines, see CONTRIBUTING.md
