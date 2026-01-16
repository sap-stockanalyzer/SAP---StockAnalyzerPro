# Day Trading Continuous Learning System

## Overview

The Day Trading Continuous Learning System is a comprehensive feedback loop that enables AION's intraday trading engine to learn from trade outcomes, missed opportunities, and automatically adjust execution parameters based on performance.

## Architecture

### Phase 1: Trade Outcome Learning + DT Brain

#### Components

1. **Trade Outcome Analyzer** (`dt_backend/ml/trade_outcome_analyzer.py`)
   - Analyzes closed trades and extracts learning signals
   - Tracks performance metrics by symbol, sector, regime, and time-of-day
   - Calculates win rates, profit factors, Sharpe ratios
   - Stores trade outcomes in compressed JSONL format

2. **DT Brain** (`dt_backend/core/dt_brain.py`)
   - Behavioral learning system that auto-adjusts execution knobs
   - Uses EMA smoothing (20% per update) for gradual adjustments
   - Adjusts 7 key knobs based on performance:
     - `DT_EXEC_MIN_CONF` (confidence threshold)
     - `DT_MAX_POSITIONS` (concurrent positions)
     - `DT_STOP_LOSS_PCT` (stop loss %)
     - `DT_TAKE_PROFIT_PCT` (take profit %)
     - `DT_POSITION_SIZE_BASE_USD` (position sizing)
     - `DT_MAX_ORDERS_PER_CYCLE` (order rate)
     - `DT_MIN_TRADE_GAP_MINUTES` (trade spacing)

3. **Post-Market Analysis Job** (`dt_backend/jobs/post_market_analysis.py`)
   - Runs daily at market close (16:05 ET)
   - Orchestrates all learning components
   - Generates daily learning reports

### Phase 2: Missed Opportunities + Auto Retrain

#### Components

1. **Missed Opportunity Tracker** (`dt_backend/ml/missed_opportunity_tracker.py`)
   - Logs high-confidence signals (≥60%) that were NOT traded
   - Tracks reasons: max_positions, confidence_threshold, risk_limit, etc.
   - Calculates hypothetical PnL to assess missed opportunities
   - Suggests knob adjustments based on profitable missed signals

2. **Automatic Retraining System** (`dt_backend/services/auto_retrain_dt.py`)
   - Monitors model performance against baseline
   - Triggers retrain when:
     - Win rate degrades 8%+ vs baseline
     - Accuracy degrades 10%+ vs baseline
     - Profit factor drops below 1.0 (from >1.2)
     - 7+ days since last retrain
   - Full workflow: rebuild dataset → train models → validate → deploy if better
   - Validation gate: only deploys if new accuracy ≥ 95% of current

### Phase 3: Dashboard + API

#### Components

1. **Learning Router** (`dt_backend/routers/learning_router.py`)
   - REST API endpoints for learning metrics
   - Endpoints:
     - `GET /api/dt/learning/metrics` - Current learning metrics
     - `GET /api/dt/learning/trade-outcomes` - Recent trade outcomes
     - `GET /api/dt/learning/missed-opportunities` - Missed signals analysis
     - `GET /api/dt/learning/brain-status` - Current knob values
     - `POST /api/dt/learning/retrain` - Manually trigger retrain
     - `GET /api/dt/learning/performance-history` - Historical metrics
     - `GET /api/dt/learning/health` - System health check

2. **Learning Dashboard** (`frontend/app/dt-learning/page.tsx`)
   - Real-time visualization of learning metrics
   - Displays:
     - Performance trends (7-day and 30-day windows)
     - Model health (days since retrain, confidence calibration)
     - Missed opportunities with suggestions
     - DT Brain knob status (current vs default values)
     - Baseline comparison
   - Auto-refreshes every 60 seconds

## Data Storage

All learning data is stored under `da_brains/dt_learning/`:

```
da_brains/dt_learning/
├── trade_outcomes.jsonl.gz       # All closed trades
├── performance_metrics.json      # Current performance metrics
├── baseline_performance.json     # Baseline from last retrain
├── missed_signals.jsonl.gz       # High-confidence missed signals
├── missed_analysis.json          # Missed opportunity analysis
├── knob_adjustments.jsonl        # History of knob adjustments
├── retrain_log.jsonl            # Retrain history
├── last_retrain.json            # Last retrain timestamp
└── reports/                      # Daily post-market reports
    ├── post_market_2026-01-16.json
    └── post_market_latest.json
```

## Integration Points

### 1. Trade Execution
- **File:** `dt_backend/services/position_manager_dt.py`
- **Function:** `record_exit()`
- **Integration:** Calls `analyze_trade_outcome()` on every trade exit

### 2. Signal Filtering
- **File:** `dt_backend/engines/trade_executor.py`
- **Integration:** Calls `track_missed_signal()` when high-confidence signal (≥60%) is not traded

### 3. Post-Market Workflow
- **File:** `dt_backend/jobs/dt_scheduler.py`
- **Integration:** Calls `run_post_market_analysis()` after EOD cleanup

## Usage

### Viewing Learning Metrics

Access the dashboard at: `http://localhost:3000/dt-learning`

### Manually Triggering Retrain

```bash
curl -X POST http://localhost:8000/api/dt/learning/retrain
```

### Querying Performance

```python
from dt_backend.ml.trade_outcome_analyzer import TradeOutcomeAnalyzer

analyzer = TradeOutcomeAnalyzer()
perf = analyzer.get_performance_window(days=7)
print(f"Win Rate: {perf['win_rate']:.2%}")
print(f"Profit Factor: {perf['profit_factor']:.2f}")
```

### Checking Missed Opportunities

```python
from dt_backend.ml.missed_opportunity_tracker import MissedOpportunityTracker

tracker = MissedOpportunityTracker()
patterns = tracker.analyze_missed_patterns()
print(f"Missed PnL: ${patterns['missed_pnl_usd']:.2f}")
print(f"Suggestions: {patterns['suggestions']}")
```

## Knob Adjustment Logic

### Confidence Threshold
- **Increase to 0.32** if win rate < 45% (conservative mode)
- **Decrease to 0.22** if win rate ≥ 60% and avg confidence ≥ 70% (aggressive mode)

### Max Positions
- **Increase to 5** if Sharpe ratio > 2.0 and profit factor > 1.5
- **Decrease to 2** if drawdown > 5%

### Stop Loss
- **Tighten to 1.5%** if average loss > -1.5%
- **Widen to 2.5%** if average loss < -3%

### Take Profit
- **Increase to 6%** if average winning hold time > 60 min (let winners run)
- **Decrease to 2.5%** if average winning hold time < 20 min (scalping mode)

### Position Sizing
- **Increase to $1500** after 5+ consecutive wins (hot streak)
- **Decrease to $700** after 3+ consecutive losses (cold streak)

### Regime-Based Adjustments
- **Volatile Regime:** Widen stops to 3%, reduce max positions to 2
- **Range Regime:** Tighten stops to 1.5%, increase order rate to 4

## Performance Metrics

### Tracked Metrics
- **Win Rate:** % of profitable trades
- **Accuracy:** Same as win rate (classification)
- **Profit Factor:** Total wins / Total losses
- **Sharpe Ratio:** Risk-adjusted returns
- **Average Win/Loss:** Mean PnL% for wins and losses
- **Hold Duration:** Average time in position
- **Drawdown:** Maximum peak-to-trough decline
- **Confidence Calibration:** Accuracy / Avg Confidence

### Aggregations
- **Global:** All trades combined
- **By Symbol:** Per-ticker performance
- **By Regime:** Performance in different market conditions
- **By Time-of-Day:** Open, mid-morning, lunch, afternoon, close
- **By Confidence Bucket:** Performance at different confidence levels

## Retrain Triggers

The system automatically triggers retraining when:

1. **Performance Degradation**
   - Win rate drops 8%+ below baseline
   - Accuracy drops 10%+ below baseline
   - Profit factor collapses (< 1.0 from > 1.2)

2. **Time-Based**
   - 7+ days since last successful retrain

3. **Manual**
   - Via API: `POST /api/dt/learning/retrain`
   - Via UI: Dashboard "Retrain" button (when implemented)

## Safety Features

### Never-Fail Principle
- All learning components are wrapped in try-except blocks
- Learning failures never crash the trading engine
- Learning is additive - doesn't modify core trading logic

### Validation Gate
- New models only deploy if accuracy ≥ 95% of current model
- Automatic rollback if new models underperform

### EMA Smoothing
- Knob adjustments use 80% old + 20% target
- Prevents wild swings in execution parameters

### Range Constraints
- All knobs have hard min/max ranges
- Prevents extreme values even during poor performance

## Testing

Run unit tests:

```bash
# Trade outcome analyzer
pytest tests/unit/test_trade_outcome_analyzer.py -v

# Missed opportunity tracker
pytest tests/unit/test_missed_opportunity_tracker.py -v
```

## Monitoring

### Daily Reports
- Stored in `da_brains/dt_learning/reports/`
- Contains status of all learning steps
- Viewable via API or file system

### Logs
- All learning activity logged to console
- Prefixed with `[trade_analyzer]`, `[missed_opp]`, `[auto_retrain]`, `[dt_brain]`, `[post_market]`
- Search logs: `grep "trade_analyzer" logs/intraday/*.log`

## Future Enhancements

### Phase 4 (Future)
- [ ] Multi-timeframe performance analysis
- [ ] Symbol-specific knob overrides
- [ ] Sector rotation learning
- [ ] News sentiment integration
- [ ] A/B testing framework for strategies

### Phase 5 (Future)
- [ ] Reinforcement learning for position sizing
- [ ] Ensemble model weight optimization
- [ ] Real-time learning (intra-day adjustments)
- [ ] Risk-adjusted portfolio optimization

## Troubleshooting

### Learning data not appearing
1. Check `da_brains/dt_learning/` exists and is writable
2. Verify trades are closing (check `dt_trades.jsonl`)
3. Check logs for learning errors

### Dashboard shows errors
1. Verify dt_backend API is running
2. Check API endpoint: `curl http://localhost:8000/api/dt/learning/health`
3. Verify frontend proxy configuration

### Retraining not triggering
1. Check baseline exists: `cat da_brains/dt_learning/baseline_performance.json`
2. Verify sufficient trade history (needs at least 10 trades)
3. Check retrain log: `cat da_brains/dt_learning/retrain_log.jsonl`

## Support

For issues or questions:
1. Check logs under `logs/intraday/`
2. Review learning reports in `da_brains/dt_learning/reports/`
3. Query health endpoint: `GET /api/dt/learning/health`
