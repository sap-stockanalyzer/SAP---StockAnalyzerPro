# AION Analytics - Alert System

## Slack Channel Architecture

| Channel | Purpose | Alert Types |
|---------|---------|-------------|
| **#errors-tracebacks** | Critical errors, crashes | Unhandled exceptions, circuit breakers (@channel) |
| **#trading-alerts** | Trading system alerts | Emergency stops, risk breaches (@channel) |
| **#day_trading** | DT-specific alerts | Cycle status, intraday trades |
| **#swing_trading** | Swing/EOD bot alerts | EOD trades, swing positions |
| **#nightly-logs-summary** | Nightly job status | Job completions, phase summaries |
| **#daily-pnl** | PnL tracking | Daily/weekly PnL, equity updates |
| **#reports** | Insights & metrics | Model metrics, regime changes |
| **#testing** | Development only | Test alerts |

## Setup

1. Copy webhook URLs to `.env`:
   ```bash
   SLACK_WEBHOOK_ERRORS=https://hooks.slack.com/...
   SLACK_WEBHOOK_TRADING=https://hooks.slack.com/...
   # ... etc
   ```

2. Configure alert settings (optional):
   ```bash
   # Rate limiting: seconds between identical alerts (default: 300)
   ALERT_RATE_LIMIT_SECONDS=300
   
   # Slack request timeout in seconds (default: 5)
   SLACK_TIMEOUT_SECONDS=5
   ```

3. Test all channels:
   ```bash
   curl -X POST http://localhost:8000/api/test-alerts
   ```

4. Check each Slack channel for test message!

## Usage

```python
from backend.monitoring.alerting import (
    alert_error,      # → #errors-tracebacks (with @channel)
    alert_critical,   # → #trading-alerts (with @channel)
    alert_dt,         # → #day_trading
    alert_swing,      # → #swing_trading
    alert_nightly,    # → #nightly-logs-summary
    alert_pnl,        # → #daily-pnl
    alert_report,     # → #reports
)

# Examples

# ✅ CORRECT: DT trade → #day_trading
alert_dt("Trade Executed", f"BUY {symbol} @ ${price}")

# ✅ CORRECT: DT position exit → #day_trading
alert_dt("Position Closed: AAPL", "Exit reason: stop_hit", 
         context={"PnL": "-1.2%", "Hold": "15 min"})

# ✅ CORRECT: DT cycle completion → #day_trading
alert_dt("DT Cycle Complete", "Lane: FAST | Orders: 3")

# ✅ CORRECT: End-of-day PnL summary → #daily-pnl
alert_pnl(f"Daily PnL: ${pnl:+.2f}", f"MTD: ${mtd:+.2f}")

# ❌ WRONG: DT activity should NOT go to #daily-pnl
# alert_pnl("Trade Executed", "BUY AAPL")  # DON'T DO THIS!

# ✅ CORRECT: Nightly job status → #nightly-logs-summary  
alert_nightly("Nightly Job Complete", "All phases succeeded")

# ✅ CORRECT: System error → #errors-tracebacks
alert_error("Database Connection Failed", "Cannot connect to Redis")
```

## Alert Context

Add structured context to alerts:

```python
alert_critical(
    "High Loss Day",
    "Daily loss exceeded threshold",
    channel="trading",
    context={
        "PnL": f"${daily_pnl:.2f}",
        "Threshold": "$500",
        "Action": "Review positions"
    }
)
```

Results in Slack message with formatted fields:
```
🚨 High Loss Day
Daily loss exceeded threshold

PnL: $-550.23
Threshold: $500  
Action: Review positions

⏰ 2026-01-16 15:30:00 UTC
```

## Alert Levels

### Critical (Red 🚨)
- Emergency stops
- System crashes
- Critical errors
- Uses @channel mention when `mention_channel=True`

### Warning (Orange ⚠️)
- Non-critical errors
- Risk threshold breaches
- Service degradations

### Info (Green ℹ️)
- Trade executions
- Job completions
- Status updates
- PnL reports

## Rate Limiting

Alerts are rate-limited to prevent spam:
- Default: 300 seconds (5 minutes) between identical alerts
- Configure via `ALERT_RATE_LIMIT_SECONDS` environment variable
- Set to `0` to disable rate limiting

## Integration Points

### Emergency Stop
```python
# dt_backend/risk/emergency_stop_dt.py
trigger_emergency_stop("market_volatility")
# → Sends alert to #trading-alerts with @channel
```

### Nightly Job
```python
# backend/jobs/nightly_job.py
# Automatically sends alerts:
# - Success → #nightly-logs-summary
# - Failure → #errors-tracebacks
```

### Unhandled Exceptions
```python
# backend/core/error_handler.py
# Global exception handler sends all unhandled exceptions
# → #errors-tracebacks with @channel
```

### Circuit Breaker
```python
from backend.core.error_handler import CircuitBreaker

breaker = CircuitBreaker("external_api", failure_threshold=5, cooldown_seconds=60)

if breaker.can_execute():
    try:
        result = external_api.call()
        breaker.record_success()
    except Exception:
        breaker.record_failure()
        # → Sends alert to #errors-tracebacks when circuit opens
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SLACK_WEBHOOK_ERRORS` | - | Webhook URL for #errors-tracebacks |
| `SLACK_WEBHOOK_TRADING` | - | Webhook URL for #trading-alerts |
| `SLACK_WEBHOOK_DT` | - | Webhook URL for #day_trading |
| `SLACK_WEBHOOK_SWING` | - | Webhook URL for #swing_trading |
| `SLACK_WEBHOOK_NIGHTLY` | - | Webhook URL for #nightly-logs-summary |
| `SLACK_WEBHOOK_PNL` | - | Webhook URL for #daily-pnl |
| `SLACK_WEBHOOK_REPORTS` | - | Webhook URL for #reports |
| `SLACK_WEBHOOK_TESTING` | - | Webhook URL for #testing |
| `ALERT_RATE_LIMIT_SECONDS` | 300 | Seconds between identical alerts (0 = no limit) |
| `SLACK_TIMEOUT_SECONDS` | 5 | HTTP timeout for Slack API calls |

### Performance Notes

- **Timeout**: Default 5-second timeout prevents blocking in high-throughput scenarios
- **Rate Limiting**: Prevents alert spam; identical alerts deduplicated within window
- **Non-Blocking**: Failed alerts don't crash application, just log errors
- **Graceful Degradation**: If webhook not configured, alert skipped with log message

## Troubleshooting

### No alerts received
1. Check webhook URLs in `.env`
2. Verify Slack workspace permissions
3. Check logs for error messages: `grep "alerting" logs/*.log`

### Rate limiting too aggressive
Adjust `ALERT_RATE_LIMIT_SECONDS` in `.env` or pass `skip_rate_limit=True`:
```python
alert_critical("Important", "Message", skip_rate_limit=True)
```

### Wrong channel
Check channel parameter:
```python
# Correct
alert_critical("Stop", "Trading halted", channel="trading")

# Wrong - goes to default channel
alert_critical("Stop", "Trading halted")  # channel defaults to "trading"
```

## Channel Routing Rules

⚠️ **IMPORTANT**: Use the correct alert function for each context:

### Day Trading (DT) Alerts → `alert_dt()` → #day_trading
- **Position exits** (stop hit, take profit, time stop, scratch)
- **Trade entries** (BUY/SELL executions)
- **Cycle completions** (fast/slow lane status)
- **DT bot activity** (ORB, momentum, etc.)

❌ **DO NOT** use `alert_pnl()` for DT trades or positions!

### PnL Reports → `alert_pnl()` → #daily-pnl
- **End-of-day PnL summaries** (daily totals)
- **Weekly/monthly PnL** aggregates
- **Equity curve updates**
- **Performance metrics** (win rate, Sharpe, etc.)

### Swing Trading → `alert_swing()` → #swing_trading
- **Swing position entries/exits**
- **EOD bot activity**
- **Multi-day holds**

### Nightly Jobs → `alert_nightly()` → #nightly-logs-summary
- **Job completions** (success/failure)
- **Phase summaries**
- **Data pipeline status**

### Errors → `alert_error()` → #errors-tracebacks
- **Unhandled exceptions**
- **Critical system failures**
- **Circuit breaker trips**

### Reports → `alert_report()` → #reports
- **Model performance metrics**
- **Regime change detection**
- **Analysis insights**

## Best Practices

1. **Use appropriate channels**: Route alerts to their intended channel for better organization
2. **DT ≠ PnL**: Day trading alerts go to #day_trading, NOT #daily-pnl
3. **Add context**: Include relevant details in the `context` parameter
4. **Use @channel sparingly**: Only for truly critical alerts that require immediate attention
5. **Test before deploying**: Use the test endpoint to verify all channels work
6. **Monitor alert volume**: Too many alerts = alert fatigue

## Security

⚠️ **Never commit webhook URLs to version control!**

- Store webhooks in `.env` (gitignored)
- Use different webhooks for dev/staging/production
- Rotate webhooks if compromised
- Restrict webhook permissions in Slack
