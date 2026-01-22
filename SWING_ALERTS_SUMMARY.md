# Swing Bot Trade Alerts - Implementation Summary

## Overview
Successfully implemented comprehensive Slack alerts for the Swing Bot trading system, providing real-time visibility into trading decisions, rejections, and portfolio performance.

## Features Implemented

### 1. Trade Entry Alerts (BUY)
- Symbol, quantity, and entry price
- AI confidence level and threshold comparison
- Expected return percentage
- Calibrated P(Hit) probability
- Entry timestamp
- Reason for selection

### 2. Trade Exit Alerts (SELL)
- Symbol, quantity, and exit price
- Profit/Loss in dollars and percentage
- Exit reason (Stop Loss, Take Profit, Time Stop, AI Confirmed, etc.)
- Hold duration (days and hours)
- Exit timestamp

### 3. Rejection Alerts (Optional)
- Symbol and current price
- Rejection reason (confidence, EV, P(Hit), etc.)
- Detailed metrics that caused rejection
- Controlled by `SWING_SEND_REJECTIONS` env var
- Rate-limited by `SWING_SEND_REJECTIONS_MAX`

### 4. Rebalance Summary
- Total trades executed (buys and sells)
- Universe statistics (analyzed, qualified, rejected)
- Top rejection reasons with counts and percentages
- Current portfolio state (positions, cash, equity)

## Configuration

### Environment Variables

```bash
# Required: Slack webhook for #swing_trading channel
SLACK_WEBHOOK_SWING=https://hooks.slack.com/services/...

# Optional: Enable individual rejection alerts (default: 0)
SWING_SEND_REJECTIONS=0

# Optional: Max rejection alerts per rebalance (default: 20)
SWING_SEND_REJECTIONS_MAX=20
```

### Usage Modes

**Default Mode (Summary Only):**
- BUY/SELL trade alerts
- Rebalance summary with aggregated rejection stats
- No individual rejection spam

**Verbose Mode:**
- All default alerts PLUS
- Individual rejection alerts for debugging
- Rate-limited to prevent channel flooding

## Integration Points

### Modified Files
1. `backend/bots/base_swing_bot.py` - Core alert logic
2. `.env.example` - Configuration documentation
3. `tests/unit/test_swing_bot_alerts.py` - Comprehensive tests
4. `validate_swing_alerts.py` - Validation script

### Alert Call Sites
1. `rebalance_full()` - BUY/SELL for rebalancing
2. `apply_loop_risk_checks()` - SELL for risk exits (SL/TP/Time/AI)
3. `build_ai_ranked_universe()` - Rejection tracking (optional)
4. `run_full()` - Rebalance summary

## Testing

### Test Coverage
- 13 unit tests, all passing
- BUY alert formatting and delivery
- SELL alert formatting (profit and loss)
- Rejection alert formatting
- Rebalance summary content and format
- Error handling and fallbacks
- Conditional alert sending based on config

### Validation
```bash
# Run unit tests
pytest tests/unit/test_swing_bot_alerts.py -v

# Run validation demo
python validate_swing_alerts.py

# Run all tests
pytest tests/unit/
```

## Alert Format Examples

### BUY Alert
```
📈 Swing Bot 1w - BUY
Symbol: AAPL
Qty: 100.00 shares @ $150.00
Confidence: 78.0% (> threshold: 55.0%)
Expected Return: +5.20%
P(Hit): 71.7%
Why Selected: Top rank in AI universe, strong signal + positive EV
Entry Time: 2026-01-22T04:58:48Z
```

### SELL Alert
```
📉 Swing Bot 1w - SELL
Symbol: AAPL
Qty: 100.00 shares @ $155.50
PnL: +$550.00 (+3.67% return)
Reason: Take Profit
Hold Duration: 5 days, 3 hours
Exit Time: 2026-01-22T04:58:48Z
```

### Rebalance Summary
```
📊 Swing Bot 1w - Rebalance Complete
Trades Executed: 3 total
  ✅ Buys: 2 positions entered
  ✅ Sells: 1 positions closed
  
Universe Analyzed: 1200 symbols
  ✅ Qualified: 47 symbols
  ❌ Rejected: 1153 symbols

Top Rejections:
  • Conf Below Threshold: 623 (54%)
  • Non Positive Ev: 412 (36%)
  • Intent Not Buy: 98 (8%)

Portfolio:
  • Positions: 2 open
  • Cash: $50,000.00
  • Equity: $125,432.18
```

## Error Handling

- All alert methods wrapped in try-except blocks
- Failures logged but don't crash the bot
- Works with or without webhook configured
- Graceful degradation if Slack is unavailable

## Performance Considerations

- Alerts sent asynchronously (skip_rate_limit=True)
- Minimal overhead on trading logic
- Rate-limiting prevents channel flooding
- Summary-first approach reduces noise

## Benefits

1. **Visibility**: Real-time insight into trading decisions
2. **Debugging**: Understand why symbols are rejected
3. **Monitoring**: Track portfolio changes and performance
4. **Accountability**: Audit trail of all trades
5. **Configurability**: Control verbosity based on needs

## Success Metrics

✅ All 13 tests passing
✅ No regressions in existing tests
✅ Validation script demonstrates all alert types
✅ Error handling prevents crashes
✅ Works with or without Slack webhook
✅ Configurable verbosity (summary vs. detailed)
✅ Comprehensive documentation

## Next Steps

1. Configure `SLACK_WEBHOOK_SWING` in production `.env`
2. Monitor alerts in #swing_trading channel
3. Adjust `SWING_SEND_REJECTIONS` based on needs
4. Review rejection patterns to improve strategy

## Support

- Test with: `python validate_swing_alerts.py`
- Check logs: `logs/bots/swing_bot.log`
- Review code: `backend/bots/base_swing_bot.py` (lines 960-1152)
- Run tests: `pytest tests/unit/test_swing_bot_alerts.py -v`
