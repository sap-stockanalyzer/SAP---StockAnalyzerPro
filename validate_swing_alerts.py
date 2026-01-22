#!/usr/bin/env python3
"""
Validation script for Swing Bot Slack Alerts

This script demonstrates the swing bot alert functionality with sample data.
It doesn't send actual Slack messages (unless webhook is configured).
"""

import os
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.bots.base_swing_bot import SwingBot, SwingBotConfig, Position, BotState, Trade


def create_sample_config():
    """Create sample swing bot configuration."""
    return SwingBotConfig(
        bot_key="demo_1w",
        horizon="1w",
        max_positions=5,
        base_risk_pct=0.02,
        conf_threshold=0.55,
        stop_loss_pct=0.08,
        take_profit_pct=0.12,
        max_weight_per_name=0.25,
        initial_cash=100000.0,
        low_conf_max_fraction=0.20,
        starter_fraction=0.50,
        add_fraction=0.33,
        max_build_stages=3,
        min_days_between_adds=2,
        add_conf_extra=0.03,
        allow_build_adds=True,
        min_hold_days=2,
        time_stop_days=10,
        exit_confirmations=2,
        exit_conf_buffer=0.05,
        use_phit=True,
        min_phit=0.55,
        loss_est_pct=0.06,
        require_positive_ev=True,
        ev_power=1.0,
    )


def create_sample_rolling():
    """Create sample rolling data."""
    return {
        "AAPL": {
            "price": 150.0,
            "policy": {
                "intent": "BUY",
                "confidence": 0.78,
                "score": 0.15,
            },
            "predictions": {
                "1w": {
                    "predicted_return": 0.052,
                }
            }
        },
        "TSLA": {
            "price": 220.0,
            "policy": {
                "intent": "BUY",
                "confidence": 0.42,
                "score": 0.05,
            },
            "predictions": {
                "1w": {
                    "predicted_return": 0.023,
                }
            }
        },
        "_GLOBAL": {
            "regime": {
                "label": "bull"
            }
        }
    }


def main():
    print("=" * 70)
    print("Swing Bot Slack Alerts - Validation Demo")
    print("=" * 70)
    print()
    
    # Check if webhook is configured
    webhook = os.getenv("SLACK_WEBHOOK_SWING", "")
    if webhook:
        print("✅ Slack webhook configured - alerts WILL be sent to #swing_trading")
    else:
        print("⚠️  No Slack webhook configured - alerts will be logged only")
    print()
    
    # Create bot
    config = create_sample_config()
    bot = SwingBot(config)
    rolling = create_sample_rolling()
    
    # Demo 1: BUY Alert
    print("Demo 1: BUY Alert")
    print("-" * 70)
    print("Sending BUY alert for AAPL...")
    bot._send_buy_alert(
        symbol="AAPL",
        qty=100.0,
        price=150.0,
        rolling=rolling,
    )
    print("✅ BUY alert sent\n")
    
    # Demo 2: SELL Alert (Profit)
    print("Demo 2: SELL Alert (Profit)")
    print("-" * 70)
    print("Sending SELL alert for AAPL with profit...")
    bot._send_sell_alert(
        symbol="AAPL",
        qty=100.0,
        price=155.50,
        entry_price=150.0,
        reason="TAKE_PROFIT",
        entry_ts="2024-01-15T09:30:00Z",
    )
    print("✅ SELL alert sent\n")
    
    # Demo 3: SELL Alert (Loss)
    print("Demo 3: SELL Alert (Loss)")
    print("-" * 70)
    print("Sending SELL alert for TSLA with loss...")
    bot._send_sell_alert(
        symbol="TSLA",
        qty=50.0,
        price=210.0,
        entry_price=220.0,
        reason="STOP_LOSS",
        entry_ts="2024-01-20T10:00:00Z",
    )
    print("✅ SELL alert sent\n")
    
    # Demo 4: Rejection Alert (if enabled)
    send_rejections = os.getenv("SWING_SEND_REJECTIONS", "0") == "1"
    print("Demo 4: Rejection Alert")
    print("-" * 70)
    if send_rejections:
        print("Sending REJECTION alert for TSLA (confidence below threshold)...")
        bot._send_rejection_alert(
            symbol="TSLA",
            price=220.0,
            reason="conf_below_threshold",
            details={
                "confidence": 0.42,
                "conf_threshold": 0.55,
                "expected_return": 0.023,
            },
        )
        print("✅ REJECTION alert sent\n")
    else:
        print("⚠️  SWING_SEND_REJECTIONS=0 - rejection alerts disabled")
        print("   Set SWING_SEND_REJECTIONS=1 to enable\n")
    
    # Demo 5: Rebalance Summary
    print("Demo 5: Rebalance Summary")
    print("-" * 70)
    print("Sending rebalance summary...")
    
    trades = [
        Trade(
            t="2024-01-22T09:30:00Z",
            symbol="AAPL",
            side="BUY",
            qty=100.0,
            price=150.0,
            reason="NEW_ENTRY",
        ),
        Trade(
            t="2024-01-22T09:31:00Z",
            symbol="MSFT",
            side="BUY",
            qty=50.0,
            price=380.0,
            reason="NEW_ENTRY",
        ),
        Trade(
            t="2024-01-22T09:32:00Z",
            symbol="TSLA",
            side="SELL",
            qty=75.0,
            price=220.0,
            reason="REMOVE_FROM_UNIVERSE",
            pnl=500.0,
        ),
    ]
    
    state = BotState(
        cash=50000.0,
        last_equity=125432.18,
        last_updated="2024-01-22T09:35:00Z",
        positions={
            "AAPL": Position(qty=100.0, entry=150.0, stop=138.0, target=168.0, goal_qty=100.0),
            "MSFT": Position(qty=50.0, entry=380.0, stop=350.0, target=427.0, goal_qty=50.0),
        },
    )
    
    rejection_counts = {
        "conf_below_threshold": 623,
        "non_positive_ev": 412,
        "intent_not_buy": 98,
        "phit_below_threshold": 20,
    }
    
    bot._send_rebalance_summary(
        trades=trades,
        universe_analyzed=1200,
        universe_qualified=47,
        rejection_counts=rejection_counts,
        state=state,
    )
    print("✅ Rebalance summary sent\n")
    
    print("=" * 70)
    print("Validation Complete!")
    print("=" * 70)
    print()
    print("All alert types have been demonstrated:")
    print("  ✅ BUY alerts (with confidence, expected return, P(Hit))")
    print("  ✅ SELL alerts (with PnL, reason, hold duration)")
    if send_rejections:
        print("  ✅ REJECTION alerts (optional, enabled)")
    else:
        print("  ⚠️  REJECTION alerts (optional, disabled)")
    print("  ✅ REBALANCE SUMMARY (with universe stats, rejections, portfolio)")
    print()
    
    if not webhook:
        print("💡 Tip: Set SLACK_WEBHOOK_SWING to test actual Slack delivery")
        print("   Example: export SLACK_WEBHOOK_SWING=https://hooks.slack.com/...")
    else:
        print("✅ Alerts were sent to #swing_trading channel")
    print()


if __name__ == "__main__":
    main()
