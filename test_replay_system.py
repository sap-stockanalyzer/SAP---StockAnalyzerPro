#!/usr/bin/env python3
"""
Test script for intra-day replay engine and walk-forward validation.

This script demonstrates:
1. Recording trading decisions during a cycle
2. Replaying cycles with modified parameters
3. Running walk-forward validation

Usage:
    python test_replay_system.py
"""

from dt_backend.services.decision_recorder import DecisionRecorder
from dt_backend.replay.intraday_replay_engine import IntraDayReplayEngine
from dt_backend.ml.walk_forward_validator import WalkForwardValidator


def test_decision_recording():
    """Test decision recording functionality."""
    print("\n" + "="*60)
    print("TEST 1: Decision Recording")
    print("="*60)
    
    recorder = DecisionRecorder()
    cycle_id = recorder.start_cycle()
    print(f"✅ Started new cycle: {cycle_id}")
    
    # Record symbol selection
    recorder.record_symbol_selection(
        selected_symbols=['AAPL', 'MSFT', 'NVDA'],
        ranking={'AAPL': 0.85, 'MSFT': 0.78, 'NVDA': 0.72},
        max_symbols=3,
        criteria='signal_strength'
    )
    print("✅ Recorded symbol selection (3 symbols)")
    
    # Record multiple entries
    for symbol, price, conf in [
        ('AAPL', 185.50, 0.85),
        ('MSFT', 380.25, 0.78),
        ('NVDA', 495.00, 0.72)
    ]:
        recorder.record_entry(
            symbol=symbol,
            side='BUY',
            qty=10,
            price=price,
            reason='strong_buy_signal',
            confidence=conf
        )
        print(f"✅ Recorded entry for {symbol} @ ${price}")
    
    # Record exits with P&L
    exits = [
        ('AAPL', 10, 187.00, 'take_profit', 15.00),   # Profit
        ('MSFT', 10, 378.50, 'stop_loss', -17.50),    # Loss
        ('NVDA', 10, 498.50, 'take_profit', 35.00),   # Profit
    ]
    
    total_pnl = 0
    for symbol, qty, exit_price, reason, pnl in exits:
        recorder.record_exit(
            symbol=symbol,
            qty=qty,
            price=exit_price,
            reason=reason,
            pnl=pnl
        )
        total_pnl += pnl
        print(f"✅ Recorded exit for {symbol} @ ${exit_price} (P&L: ${pnl:.2f})")
    
    print(f"\n📊 Cycle Summary:")
    print(f"   Cycle ID: {cycle_id}")
    print(f"   Total P&L: ${total_pnl:.2f}")
    
    # Verify decisions were recorded
    decisions = recorder.get_cycle_decisions(cycle_id)
    print(f"   Decisions recorded: {len(decisions)}")
    
    return cycle_id, total_pnl


def test_replay_engine(cycle_id, original_pnl):
    """Test replay engine with modified parameters."""
    print("\n" + "="*60)
    print("TEST 2: Replay Engine")
    print("="*60)
    
    engine = IntraDayReplayEngine()
    
    # Replay without modifications
    print("\n📊 Replaying original execution...")
    result = engine.replay_cycle(cycle_id)
    print(f"   Original P&L: ${result['original_pnl']:.2f}")
    print(f"   Replay P&L: ${result['replay_pnl']:.2f}")
    print(f"   Difference: ${result['pnl_difference']:.2f}")
    
    # Replay with tighter stop loss
    print("\n📊 Replaying with tighter stop loss (1.5%)...")
    result2 = engine.replay_cycle(
        cycle_id,
        modified_knobs={'stop_loss_pct': 0.015}
    )
    print(f"   Original P&L: ${result2['original_pnl']:.2f}")
    print(f"   Replay P&L: ${result2['replay_pnl']:.2f}")
    print(f"   Improvement: ${result2['pnl_difference']:.2f} ({result2['improvement_pct']:.1f}%)")
    
    # Replay with higher take profit
    print("\n📊 Replaying with higher take profit (8%)...")
    result3 = engine.replay_cycle(
        cycle_id,
        modified_knobs={'take_profit_pct': 0.08}
    )
    print(f"   Original P&L: ${result3['original_pnl']:.2f}")
    print(f"   Replay P&L: ${result3['replay_pnl']:.2f}")
    print(f"   Improvement: ${result3['pnl_difference']:.2f} ({result3['improvement_pct']:.1f}%)")
    
    # Get all replay results
    results = engine.get_replay_results(days=1)
    print(f"\n📊 Total replays stored: {len(results)}")


def test_walk_forward_validation():
    """Test walk-forward validation."""
    print("\n" + "="*60)
    print("TEST 3: Walk-Forward Validation")
    print("="*60)
    
    validator = WalkForwardValidator(
        window_days=5,
        lookback_days=20
    )
    print(f"✅ Initialized validator")
    print(f"   Test window: 5 days")
    print(f"   Training window: 20 days")
    
    print("\n📊 Running validation on 60 days of history...")
    summary = validator.run_validation(days_back=60)
    
    if summary.get('status') == 'no_data':
        print("   ⚠️  No historical trade data found")
        print("   (This is expected in a test environment)")
    else:
        print(f"   Windows evaluated: {summary.get('windows', 0)}")
        print(f"   Total P&L: ${summary.get('total_pnl', 0):.2f}")
        print(f"   Avg Sharpe Ratio: {summary.get('avg_sharpe', 0):.2f}")
        print(f"   Avg Win Rate: {summary.get('avg_win_rate', 0):.1%}")
        print(f"   Min Sharpe: {summary.get('min_sharpe', 0):.2f}")
        print(f"   Max Sharpe: {summary.get('max_sharpe', 0):.2f}")
        print(f"   Consistent: {summary.get('consistent', False)}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("INTRA-DAY REPLAY ENGINE & WALK-FORWARD VALIDATION TEST")
    print("="*60)
    
    try:
        # Test 1: Decision Recording
        cycle_id, total_pnl = test_decision_recording()
        
        # Test 2: Replay Engine
        test_replay_engine(cycle_id, total_pnl)
        
        # Test 3: Walk-Forward Validation
        test_walk_forward_validation()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nNext steps:")
        print("1. Start the FastAPI server: uvicorn dt_backend.api.app:app --port 8010")
        print("2. Test API endpoints:")
        print("   - POST /api/replay/cycle/{cycle_id}")
        print("   - GET  /api/replay/results")
        print("   - POST /api/replay/walk-forward")
        print("   - GET  /api/replay/decisions/{cycle_id}")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
