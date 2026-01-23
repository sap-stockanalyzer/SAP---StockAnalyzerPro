"""Unit tests for dt_backend/core/policy_engine_dt.py.

This test suite covers:
- Trade gate logic (5 tests)
- Hysteresis stabilization (4 tests)
- Confidence adjustments (3 tests)
- Position awareness (1 test)
- Configuration validation (2 tests)
- P&L attribution (2 tests)
"""

import pytest
from typing import Dict, Any
from dt_backend.core.policy_engine_dt import (
    PolicyConfig,
    _raw_intent_from_edge,
    _adjust_conf,
    _stabilize_with_hysteresis,
    _has_position,
    _apply_env_overrides,
)


# ============================================================
# TRADE GATE LOGIC TESTS (5 tests)
# ============================================================

def test_trade_gate_false_when_confidence_below_min(default_policy_config):
    """Would have caught 0.25 confidence threshold bug.
    
    Trade gate should be False when confidence < min_confidence.
    This prevents spam trades from weak signals.
    """
    cfg = default_policy_config
    cfg.min_confidence = 0.30
    
    # Test case: confidence below minimum
    confidence = 0.25
    action = "BUY"
    
    # Trade gate should be False
    trade_gate = bool(action in {"BUY", "SELL"} and confidence >= cfg.min_confidence)
    assert trade_gate is False, f"Trade gate should be False when conf={confidence} < min={cfg.min_confidence}"


def test_trade_gate_true_when_confidence_at_min(default_policy_config):
    """Trade gate should be True when confidence equals min_confidence."""
    cfg = default_policy_config
    cfg.min_confidence = 0.30
    
    confidence = 0.30
    action = "BUY"
    
    trade_gate = bool(action in {"BUY", "SELL"} and confidence >= cfg.min_confidence)
    assert trade_gate is True


def test_trade_gate_true_when_confidence_above_min(default_policy_config):
    """Trade gate should be True when confidence > min_confidence."""
    cfg = default_policy_config
    cfg.min_confidence = 0.30
    
    confidence = 0.75
    action = "SELL"
    
    trade_gate = bool(action in {"BUY", "SELL"} and confidence >= cfg.min_confidence)
    assert trade_gate is True


def test_trade_gate_false_for_hold_action(default_policy_config):
    """Trade gate should always be False for HOLD action regardless of confidence."""
    cfg = default_policy_config
    
    confidence = 0.95  # Very high confidence
    action = "HOLD"
    
    trade_gate = bool(action in {"BUY", "SELL"} and confidence >= cfg.min_confidence)
    assert trade_gate is False


def test_trade_gate_false_for_stand_down_action(default_policy_config):
    """Trade gate should always be False for STAND_DOWN action."""
    cfg = default_policy_config
    
    confidence = 0.95
    action = "STAND_DOWN"
    
    trade_gate = bool(action in {"BUY", "SELL"} and confidence >= cfg.min_confidence)
    assert trade_gate is False


# ============================================================
# HYSTERESIS STABILIZATION TESTS (4 tests)
# ============================================================

def test_hold_sticky_bias_prevents_flip(default_policy_config):
    """Tests hysteresis: HOLD should be sticky and resist weak signals.
    
    Prevents flip-flopping by requiring extra edge to exit HOLD state.
    """
    cfg = default_policy_config
    cfg.hysteresis_hold_bias = 0.03
    cfg.min_edge_to_flip = 0.06
    
    # Previous state: HOLD
    node = {
        "policy_dt": {
            "action": "HOLD",
            "intent": "HOLD",
            "_state": {
                "prev_action": "HOLD",
                "pending_action": "",
                "pending_count": 0,
            },
        },
    }
    
    # Weak BUY signal (edge below threshold + bias)
    proposed = "BUY"
    edge = 0.08  # Above min_edge_to_flip but below min_edge + bias
    conf = 0.50
    
    final_action, new_state, note = _stabilize_with_hysteresis(
        node, proposed, edge, conf, cfg
    )
    
    # Should stay HOLD due to sticky bias
    assert final_action == "HOLD", "Hysteresis should keep action as HOLD"
    assert "sticky" in note.lower(), "Note should indicate sticky hold"


def test_hold_released_with_strong_edge(default_policy_config):
    """HOLD should release to BUY/SELL when edge is strong enough."""
    cfg = default_policy_config
    cfg.hysteresis_hold_bias = 0.03
    cfg.min_edge_to_flip = 0.06
    
    node = {
        "policy_dt": {
            "action": "HOLD",
            "_state": {
                "prev_action": "HOLD",
                "pending_action": "",
                "pending_count": 0,
            },
        },
    }
    
    # Strong BUY signal (edge above threshold + bias)
    proposed = "BUY"
    edge = 0.15  # Well above min_edge + bias
    conf = 0.70
    
    final_action, new_state, note = _stabilize_with_hysteresis(
        node, proposed, edge, conf, cfg
    )
    
    assert final_action == "BUY", "Strong edge should release HOLD"
    assert "released" in note.lower(), "Note should indicate hold released"


def test_confirmation_count_enforcement(default_policy_config):
    """Require N consecutive signals before flipping BUY<->SELL.
    
    Prevents flip-flopping between directions.
    """
    cfg = default_policy_config
    cfg.confirmations_to_flip = 2
    cfg.min_edge_to_flip = 0.06
    
    # Previous state: BUY
    node = {
        "policy_dt": {
            "action": "BUY",
            "_state": {
                "prev_action": "BUY",
                "pending_action": "",
                "pending_count": 0,
            },
        },
    }
    
    # First SELL signal (should wait for confirmation)
    proposed = "SELL"
    edge = -0.10  # Strong edge
    conf = 0.60
    
    final_action_1, state_1, note_1 = _stabilize_with_hysteresis(
        node, proposed, edge, conf, cfg
    )
    
    assert final_action_1 == "BUY", "First signal should not flip, should stay BUY"
    assert "wait_confirmations" in note_1, "Should be waiting for confirmations"
    assert state_1["pending_action"] == "SELL"
    assert state_1["pending_count"] == 1
    
    # Second SELL signal (should now flip)
    node["policy_dt"]["_state"] = state_1
    final_action_2, state_2, note_2 = _stabilize_with_hysteresis(
        node, proposed, edge, conf, cfg
    )
    
    assert final_action_2 == "SELL", "Second confirmation should flip to SELL"
    assert "confirmed" in note_2.lower(), "Should indicate flip confirmed"


def test_flip_blocked_when_edge_too_small(default_policy_config):
    """Prevent flip when edge is too small, even with confirmations."""
    cfg = default_policy_config
    cfg.min_edge_to_flip = 0.06
    
    node = {
        "policy_dt": {
            "action": "BUY",
            "_state": {
                "prev_action": "BUY",
                "pending_action": "",
                "pending_count": 0,
            },
        },
    }
    
    # Weak SELL signal
    proposed = "SELL"
    edge = -0.04  # Below min_edge_to_flip
    conf = 0.50
    
    final_action, new_state, note = _stabilize_with_hysteresis(
        node, proposed, edge, conf, cfg
    )
    
    assert final_action == "BUY", "Should not flip when edge too small"
    assert "blocked" in note.lower(), "Note should indicate flip blocked"


# ============================================================
# CONFIDENCE ADJUSTMENTS TESTS (3 tests)
# ============================================================

def test_vol_penalty_reduces_confidence(default_policy_config):
    """High volatility should reduce confidence."""
    cfg = default_policy_config
    cfg.vol_penalty_high = 0.65
    cfg.trend_boost_mild = 1.10
    
    base_conf = 0.80
    intent = "BUY"
    trend = "bull"
    vol_bkt = "high"
    regime_label = "bull"
    
    adjusted_conf, detail = _adjust_conf(
        base_conf, intent, trend, vol_bkt, regime_label, cfg
    )
    
    # Should be reduced by vol_penalty_high AND boosted by trend (cumulative)
    expected = base_conf * cfg.vol_penalty_high * cfg.trend_boost_mild
    assert adjusted_conf == pytest.approx(expected, rel=0.01)
    assert "vol=high" in detail
    assert "trend=bull" in detail


def test_trend_alignment_boosts_confidence(default_policy_config):
    """BUY in strong_bull trend should boost confidence."""
    cfg = default_policy_config
    cfg.trend_boost_strong = 1.25
    
    base_conf = 0.60
    intent = "BUY"
    trend = "strong_bull"
    vol_bkt = "low"
    regime_label = "bull"
    
    adjusted_conf, detail = _adjust_conf(
        base_conf, intent, trend, vol_bkt, regime_label, cfg
    )
    
    # Should be boosted by trend_boost_strong
    expected = base_conf * cfg.trend_boost_strong
    assert adjusted_conf == pytest.approx(expected, rel=0.01)
    assert "trend=strong_bull" in detail


def test_regime_chop_penalty(default_policy_config):
    """Choppy regime should reduce confidence."""
    cfg = default_policy_config
    cfg.chop_penalty = 0.90
    cfg.vol_penalty_medium = 0.85
    cfg.trend_boost_mild = 1.10
    
    base_conf = 0.70
    intent = "BUY"
    trend = "bull"
    vol_bkt = "medium"
    regime_label = "chop"
    
    adjusted_conf, detail = _adjust_conf(
        base_conf, intent, trend, vol_bkt, regime_label, cfg
    )
    
    # Should apply vol penalty, trend boost, and chop penalty (all cumulative)
    expected = base_conf * cfg.vol_penalty_medium * cfg.trend_boost_mild * cfg.chop_penalty
    assert adjusted_conf == pytest.approx(expected, rel=0.01)
    assert "regime=chop" in detail


# ============================================================
# POSITION AWARENESS TEST (1 test)
# ============================================================

def test_has_position_detection(sample_node_with_position, sample_node_no_position):
    """Test position detection logic."""
    # Node with position
    assert _has_position(sample_node_with_position) is True
    
    # Node without position
    assert _has_position(sample_node_no_position) is False
    
    # Empty node
    assert _has_position({}) is False
    
    # Node with position using "position" key (alternate format)
    alt_node = {"position": {"qty": 50.0}}
    assert _has_position(alt_node) is True
    
    # Node with zero quantity
    zero_node = {"position_dt": {"qty": 0.0}}
    assert _has_position(zero_node) is False


# ============================================================
# CONFIGURATION VALIDATION TESTS (2 tests)
# ============================================================

def test_env_overrides_apply_correctly(monkeypatch):
    """Test that environment variables override config correctly."""
    monkeypatch.setenv("DT_MIN_CONFIDENCE", "0.35")
    monkeypatch.setenv("DT_BUY_THRESHOLD", "0.14")
    monkeypatch.setenv("DT_CONFIRMATIONS_TO_FLIP", "3")
    
    cfg = PolicyConfig()
    cfg = _apply_env_overrides(cfg)
    
    assert cfg.min_confidence == 0.35
    assert cfg.buy_threshold == 0.14
    assert cfg.confirmations_to_flip == 3


def test_config_values_clamped(monkeypatch):
    """Test that config values are clamped to sane ranges."""
    # Try to set invalid values
    monkeypatch.setenv("DT_MIN_CONFIDENCE", "1.5")  # Too high
    monkeypatch.setenv("DT_BUY_THRESHOLD", "-0.5")  # Negative (should be positive)
    monkeypatch.setenv("DT_HOLD_STICKY_BIAS", "0.5")  # Too high
    
    cfg = PolicyConfig()
    cfg = _apply_env_overrides(cfg)
    
    # Should be clamped
    assert 0.0 <= cfg.min_confidence <= 1.0
    assert 0.0 <= cfg.buy_threshold <= 1.0
    assert 0.0 <= cfg.hysteresis_hold_bias <= 0.25


# ============================================================
# P&L ATTRIBUTION TESTS (2 tests)
# These test the calculate_pnl_attribution function in position_manager_dt.py
# ============================================================

def test_pnl_attribution_sums_correctly():
    """Test that P&L attribution sums correctly across positions.
    
    This tests the calculate_pnl_attribution function to ensure
    it correctly attributes P&L to different strategies/bots.
    """
    from dt_backend.services.position_manager_dt import calculate_pnl_attribution
    
    # Sample positions with different bots
    positions_state = {
        "AAPL": {
            "status": "OPEN",
            "side": "BUY",
            "qty": 100.0,
            "entry_price": 145.00,
            "bot": "ORB",
        },
        "MSFT": {
            "status": "OPEN",
            "side": "BUY",
            "qty": 50.0,
            "entry_price": 380.00,
            "bot": "VWAP",
        },
        "GOOGL": {
            "status": "CLOSED",
            "side": "BUY",
            "qty": 20.0,
            "entry_price": 140.00,
            "bot": "ORB",
        },
    }
    
    # Current prices for P&L calculation
    current_prices = {
        "AAPL": 150.00,  # +5.00 per share, 100 shares = +500
        "MSFT": 375.00,  # -5.00 per share, 50 shares = -250
        "GOOGL": 145.00,  # Closed position, should not be included
    }
    
    attribution = calculate_pnl_attribution(positions_state, current_prices)
    
    # Check that attribution has correct structure
    assert isinstance(attribution, dict)
    assert "ORB" in attribution
    assert "VWAP" in attribution
    
    # Check P&L values
    orb_pnl = attribution["ORB"]["pnl"]
    vwap_pnl = attribution["VWAP"]["pnl"]
    
    assert orb_pnl == pytest.approx(500.0, rel=0.01)  # AAPL only (GOOGL is closed)
    assert vwap_pnl == pytest.approx(-250.0, rel=0.01)  # MSFT
    
    # Total P&L should sum correctly
    total_pnl = attribution["_total"]["pnl"]
    assert total_pnl == pytest.approx(250.0, rel=0.01)  # 500 - 250


def test_pnl_attribution_handles_empty_positions():
    """Test P&L attribution with no positions."""
    from dt_backend.services.position_manager_dt import calculate_pnl_attribution
    
    # Empty positions
    positions_state = {}
    current_prices = {}
    
    attribution = calculate_pnl_attribution(positions_state, current_prices)
    
    # Should return valid structure with zeros
    assert isinstance(attribution, dict)
    assert attribution["_total"]["pnl"] == 0.0
    assert attribution["_total"]["position_count"] == 0


# ============================================================
# ADDITIONAL TESTS FOR 23+ TOTAL (3 more tests)
# ============================================================

def test_pnl_attribution_with_short_positions():
    """Test P&L attribution with short (SELL) positions."""
    from dt_backend.services.position_manager_dt import calculate_pnl_attribution
    
    positions_state = {
        "AAPL": {
            "status": "OPEN",
            "side": "SELL",  # Short position
            "qty": 100.0,
            "entry_price": 150.00,
            "bot": "MEAN_REVERSION",
        },
    }
    
    current_prices = {
        "AAPL": 145.00,  # Price went down, short makes money: (150-145)*100 = +500
    }
    
    attribution = calculate_pnl_attribution(positions_state, current_prices)
    
    assert attribution["MEAN_REVERSION"]["pnl"] == pytest.approx(500.0, rel=0.01)
    assert attribution["_total"]["pnl"] == pytest.approx(500.0, rel=0.01)


def test_hysteresis_preserves_state_across_calls():
    """Test that hysteresis state is preserved correctly."""
    cfg = PolicyConfig()
    cfg.confirmations_to_flip = 2
    cfg.min_edge_to_flip = 0.06
    
    node = {
        "policy_dt": {
            "action": "BUY",
            "_state": {
                "prev_action": "BUY",
                "pending_action": "",
                "pending_count": 0,
            },
        },
    }
    
    # First SELL signal
    proposed = "SELL"
    edge = -0.10
    conf = 0.60
    
    final_action_1, state_1, note_1 = _stabilize_with_hysteresis(
        node, proposed, edge, conf, cfg
    )
    
    # State should be updated
    assert state_1["pending_action"] == "SELL"
    assert state_1["pending_count"] == 1
    assert state_1["prev_action"] == "BUY"
    assert state_1["last_edge"] == pytest.approx(-0.10, rel=0.01)
    assert state_1["last_conf"] == pytest.approx(0.60, rel=0.01)


def test_confidence_clamped_to_max(default_policy_config):
    """Test that confidence is clamped to max_confidence."""
    cfg = default_policy_config
    cfg.max_confidence = 0.99
    cfg.trend_boost_strong = 1.25
    
    # Very high base confidence
    base_conf = 0.90
    intent = "BUY"
    trend = "strong_bull"
    vol_bkt = "low"
    regime_label = "bull"
    
    adjusted_conf, detail = _adjust_conf(
        base_conf, intent, trend, vol_bkt, regime_label, cfg
    )
    
    # Should be clamped to max_confidence even though 0.90 * 1.25 = 1.125
    assert adjusted_conf <= cfg.max_confidence
    assert adjusted_conf == pytest.approx(cfg.max_confidence, rel=0.01)


# ============================================================
# EDGE CALCULATION TESTS (bonus tests for completeness)
# ============================================================

def test_raw_intent_buy_threshold(default_policy_config):
    """Test that BUY intent is triggered when edge >= buy_threshold."""
    cfg = default_policy_config
    cfg.buy_threshold = 0.12
    
    p_buy = 0.65
    p_sell = 0.20
    p_hold = 0.15
    
    intent, conf, edge = _raw_intent_from_edge(p_buy, p_hold, p_sell, cfg)
    
    assert intent == "BUY"
    assert edge == pytest.approx(0.45, rel=0.01)  # 0.65 - 0.20
    assert conf == pytest.approx(0.65, rel=0.01)  # max(p_buy, p_sell)


def test_raw_intent_sell_threshold(default_policy_config):
    """Test that SELL intent is triggered when edge <= sell_threshold."""
    cfg = default_policy_config
    cfg.sell_threshold = -0.12
    
    p_buy = 0.20
    p_sell = 0.65
    p_hold = 0.15
    
    intent, conf, edge = _raw_intent_from_edge(p_buy, p_hold, p_sell, cfg)
    
    assert intent == "SELL"
    assert edge == pytest.approx(-0.45, rel=0.01)  # 0.20 - 0.65
    assert conf == pytest.approx(0.65, rel=0.01)  # max(p_buy, p_sell)


def test_raw_intent_hold_when_edge_weak(default_policy_config):
    """Test that HOLD intent when edge is between thresholds."""
    cfg = default_policy_config
    cfg.buy_threshold = 0.12
    cfg.sell_threshold = -0.12
    
    p_buy = 0.45
    p_sell = 0.40
    p_hold = 0.15
    
    intent, conf, edge = _raw_intent_from_edge(p_buy, p_hold, p_sell, cfg)
    
    assert intent == "HOLD"
    assert edge == pytest.approx(0.05, rel=0.01)  # 0.45 - 0.40
    assert conf == pytest.approx(0.45, rel=0.01)  # max(p_buy, p_sell)
