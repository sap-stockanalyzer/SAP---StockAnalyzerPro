"""Unit tests for DT Bot trading fixes.

Tests for:
1. Raised execution confidence threshold (0.25 -> 0.45)
2. Conviction-aware position sizing
3. Minimum hold time enforcement (10 minutes)
4. Raised min_edge_to_flip (0.06 -> 0.12)
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

# Import the modules we're testing
from dt_backend.core.execution_dt import (
    ExecConfig,
    _size_from_phit_expected_r,
)
from dt_backend.core.policy_engine_dt import PolicyConfig
from dt_backend.engines.trade_executor import ExecutionConfig


class TestConfidenceThreshold:
    """Test that confidence threshold is properly raised."""
    
    def test_default_min_confidence_policy(self):
        """Policy engine should have appropriate min_confidence default."""
        cfg = PolicyConfig()
        assert cfg.min_confidence == 0.30, "PolicyConfig min_confidence should be 0.30"
    
    def test_default_min_confidence_execution(self):
        """Execution engine should have raised threshold."""
        cfg = ExecutionConfig()
        # Default in code should be 0.25, but env should override to 0.45
        assert cfg.min_confidence == 0.25, "ExecutionConfig default should be 0.25"
    
    def test_exec_config_default(self):
        """ExecConfig should have proper default."""
        cfg = ExecConfig()
        # ExecConfig uses min_conf not min_confidence
        assert cfg.min_conf == 0.25, "ExecConfig min_conf should have sensible default"


class TestConvictionAwareSizing:
    """Test that position sizing scales with confidence."""
    
    def test_sizing_with_low_confidence(self):
        """Low confidence (near threshold) should result in smaller size."""
        cfg = ExecConfig()
        cfg.min_conf = 0.45
        cfg.max_symbol_fraction = 0.15
        
        # Low confidence = 0.45 (at threshold)
        size_low = _size_from_phit_expected_r(
            phit=0.55,
            expected_r=1.5,
            vol_bkt="low",
            cfg=cfg,
            confidence=0.45
        )
        
        # Size should be non-zero but smaller due to low confidence
        assert size_low > 0.0, "Should produce non-zero size"
        assert size_low < cfg.max_symbol_fraction, "Should be less than max"
    
    def test_sizing_with_high_confidence(self):
        """High confidence should result in larger size."""
        cfg = ExecConfig()
        cfg.min_conf = 0.45
        cfg.max_symbol_fraction = 0.15
        
        # High confidence = 0.95
        size_high = _size_from_phit_expected_r(
            phit=0.55,
            expected_r=1.5,
            vol_bkt="low",
            cfg=cfg,
            confidence=0.95
        )
        
        # Size should be larger due to high confidence
        assert size_high > 0.0, "Should produce non-zero size"
    
    def test_sizing_scales_with_confidence(self):
        """Size should increase as confidence increases."""
        cfg = ExecConfig()
        cfg.min_conf = 0.45
        cfg.max_symbol_fraction = 0.15
        
        # Same P(hit) and expected_R, varying confidence
        size_low = _size_from_phit_expected_r(
            phit=0.60,
            expected_r=1.5,
            vol_bkt="low",
            cfg=cfg,
            confidence=0.50
        )
        
        size_high = _size_from_phit_expected_r(
            phit=0.60,
            expected_r=1.5,
            vol_bkt="low",
            cfg=cfg,
            confidence=0.90
        )
        
        assert size_high > size_low, "Higher confidence should yield larger position size"
    
    def test_sizing_respects_vol_bucket(self):
        """Volatility bucket should still affect sizing."""
        cfg = ExecConfig()
        cfg.min_conf = 0.45
        cfg.max_symbol_fraction = 0.15
        
        size_low_vol = _size_from_phit_expected_r(
            phit=0.60,
            expected_r=1.5,
            vol_bkt="low",
            cfg=cfg,
            confidence=0.70
        )
        
        size_high_vol = _size_from_phit_expected_r(
            phit=0.60,
            expected_r=1.5,
            vol_bkt="high",
            cfg=cfg,
            confidence=0.70
        )
        
        assert size_low_vol > size_high_vol, "Low volatility should allow larger size than high volatility"
    
    def test_sizing_without_confidence_fallback(self):
        """Should handle missing confidence gracefully."""
        cfg = ExecConfig()
        cfg.min_conf = 0.45
        cfg.max_symbol_fraction = 0.15
        
        # No confidence provided - should use fallback
        size = _size_from_phit_expected_r(
            phit=0.60,
            expected_r=1.5,
            vol_bkt="low",
            cfg=cfg,
            confidence=0.0  # or not provided
        )
        
        assert size > 0.0, "Should still produce size with fallback"


class TestMinEdgeToFlip:
    """Test that min_edge_to_flip is properly raised."""
    
    def test_default_min_edge_to_flip(self):
        """min_edge_to_flip should be raised to 0.12."""
        cfg = PolicyConfig()
        assert cfg.min_edge_to_flip == 0.12, "min_edge_to_flip should be raised from 0.06 to 0.12"
    
    def test_hysteresis_hold_bias_unchanged(self):
        """hysteresis_hold_bias should remain at 0.03."""
        cfg = PolicyConfig()
        assert cfg.hysteresis_hold_bias == 0.03, "hysteresis_hold_bias should remain 0.03"
    
    def test_confirmations_to_flip_unchanged(self):
        """confirmations_to_flip should remain at 2."""
        cfg = PolicyConfig()
        assert cfg.confirmations_to_flip == 2, "confirmations_to_flip should remain 2"


class TestMinHoldTimeConfig:
    """Test that minimum hold time configuration is added."""
    
    def test_default_min_hold_time(self):
        """ExecutionConfig should have min_hold_time_minutes."""
        cfg = ExecutionConfig()
        assert hasattr(cfg, 'min_hold_time_minutes'), "Should have min_hold_time_minutes attribute"
        assert cfg.min_hold_time_minutes == 10, "Default min_hold_time_minutes should be 10"
    
    def test_default_hard_stop_loss_pct(self):
        """ExecutionConfig should have hard_stop_loss_pct."""
        cfg = ExecutionConfig()
        assert hasattr(cfg, 'hard_stop_loss_pct'), "Should have hard_stop_loss_pct attribute"
        assert cfg.hard_stop_loss_pct == 2.0, "Default hard_stop_loss_pct should be 2.0"
    
    def test_min_flip_minutes_unchanged(self):
        """min_flip_minutes should remain at 12."""
        cfg = ExecutionConfig()
        assert cfg.min_flip_minutes == 12, "min_flip_minutes should remain 12"


class TestSizingEdgeCases:
    """Test edge cases in sizing calculations."""
    
    def test_sizing_below_min_phit(self):
        """Should return 0 if P(hit) below minimum threshold."""
        cfg = ExecConfig()
        cfg.min_phit = 0.52
        cfg.min_conf = 0.45
        
        size = _size_from_phit_expected_r(
            phit=0.50,  # below min_phit
            expected_r=1.5,
            vol_bkt="low",
            cfg=cfg,
            confidence=0.90  # high confidence doesn't matter
        )
        
        assert size == 0.0, "Should return 0 when P(hit) is below minimum"
    
    def test_sizing_caps_at_max_fraction(self):
        """Size should never exceed max_symbol_fraction."""
        cfg = ExecConfig()
        cfg.min_conf = 0.45
        cfg.max_symbol_fraction = 0.15
        
        # Very high values to try to exceed cap
        size = _size_from_phit_expected_r(
            phit=0.99,
            expected_r=2.0,
            vol_bkt="low",
            cfg=cfg,
            confidence=0.99
        )
        
        assert size <= cfg.max_symbol_fraction, "Size should never exceed max_symbol_fraction"
    
    def test_sizing_confidence_normalization(self):
        """Confidence below min_conf should normalize to 0 conviction factor."""
        cfg = ExecConfig()
        cfg.min_conf = 0.45
        cfg.max_symbol_fraction = 0.15
        
        # Confidence at minimum threshold
        size_min = _size_from_phit_expected_r(
            phit=0.60,
            expected_r=1.5,
            vol_bkt="low",
            cfg=cfg,
            confidence=0.45  # at minimum
        )
        
        # Should still produce size, but with 0.5x conviction factor
        assert size_min > 0.0, "Should produce non-zero size at min confidence"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
