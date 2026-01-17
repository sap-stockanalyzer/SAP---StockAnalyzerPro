"""Shared utility functions for day trading backend.

Common helpers used across multiple modules to avoid duplication.
"""

from typing import Any, Dict, List


def safe_float(x: Any, default: float = 0.0) -> float:
    """Safely convert value to float with fallback."""
    try:
        import math
        v = float(x)
        if not math.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def sort_by_ranking_metric(symbols: List[str], rolling: Dict[str, Any]) -> List[str]:
    """Sort symbols by signal strength + confidence (descending), NOT alphabetically.
    
    Human day traders prioritize by signal strength + volume + confidence.
    Order matters: limited slots should go to highest-conviction setups.
    
    DO NOT sort alphabetically - causes "A" ticker bias where AAPL/AMD always
    get priority over higher-quality setups with later alphabetical tickers.
    
    Args:
        symbols: List of symbol strings to sort
        rolling: Rolling data dict containing features_dt and policy_dt for each symbol
        
    Returns:
        List of symbols sorted by composite score (highest first)
    """
    if not symbols or not rolling:
        return symbols
    
    ranked = []
    for sym in symbols:
        node = rolling.get(sym, {})
        if not isinstance(node, dict):
            ranked.append((0.0, sym))
            continue
        
        # Primary: signal strength from features_dt (volume, liquidity, momentum)
        feats = node.get("features_dt", {})
        if not isinstance(feats, dict):
            feats = {}
        
        signal_strength = safe_float(feats.get("signal_strength"), 0.0)
        
        # Fallback hierarchy: volume -> liquidity_score -> rel_volume
        if signal_strength == 0.0:
            signal_strength = safe_float(feats.get("volume"), 0.0)
        if signal_strength == 0.0:
            signal_strength = safe_float(feats.get("liquidity_score"), 0.0)
        if signal_strength == 0.0:
            signal_strength = safe_float(feats.get("rel_volume"), 0.0)
        
        # Normalize large volume numbers to 0-1 range
        if signal_strength > 1.0:
            signal_strength = min(1.0, signal_strength / 1000000.0)
        
        # Secondary: confidence from policy_dt
        pol = node.get("policy_dt", {})
        if not isinstance(pol, dict):
            pol = {}
        confidence = safe_float(pol.get("confidence"), 0.0)
        
        # Tertiary: p_hit from policy_dt (calibrated probability)
        p_hit = safe_float(pol.get("p_hit"), confidence)
        
        # Composite score: 50% signal_strength + 30% confidence + 20% p_hit
        # This balances conviction with signal quality
        score = (0.5 * signal_strength) + (0.3 * confidence) + (0.2 * p_hit)
        
        ranked.append((score, sym))
    
    # Sort descending (highest score first) - top setups get priority for limited slots
    ranked.sort(reverse=True, key=lambda x: x[0])
    return [sym for _, sym in ranked]
