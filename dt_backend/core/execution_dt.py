# dt_backend/core/execution_dt.py — v1.1 (LANE-AWARE + STAND_DOWN SAFE)
"""
Advanced execution intent layer for dt_backend.

Converts policy_dt (intents) into execution_dt blocks that downstream
executors (e.g. backend trade bots) can consume.

Writes:
    rolling[sym]["execution_dt"] = {
        "side": "BUY" | "SELL" | "FLAT",
        "size": 0.0–1.0,          # fraction of max capital per symbol
        "confidence_adj": 0.0–1.0,
        "cooldown": bool,
        "valid_until": <ISO8601 UTC>,
        "ts": <ISO8601 UTC>,

        # Optional Phase 7/3 extras (safe for downstream to ignore)
        "p_hit": float,
        "expected_r": float,
        "bot": str|None,
        "risk": dict|None,
    }

v1.1 additions
--------------
- Lane-aware: accepts optional `symbols=[...]` and `max_symbols=...` so the
  fast-lane / slow-lane orchestrator can scope work without touching other nodes.
- STAND_DOWN handling: if policy action/intent is STAND_DOWN, we force FLAT.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from dt_backend.tuning.dt_profile_loader import load_dt_profile

from .data_pipeline_dt import _read_rolling, save_rolling, ensure_symbol_node, log
from dt_backend.utils.trading_utils_dt import sort_by_ranking_metric
from dt_backend.core.constants_dt import (
    POSITION_MAX_FRACTION,
    POSITION_PROBE_FRACTION,
    POSITION_PRESS_MULT,
    PHIT_MIN,
    PHIT_PRESS_MIN,
    CONFIDENCE_MIN_PROBE,
    CONFIDENCE_MIN_EXEC,
    CONFIDENCE_MAX,
    COOLDOWN_AFTER_BUY_MINUTES,
)

# Import broker API for position sync (late import to avoid circular deps)
try:
    from dt_backend.engines.broker_api import get_positions
except ImportError:
    get_positions = None  # type: ignore


class ExecConfig:
    """Tunable knobs for execution behavior."""

    # Maximum notional fraction allocated per symbol for a *full* conviction signal.
    max_symbol_fraction: float = POSITION_MAX_FRACTION

    # Minimum confidence required to allocate anything.
    min_conf: float = CONFIDENCE_MIN_EXEC

    # Phase 7: minimum calibrated P(hit) required to size above zero.
    min_phit: float = PHIT_MIN

    # Hard cap on adjusted confidence (after volatility / regime).
    max_conf_cap: float = CONFIDENCE_MAX

    # Cooldown window to avoid rapid flips between BUY and SELL.
    cooldown_minutes: int = COOLDOWN_AFTER_BUY_MINUTES

    # Base validity window for an execution intent.
    valid_minutes: int = 15

    # -----------------------------
    # Phase 4: action tiers
    # -----------------------------
    # PROBE: allow lower-confidence, micro-sized entries.
    probe_min_conf: float = CONFIDENCE_MIN_PROBE
    probe_size_fraction: float = POSITION_PROBE_FRACTION

    # PRESS: modest scale-up when P(hit) is strong.
    press_min_phit: float = PHIT_PRESS_MIN
    press_size_mult: float = POSITION_PRESS_MULT


def _env_float(name: str, default: float) -> float:
    try:
        import os
        raw = (os.getenv(name, "") or "").strip()
        return float(raw) if raw else float(default)
    except Exception:
        return float(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default

def _get_regime_label_dt(rolling: Dict[str, Any]) -> str:
    try:
        g = rolling.get("_GLOBAL_DT") or {}
        if not isinstance(g, dict):
            return "unknown"
        reg = g.get("regime_dt")
        if isinstance(reg, dict) and reg.get("label"):
            return str(reg.get("label"))
        reg2 = g.get("regime")
        if isinstance(reg2, dict) and reg2.get("label"):
            return str(reg2.get("label"))
        return str(g.get("regime_label") or "unknown")
    except Exception:
        return "unknown"


def _trend_and_vol(context_dt: Dict[str, Any]) -> Tuple[str, str]:
    trend = str(context_dt.get("intraday_trend") or "").strip()
    vol_bkt = str(context_dt.get("vol_bucket") or "").strip()

    if not trend:
        r = _safe_float(context_dt.get("intraday_return"), 0.0)
        if r >= 0.01:
            trend = "strong_bull"
        elif r >= 0.003:
            trend = "bull"
        elif r <= -0.01:
            trend = "strong_bear"
        elif r <= -0.003:
            trend = "bear"
        else:
            trend = "flat"

    if not vol_bkt:
        vol = _safe_float(context_dt.get("intraday_vol"), 0.0)
        if vol >= 0.02:
            vol_bkt = "high"
        elif vol >= 0.007:
            vol_bkt = "medium"
        else:
            vol_bkt = "low"

    return trend, vol_bkt


def _size_from_conf_and_vol(conf: float, vol_bkt: str, cfg: ExecConfig) -> float:
    """Legacy sizing (kept for compatibility / future fallback)."""
    if conf <= 0.0:
        return 0.0

    conf = max(0.0, min(cfg.max_conf_cap, conf))

    if vol_bkt == "high":
        scale = 0.4
    elif vol_bkt == "medium":
        scale = 0.7
    else:
        scale = 1.0

    raw_size = conf * cfg.max_symbol_fraction * scale
    return max(0.0, min(cfg.max_symbol_fraction, raw_size))


def _size_from_phit_expected_r(
    phit: float,
    expected_r: float,
    vol_bkt: str,
    cfg: ExecConfig,
    confidence: float = 0.0,
) -> float:
    """Phase 7 sizing: size ~ f(P(hit), expected_R, confidence) with vol scaling.
    
    Enhanced to incorporate confidence directly for conviction-aware sizing.
    Higher confidence -> larger position size.
    """
    phit = _safe_float(phit, 0.0)
    if phit < cfg.min_phit:
        return 0.0

    edge = max(0.0, min(1.0, (phit - 0.5) / 0.5))

    er = max(0.5, min(2.0, _safe_float(expected_r, 1.0)))
    r_factor = 0.5 + 0.5 * (er / 2.0)  # 0.5..1.0
    
    # Conviction factor: scale size by confidence
    # confidence ranges from min_conf (0.45) to 1.0
    # Map to scaling factor 0.5 to 1.5 for conviction-aware sizing
    confidence = _safe_float(confidence, 0.0)
    if confidence > 0.0 and cfg.min_conf < 1.0:  # Guard against division by zero
        # Normalize confidence to 0..1 range based on min threshold
        conf_normalized = max(0.0, min(1.0, (confidence - cfg.min_conf) / (1.0 - cfg.min_conf)))
        # Scale from 0.5x to 1.5x based on conviction
        conviction_factor = 0.5 + conf_normalized
    else:
        conviction_factor = 1.0  # fallback if confidence not provided or min_conf >= 1.0

    if vol_bkt == "high":
        vol_scale = 0.4
    elif vol_bkt == "medium":
        vol_scale = 0.7
    else:
        vol_scale = 1.0

    raw_size = cfg.max_symbol_fraction * edge * r_factor * vol_scale * conviction_factor
    return max(0.0, min(cfg.max_symbol_fraction, raw_size))


def _expected_r_from_plan(node: Dict[str, Any]) -> float:
    """Infer reward-to-risk multiple from execution_plan_dt + last_price."""
    try:
        plan = node.get("execution_plan_dt") if isinstance(node.get("execution_plan_dt"), dict) else {}
        risk = plan.get("risk") if isinstance(plan, dict) and isinstance(plan.get("risk"), dict) else {}

        rt = risk.get("r_target")
        if rt is not None:
            return max(0.1, min(3.0, _safe_float(rt, 1.0)))

        feats = node.get("features_dt") if isinstance(node.get("features_dt"), dict) else {}
        last_px = _safe_float(feats.get("last_price"), 0.0)
        stop = _safe_float(risk.get("stop"), 0.0)
        tp = _safe_float(risk.get("take_profit"), 0.0)
        if last_px <= 0 or stop <= 0 or tp <= 0:
            return 1.0

        rr = abs(tp - last_px) / max(1e-9, abs(last_px - stop))
        return max(0.1, min(3.0, float(rr)))
    except Exception:
        return 1.0


def _parse_iso_ts(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = str(ts).strip()
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _cooldown_active(
    prev_exec: Dict[str, Any],
    new_side: str,
    cfg: ExecConfig,
    *,
    now_utc: Optional[datetime] = None,
) -> bool:
    """Return True if we are within cooldown window from a conflicting action."""
    if not prev_exec:
        return False

    prev_side = str(prev_exec.get("side") or "").upper()
    new_side = str(new_side or "").upper()

    if prev_side not in {"BUY", "SELL"} or new_side not in {"BUY", "SELL"}:
        return False
    if prev_side == new_side:
        return False

    ts = _parse_iso_ts(prev_exec.get("ts"))
    if ts is None:
        return False

    now_utc = now_utc or datetime.now(timezone.utc)
    delta = now_utc - ts
    return delta.total_seconds() < cfg.cooldown_minutes * 60


def _attach_positions_to_rolling(rolling: Dict[str, Any], *, now_utc: Optional[datetime] = None) -> int:
    """Read broker positions and attach them to rolling nodes as position_dt.
    
    This enables the policy engine's _has_position() checks to see current positions
    and avoid redundant entry orders.
    
    Returns the number of positions attached.
    """
    if get_positions is None:
        return 0
    
    try:
        positions = get_positions()
        if not isinstance(positions, dict):
            return 0
        
        now_utc = now_utc or datetime.now(timezone.utc)
        ts = now_utc.isoformat().replace("+00:00", "Z")
        
        attached = 0
        for sym, pos in positions.items():
            sym = str(sym).strip().upper()
            if not sym or sym.startswith("_"):
                continue
            
            try:
                qty = float(getattr(pos, "qty", 0.0))
                avg_price = float(getattr(pos, "avg_price", 0.0))
                
                # Only attach non-zero positions
                if qty == 0.0:
                    continue
                
                # Determine side
                if qty > 0:
                    side = "LONG"
                elif qty < 0:
                    side = "SHORT"
                else:
                    side = "FLAT"
                
                # Get or create node
                node = ensure_symbol_node(rolling, sym)
                
                # Write position_dt
                node["position_dt"] = {
                    "qty": float(qty),
                    "avg_price": float(avg_price),
                    "side": side,
                    "ts": ts,
                }
                
                rolling[sym] = node
                attached += 1
                
            except Exception:
                continue
        
        return attached
        
    except Exception:
        return 0


def _symbols_to_process(
    rolling: Dict[str, Any],
    *,
    symbols: Optional[List[str]] = None,
    max_symbols: Optional[int] = None,
) -> List[str]:
    """Deterministic symbol selection for lane-aware processing.
    
    CRITICAL: Symbols are sorted by signal strength + confidence, NOT alphabetically.
    Alphabetical sorting causes "A" ticker bias where AAPL/AMD always get priority.
    Human day traders prioritize highest-conviction setups, not alphabet order.
    """
    if isinstance(symbols, list) and symbols:
        wanted = {str(s).strip().upper() for s in symbols if str(s).strip()}
        syms = [str(s).upper() for s in rolling.keys() if isinstance(s, str) and not s.startswith("_") and str(s).upper() in wanted]
    else:
        syms = [str(s) for s in rolling.keys() if isinstance(s, str) and not s.startswith("_")]

    # DO NOT sort alphabetically - use signal-based ranking instead
    syms = list(set(syms))
    syms = sort_by_ranking_metric(syms, rolling)
    
    if max_symbols is not None:
        try:
            syms = syms[: max(0, int(max_symbols))]
        except Exception:
            pass
    return syms


def run_execution_intraday(
    cfg: ExecConfig | None = None,
    *,
    now_utc: datetime | None = None,
    rolling_override: Dict[str, Any] | None = None,
    save: bool = True,
    policy_key: str = "policy_dt",
    out_key: str = "execution_dt",
    symbols: Optional[List[str]] = None,
    max_symbols: Optional[int] = None,
) -> Dict[str, Any]:
    """Main entry point.

    Reads rolling, converts policy_dt into execution_dt, writes back.

    Args:
        symbols: optional explicit universe (fast/slow lane). If provided, only
                 those symbols are updated.
        max_symbols: optional cap (applies after symbol filtering).

    Returns summary dict.
    """
    cfg = cfg or ExecConfig()

    # Optional env overrides (keeps knobs centralized in dt_knobs.env)
    cfg.probe_min_conf = _env_float("DT_PROBE_MIN_CONF", cfg.probe_min_conf)
    cfg.probe_size_fraction = _env_float("DT_PROBE_SIZE_FRAC", cfg.probe_size_fraction)
    cfg.press_min_phit = _env_float("DT_PRESS_MIN_PHIT", cfg.press_min_phit)
    cfg.press_size_mult = _env_float("DT_PRESS_SIZE_MULT", cfg.press_size_mult)

    # sane clamps
    cfg.probe_min_conf = max(0.0, min(0.60, float(cfg.probe_min_conf)))
    cfg.probe_size_fraction = max(0.0, min(1.0, float(cfg.probe_size_fraction)))
    cfg.press_min_phit = max(0.50, min(0.90, float(cfg.press_min_phit)))
    cfg.press_size_mult = max(1.0, min(2.50, float(cfg.press_size_mult)))
    rolling = rolling_override if isinstance(rolling_override, dict) else _read_rolling()
    # Phase 5: playbook profile sizing overrides (soft sizing knobs only)
    try:
        reg_label = _get_regime_label_dt(rolling if isinstance(rolling, dict) else {})
        prof = load_dt_profile(reg_label)
        p = prof.get("probe") if isinstance(prof, dict) else None
        pr = prof.get("press") if isinstance(prof, dict) else None
        if isinstance(p, dict) and "size_frac" in p:
            cfg.probe_size_fraction = float(p.get("size_frac"))
        if isinstance(p, dict) and "min_conf" in p:
            cfg.probe_min_conf = float(p.get("min_conf"))
        if isinstance(pr, dict) and "min_phit" in pr:
            cfg.press_min_phit = float(pr.get("min_phit"))
        if isinstance(pr, dict) and "size_mult" in pr:
            cfg.press_size_mult = float(pr.get("size_mult"))
    except Exception:
        pass

    if not rolling or not isinstance(rolling, dict):
        log("[exec_dt] ⚠️ rolling empty, nothing to do.")
        return {"symbols": 0, "updated": 0}

    now_utc = now_utc or datetime.now(timezone.utc)

    # Attach broker positions to rolling cache so policy can see current holdings
    try:
        pos_attached = _attach_positions_to_rolling(rolling, now_utc=now_utc)
        if pos_attached > 0:
            log(f"[exec_dt] 📌 attached {pos_attached} broker positions to rolling cache")
    except Exception as e:
        log(f"[exec_dt] ⚠️ failed to attach positions: {e}")

    # Phase 4: risk mode can modestly scale sizing (keeps a single code path).
    try:
        g = rolling.get("_GLOBAL_DT") or {}
        plan = g.get("daily_plan_dt") if isinstance(g, dict) else None
        risk_mode = str((plan or {}).get("risk_mode") or "").upper() if isinstance(plan, dict) else ""
        if risk_mode == "CONSERVATIVE":
            cfg.max_symbol_fraction = min(cfg.max_symbol_fraction, 0.10)
            cfg.min_conf = max(cfg.min_conf, 0.28)
        elif risk_mode == "AGGRESSIVE":
            cfg.max_symbol_fraction = max(cfg.max_symbol_fraction, 0.18)
            cfg.min_conf = min(cfg.min_conf, 0.22)
    except Exception:
        pass

    syms = _symbols_to_process(rolling, symbols=symbols, max_symbols=max_symbols)

    updated = 0
    for sym in syms:
        node_raw = rolling.get(sym)
        if not isinstance(node_raw, dict):
            continue

        node = ensure_symbol_node(rolling, sym)
        policy = node.get(policy_key) if isinstance(node.get(policy_key), dict) else {}
        ctx = node.get("context_dt") if isinstance(node.get("context_dt"), dict) else {}

        # Hard safety: STAND_DOWN => FLAT
        pol_action = str((policy or {}).get("action") or "").upper()
        if pol_action == "STAND_DOWN":
            node[out_key] = {
                "side": "FLAT",
                "size": 0.0,
                "confidence_adj": 0.0,
                "p_hit": 0.0,
                "expected_r": 0.0,
                "cooldown": False,
                "valid_until": (now_utc + timedelta(minutes=cfg.valid_minutes)).isoformat().replace("+00:00", "Z"),
                "ts": now_utc.isoformat().replace("+00:00", "Z"),
                "bot": None,
                "risk": None,
            }
            rolling[sym] = node
            updated += 1
            continue

        # Phase 3/4: strategy bots can publish an explicit trade plan (with optional tiers).
        plan = node.get("execution_plan_dt") if isinstance(node.get("execution_plan_dt"), dict) else None
        if isinstance(plan, dict):
            try:
                p_side = str(plan.get("side") or "").upper()
                p_conf = _safe_float(plan.get("confidence"), 0.0)
                p_tier = str(plan.get("tier") or "").upper()

                # PROBE tier can run with lower confidence.
                min_needed = cfg.probe_min_conf if p_tier == "PROBE" else cfg.min_conf

                if p_side in {"BUY", "SELL"} and p_conf >= float(min_needed):
                    policy = {
                        **(policy if isinstance(policy, dict) else {}),
                        "intent": p_side,
                        "confidence": float(min(cfg.max_conf_cap, p_conf)),
                        **({"tier": p_tier} if p_tier else {}),
                    }
                    node["_execution_plan_used"] = True
                else:
                    node["_execution_plan_used"] = False
            except Exception:
                node["_execution_plan_used"] = False

        tier = str((policy or {}).get("tier") or "").upper()

        # Phase 4: PROBE uses signal_* fields (keeps UI policy stable while letting execution take micro-risk).
        if tier == "PROBE":
            intent = str((policy or {}).get("signal_intent") or (policy or {}).get("intent") or "").upper()
            conf = _safe_float((policy or {}).get("signal_confidence"), _safe_float((policy or {}).get("confidence"), 0.0))
            phit = _safe_float((policy or {}).get("signal_p_hit"), _safe_float((policy or {}).get("p_hit"), conf))
        else:
            intent = str((policy or {}).get("intent") or "").upper()
            conf = _safe_float((policy or {}).get("confidence"), 0.0)
            # Phase 7: calibrated probability-of-hit (falls back to confidence)
            phit = _safe_float((policy or {}).get("p_hit"), conf)

        side = "FLAT"
        conf_adj = 0.0
        size = 0.0

        if intent in {"BUY", "SELL"}:
            _, vol_bkt = _trend_and_vol(ctx)
            expected_r = _expected_r_from_plan(node)

            if tier == "PROBE":
                if conf >= float(cfg.probe_min_conf):
                    size = _size_from_conf_and_vol(conf, vol_bkt, cfg) * float(cfg.probe_size_fraction)
                    size = max(0.0, min(cfg.max_symbol_fraction, float(size)))
            else:
                if conf >= float(cfg.min_conf):
                    size = _size_from_phit_expected_r(phit, expected_r, vol_bkt, cfg, confidence=conf)
                    if tier == "PRESS" and float(phit) >= float(cfg.press_min_phit):
                        size = max(0.0, min(cfg.max_symbol_fraction, float(size) * float(cfg.press_size_mult)))

            if size > 0.0:
                side = intent
                conf_adj = min(cfg.max_conf_cap, max(conf, phit))

        prev_exec = node.get(out_key) if isinstance(node.get(out_key), dict) else {}
        cooldown = _cooldown_active(prev_exec, side, cfg, now_utc=now_utc)

        if cooldown and side in {"BUY", "SELL"}:
            side = "FLAT"
            size = 0.0
            conf_adj = 0.0

        valid_until = (now_utc + timedelta(minutes=cfg.valid_minutes)).isoformat().replace("+00:00", "Z")

        node[out_key] = {
            "side": side,
            "size": float(size) if side in {"BUY", "SELL"} else 0.0,
            "confidence_adj": float(conf_adj) if side in {"BUY", "SELL"} else 0.0,
            "p_hit": float(phit) if side in {"BUY", "SELL"} else 0.0,
            "expected_r": float(_expected_r_from_plan(node)) if side in {"BUY", "SELL"} else 0.0,
            "cooldown": bool(cooldown),
            "valid_until": valid_until,
            "ts": now_utc.isoformat().replace("+00:00", "Z"),
            "bot": (plan.get("bot") if isinstance(plan, dict) else None),
            "risk": (plan.get("risk") if isinstance(plan, dict) else None),
        }

        rolling[sym] = node
        updated += 1

    if save:
        save_rolling(rolling)

    log(f"[exec_dt] ✅ updated {out_key} for {updated} symbols (seen={len(syms)}).")
    return {"symbols": len(syms), "updated": updated}


def main() -> None:
    run_execution_intraday()


if __name__ == "__main__":
    main()
