# dt_backend/core/execution_dt.py — Advanced execution intent layer.
"""
Converts policy_dt (intents) into execution_dt blocks that downstream
executors (e.g. backend trade bots) can consume.

    rolling[sym]["execution_dt"] = {
        "side": "BUY" | "SELL" | "FLAT",
        "size": 0.0–1.0,          # fraction of max capital per symbol
        "confidence_adj": 0.0–1.0,
        "cooldown": bool,
        "valid_until": <ISO8601 UTC>,
        "ts": <ISO8601 UTC>
    }
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple

from .data_pipeline_dt import _read_rolling, save_rolling, ensure_symbol_node, log


class ExecConfig:
    """Tunable knobs for execution behavior."""

    # Maximum notional fraction allocated per symbol for a *full* conviction signal.
    max_symbol_fraction: float = 0.15

    # Minimum confidence required to allocate anything.
    min_conf: float = 0.25

    # Phase 7: minimum calibrated P(hit) required to size above zero.
    min_phit: float = 0.52

    # Hard cap on adjusted confidence (after volatility / regime).
    max_conf_cap: float = 0.95

    # Cooldown window to avoid rapid flips between BUY and SELL.
    cooldown_minutes: int = 10

    # Base validity window for an execution intent.
    valid_minutes: int = 15


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


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
    if conf <= 0.0:
        return 0.0

    conf = max(0.0, min(cfg.max_conf_cap, conf))

    # Volatility-aware scaling: risk-off when vol is high.
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
) -> float:
    """Phase 7 sizing: size ~ f(P(hit), expected_R) with vol scaling.

    * expected_r is reward-to-risk multiple (e.g., 1.0 = 1R target).
    * This is intentionally conservative; we only allocate when phit is above
      coin-flip + margin.
    """
    phit = _safe_float(phit, 0.0)
    if phit < cfg.min_phit:
        return 0.0

    # Convert probability edge into [0,1].
    edge = max(0.0, min(1.0, (phit - 0.5) / 0.5))

    # expected_r factor: reward-to-risk helps, but don't let it explode.
    er = max(0.5, min(2.0, _safe_float(expected_r, 1.0)))
    r_factor = 0.5 + 0.5 * (er / 2.0)  # 0.5..1.0

    # Volatility-aware scaling: risk-off when vol is high.
    if vol_bkt == "high":
        vol_scale = 0.4
    elif vol_bkt == "medium":
        vol_scale = 0.7
    else:
        vol_scale = 1.0

    raw_size = cfg.max_symbol_fraction * edge * r_factor * vol_scale
    return max(0.0, min(cfg.max_symbol_fraction, raw_size))


def _expected_r_from_plan(node: Dict[str, Any]) -> float:
    """Infer reward-to-risk multiple from execution_plan_dt + last_price."""
    try:
        plan = node.get("execution_plan_dt") if isinstance(node.get("execution_plan_dt"), dict) else {}
        risk = plan.get("risk") if isinstance(plan, dict) and isinstance(plan.get("risk"), dict) else {}
        # prefer explicit r_target if provided by the strategy
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


def _parse_iso_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _cooldown_active(prev_exec: Dict[str, Any], new_side: str, cfg: ExecConfig, now: datetime | None = None) -> bool:
    """Return True if we are within cooldown window from a conflicting action."""
    if not prev_exec:
        return False

    prev_side = str(prev_exec.get("side") or "").upper()
    if prev_side not in {"BUY", "SELL"} or new_side not in {"BUY", "SELL"}:
        return False

    # Only enforce cooldown when flipping direction (BUY → SELL or SELL → BUY)
    if prev_side == new_side:
        return False

    ts = _parse_iso_ts(prev_exec.get("ts"))
    if ts is None:
        return False

    now = now_utc or datetime.now(timezone.utc)
    delta = now - ts
    return delta.total_seconds() < cfg.cooldown_minutes * 60


def run_execution_intraday(
    cfg: ExecConfig | None = None,
    *,
    now_utc: datetime | None = None,
    rolling_override: Dict[str, Any] | None = None,
    save: bool = True,
    policy_key: str = "policy_dt",
    out_key: str = "execution_dt",
) -> Dict[str, Any]:
    """Main entry point.

    Reads rolling, converts policy_dt into execution_dt, writes back.
    Returns summary dict.
    """
    cfg = cfg or ExecConfig()
    rolling = rolling_override if isinstance(rolling_override, dict) else _read_rolling()
    if not rolling:
        log("[exec_dt] ⚠️ rolling empty, nothing to do.")
        return {"symbols": 0, "updated": 0}

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

    now = now or datetime.now(timezone.utc)
    updated = 0

    for sym, node_raw in list(rolling.items()):
        if sym.startswith("_"):
            continue

        node = ensure_symbol_node(rolling, sym)
        policy = node.get(policy_key) or {}
        ctx = node.get("context_dt") or {}

        # Phase 3: strategy bots can publish an explicit trade plan.
        # If present, prefer it over pure policy intent sizing.
        plan = node.get("execution_plan_dt")
        if isinstance(plan, dict):
            try:
                p_side = str(plan.get("side") or "").upper()
                p_conf = _safe_float(plan.get("confidence"), 0.0)
                if p_side in {"BUY", "SELL"} and p_conf >= cfg.min_conf:
                    # Adopt plan as our intent baseline.
                    policy = {
                        **(policy if isinstance(policy, dict) else {}),
                        "intent": p_side,
                        "confidence": float(min(cfg.max_conf_cap, p_conf)),
                    }
                    node["_execution_plan_used"] = True
                else:
                    node["_execution_plan_used"] = False
            except Exception:
                node["_execution_plan_used"] = False

        intent = str(policy.get("intent") or "").upper()
        conf = _safe_float(policy.get("confidence"), 0.0)
        # Phase 7: calibrated probability-of-hit (falls back to confidence)
        phit = _safe_float(policy.get("p_hit"), conf)

        if intent not in {"BUY", "SELL"} or conf < cfg.min_conf:
            side = "FLAT"
            conf_adj = 0.0
        else:
            trend, vol_bkt = _trend_and_vol(ctx)
            expected_r = _expected_r_from_plan(node)
            size = _size_from_phit_expected_r(phit, expected_r, vol_bkt, cfg)

            # If size is effectively zero, treat as FLAT.
            if size <= 0.0:
                side = "FLAT"
                conf_adj = 0.0
            else:
                side = intent
                # Phase 7: confidence_adj reflects *probability of hit* when available.
                conf_adj = min(cfg.max_conf_cap, max(conf, phit))

        # Determine cooldown and adjust side/size accordingly.
        prev_exec = node.get(out_key) or {}
        cooldown = _cooldown_active(prev_exec, side, cfg, now=now)

        if cooldown and side in {"BUY", "SELL"}:
            # Respect cooldown by downgrading to FLAT for this cycle.
            side = "FLAT"

        # Compute final size if side is active; zero otherwise.
        if side in {"BUY", "SELL"}:
            trend, vol_bkt = _trend_and_vol(ctx)
            expected_r = _expected_r_from_plan(node)
            size = _size_from_phit_expected_r(phit, expected_r, vol_bkt, cfg)
            conf_adj = min(cfg.max_conf_cap, max(conf, phit))
        else:
            size = 0.0

        valid_until = (now + timedelta(minutes=cfg.valid_minutes)).isoformat()

        node[out_key] = {
            "side": side,
            "size": float(size),
            "confidence_adj": float(conf_adj),
            "p_hit": float(phit) if side in {"BUY", "SELL"} else 0.0,
            "expected_r": float(_expected_r_from_plan(node)) if side in {"BUY", "SELL"} else 0.0,
            "cooldown": bool(cooldown),
            "valid_until": valid_until,
            "ts": now.isoformat(),
            # Phase 3 metadata (ignored by Phase 0 executor, but logged/useful)
            "bot": (plan.get("bot") if isinstance(plan, dict) else None),
            "risk": (plan.get("risk") if isinstance(plan, dict) else None),
        }

        rolling[sym] = node
        updated += 1

    if save:
        save_rolling(rolling)
    log(f"[exec_dt] ✅ updated {out_key} for {updated} symbols.")
    return {"symbols": len(rolling), "updated": updated}


def main() -> None:
    run_execution_intraday()


if __name__ == "__main__":
    main()
