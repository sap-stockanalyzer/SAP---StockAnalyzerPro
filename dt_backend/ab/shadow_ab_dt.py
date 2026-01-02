# dt_backend/ab/shadow_ab_dt.py — Phase 6.5
"""Shadow mode + A/B comparison utilities.

Design goals
------------
* **Zero blast radius**: shadow never places orders.
* **Deterministic**: shadow writes into separate keys in rolling:
    - policy_dt_shadow
    - execution_dt_shadow
* **Comparable**: produce small summary stats and per-symbol diffs.

Shadow versions are configured via a JSON file.

Example override file
---------------------
{
  "version": "vNEXT",
  "policy": {
    "min_confidence": 0.30,
    "buy_threshold": 0.12
  },
  "exec": {
    "min_phit": 0.55,
    "max_symbol_fraction": 0.12
  },
  "env": {
    "DT_DISABLE_MODEL_FALLBACK": "1"
  }
}
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, Tuple

from dt_backend.core.policy_engine_dt import PolicyConfig, apply_intraday_policy
from dt_backend.core.execution_dt import ExecConfig, run_execution_intraday


def _deepish_copy_rolling(rolling: Dict[str, Any]) -> Dict[str, Any]:
    """Copy dict-of-dicts but keep large arrays shared (bars, etc.)."""
    if not isinstance(rolling, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in rolling.items():
        if isinstance(v, dict):
            out[k] = dict(v)
        else:
            out[k] = v
    return out


def load_shadow_override(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


class _Env:
    def __init__(self, overrides: Dict[str, str]):
        self.overrides = {str(k): str(v) for k, v in (overrides or {}).items()}
        self.prev: Dict[str, str | None] = {}

    def __enter__(self):
        for k, v in self.overrides.items():
            self.prev[k] = os.environ.get(k)
            os.environ[k] = v
        return self

    def __exit__(self, exc_type, exc, tb):
        for k, old in self.prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _apply_cfg_overrides(cfg: Any, overrides: Dict[str, Any]) -> Any:
    if not overrides:
        return cfg
    for k, v in overrides.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    return cfg


def run_shadow_ab(
    rolling_live: Dict[str, Any],
    *,
    override_path: str | None,
    shadow_policy_key: str = "policy_dt_shadow",
    shadow_exec_key: str = "execution_dt_shadow",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run shadow policy+execution on a copy of rolling and write results into live rolling.

    Returns (summary, per_symbol_diffs).
    """
    overrides = load_shadow_override(override_path)
    env_over = overrides.get("env") if isinstance(overrides.get("env"), dict) else {}
    pol_over = overrides.get("policy") if isinstance(overrides.get("policy"), dict) else {}
    exe_over = overrides.get("exec") if isinstance(overrides.get("exec"), dict) else {}
    version = str(overrides.get("version") or os.environ.get("DT_SHADOW_VERSION") or "shadow")

    rolling_shadow = _deepish_copy_rolling(rolling_live)

    pol_cfg = _apply_cfg_overrides(PolicyConfig(), pol_over)
    exe_cfg = _apply_cfg_overrides(ExecConfig(), exe_over)

    with _Env(env_over):
        apply_intraday_policy(pol_cfg, rolling_override=rolling_shadow, save=False, out_key=shadow_policy_key)
        run_execution_intraday(exe_cfg, rolling_override=rolling_shadow, save=False, policy_key=shadow_policy_key, out_key=shadow_exec_key)

    # Merge shadow results back into live rolling
    diffs: Dict[str, Any] = {}
    live_trades = 0
    shad_trades = 0
    agree = 0
    total = 0

    for sym, node in list(rolling_live.items()):
        if not isinstance(sym, str) or sym.startswith("_"):
            continue
        if not isinstance(node, dict):
            continue
        sh_node = rolling_shadow.get(sym)
        if not isinstance(sh_node, dict):
            continue

        lp = node.get("policy_dt") if isinstance(node.get("policy_dt"), dict) else {}
        sp = sh_node.get(shadow_policy_key) if isinstance(sh_node.get(shadow_policy_key), dict) else {}
        le = node.get("execution_dt") if isinstance(node.get("execution_dt"), dict) else {}
        se = sh_node.get(shadow_exec_key) if isinstance(sh_node.get(shadow_exec_key), dict) else {}

        node[shadow_policy_key] = sp
        node[shadow_exec_key] = se
        rolling_live[sym] = node

        la = str(lp.get("action") or lp.get("intent") or "HOLD").upper()
        sa = str(sp.get("action") or sp.get("intent") or "HOLD").upper()
        total += 1
        if la == sa:
            agree += 1
        if la in {"BUY", "SELL"} and bool(lp.get("trade_gate")):
            live_trades += 1
        if sa in {"BUY", "SELL"} and bool(sp.get("trade_gate")):
            shad_trades += 1
        if la != sa:
            diffs[sym] = {
                "live": {"action": la, "conf": lp.get("confidence"), "p_hit": lp.get("p_hit"), "bot": lp.get("bot")},
                "shadow": {"action": sa, "conf": sp.get("confidence"), "p_hit": sp.get("p_hit"), "bot": sp.get("bot")},
                "live_size": le.get("size"),
                "shadow_size": se.get("size"),
            }

    summary = {
        "shadow_version": version,
        "symbols": total,
        "agree_rate": (agree / total) if total else 1.0,
        "live_trade_gates": live_trades,
        "shadow_trade_gates": shad_trades,
        "diff_symbols": len(diffs),
        "policy_overrides": pol_over,
        "exec_overrides": exe_over,
        "env_overrides": env_over,
    }
    return summary, diffs
