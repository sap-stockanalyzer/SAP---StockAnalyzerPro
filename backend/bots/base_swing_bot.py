# backend/bots/base_swing_bot.py
"""
Base EOD Swing Bot Engine — AION Analytics (Policy + Regression v2, Runtime-Config)

This module implements a generic, AI-powered EOD swing bot, parameterized by
a SwingBotConfig.

It now uses:
    • node["policy"] intent/score/confidence (from policy_engine v6)
    • node["predictions"][horizon].predicted_return for ranking
    • context/buzz/social only as light modifiers
    • runtime overrides from stock_cache/master/bot/configs.json

All horizon-specific differences (1w/2w/4w) are captured in the config:
    • horizon
    • bot_key
    • max_positions
    • base_risk_pct
    • conf_threshold
    • stop_loss_pct
    • take_profit_pct
    • max_weight_per_name

Runners (runner_1w/2w/4w) supply an appropriate config and call:
    SwingBot(config).run(mode="full"|"loop")

Paths used:
    • rolling:       PATHS["ml_data"]/rolling.json.gz   (via core.data_pipeline)
    • bot state:     PATHS["stock_cache"]/master/bot/rolling_<bot_key>.json.gz
    • bot logs:      PATHS["ml_data"]/bot_logs/<horizon>/bot_activity_YYYY-MM-DD.json
    • insights:      PATHS["insights"]/top50_<horizon>.json
    • bot configs:   PATHS["stock_cache"]/master/bot/configs.json
"""

from __future__ import annotations

import json
import gzip
from dataclasses import dataclass, asdict
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from dt_backend.core.constants_dt import (
    CONFIDENCE_MIN,
    POSITIONS_MAX_OPEN,
    REGIME_EXPOSURE,
)

# Import alerting module for Slack notifications
try:
    from backend.monitoring.alerting import alert_swing
except Exception:
    # Fallback if alerting module not available
    def alert_swing(title: str, message: str, **kwargs) -> None:  # type: ignore
        pass

# ---------------------------------------------------------------------------
# Env helpers (swing knobs live in env; see knobs.env)
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        raw = (os.getenv(name, "") or "").strip()
        return float(raw) if raw else float(default)
    except Exception:
        return float(default)

def _env_int(name: str, default: int) -> int:
    try:
        raw = (os.getenv(name, "") or "").strip()
        return int(float(raw)) if raw else int(default)
    except Exception:
        return int(default)

def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name, "") or "").strip().lower()
    if raw in {"1","true","yes","y","on"}:
        return True
    if raw in {"0","false","no","n","off"}:
        return False
    return bool(default)

def apply_swing_env_overrides(cfg: "SwingBotConfig") -> "SwingBotConfig":
    """Mutate cfg in-place using SWING_* env knobs (best-effort)."""
    try:
        cfg.conf_threshold = _env_float("SWING_CONF_THRESHOLD", cfg.conf_threshold)
        cfg.max_positions = _env_int("SWING_MAX_POSITIONS", cfg.max_positions)
        cfg.max_weight_per_name = _env_float("SWING_MAX_WEIGHT_PER_NAME", cfg.max_weight_per_name)
        cfg.low_conf_max_fraction = _env_float("SWING_LOW_CONF_MAX_FRACTION", cfg.low_conf_max_fraction)

        # Phase 4
        cfg.starter_fraction = _env_float("SWING_STARTER_FRAC", cfg.starter_fraction)
        cfg.add_fraction = _env_float("SWING_ADD_FRAC", cfg.add_fraction)
        cfg.max_build_stages = _env_int("SWING_MAX_BUILD_STAGES", cfg.max_build_stages)
        cfg.min_days_between_adds = _env_int("SWING_MIN_DAYS_BETWEEN_ADDS", cfg.min_days_between_adds)
        cfg.add_conf_extra = _env_float("SWING_ADD_CONF_EXTRA", cfg.add_conf_extra)
        cfg.allow_build_adds = _env_bool("SWING_ALLOW_BUILDS", True)
    except Exception:
        pass
    # Phase 5: time-based holding / exit discipline
    cfg.min_hold_days = int(_env_int("SWING_MIN_HOLD_DAYS", cfg.min_hold_days))
    cfg.time_stop_days = int(_env_int("SWING_TIME_STOP_DAYS", cfg.time_stop_days))
    cfg.exit_confirmations = int(_env_int("SWING_EXIT_CONFIRMATIONS", cfg.exit_confirmations))
    cfg.exit_conf_buffer = float(_env_float("SWING_EXIT_CONF_BUFFER", cfg.exit_conf_buffer))

    # Phase 7: calibrated P(hit) + EV sizing
    cfg.use_phit = _env_bool("SWING_USE_PHIT", cfg.use_phit)
    cfg.min_phit = float(_env_float("SWING_MIN_PHIT", cfg.min_phit))
    cfg.loss_est_pct = float(_env_float("SWING_LOSS_EST_PCT", cfg.loss_est_pct))
    cfg.require_positive_ev = _env_bool("SWING_REQUIRE_POS_EV", cfg.require_positive_ev)
    cfg.ev_power = float(_env_float("SWING_EV_POWER", cfg.ev_power))

    # sane clamps
    cfg.min_hold_days = max(0, int(cfg.min_hold_days))
    cfg.time_stop_days = max(cfg.min_hold_days + 1, int(cfg.time_stop_days))
    cfg.exit_confirmations = max(1, int(cfg.exit_confirmations))
    cfg.exit_conf_buffer = max(0.0, min(0.50, float(cfg.exit_conf_buffer)))

    # Phase 7 clamps
    cfg.min_phit = max(0.50, min(0.90, float(cfg.min_phit)))
    cfg.loss_est_pct = max(0.0, min(0.50, float(cfg.loss_est_pct)))
    cfg.ev_power = max(0.25, min(3.0, float(cfg.ev_power)))

    return cfg
from backend.bots.brains.swing_brain_v2 import rank_universe_v2
from backend.tuning.swing_profile_loader import load_swing_profile

# ---------------------------------------------------------------------
# Config & logging
# ---------------------------------------------------------------------

try:
    from backend.core.config import PATHS
except Exception:
    from backend.config import PATHS  # type: ignore

# ---------------------------------------------------------------------
# Module-level logging functions
# ---------------------------------------------------------------------

def _now_iso() -> str:
    """Current time in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def log(msg: str) -> None:
    """Log swing bot activity to stdout and file."""
    ts = _now_iso()
    line = f"[swing_bot] {ts} {msg}"
    print(line, flush=True)
    
    # Also write to log file
    try:
        log_file = Path(PATHS.get("logs", Path(".") / "logs")) / "bots" / "swing_bot.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8", errors="replace") as f:
            f.write(line + "\n")
    except Exception:
        pass

# ---------------------------------------------------------------------
# Data pipeline fallback
# ---------------------------------------------------------------------

try:
    from backend.core.data_pipeline import _read_rolling  # type: ignore
except Exception:  # Fallback (e.g. older envs)
    from pathlib import Path as _Path
    import json as _json
    import gzip as _gzip

    ROOT = _Path(PATHS.get("root", "."))
    ROLLING_PATH = _Path(PATHS["ml_data"]) / "rolling.json.gz"

    def _read_rolling() -> dict:
        if not ROLLING_PATH.exists():
            return {}
        try:
            with _gzip.open(ROLLING_PATH, "rt", encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            return {}

# Swing Phase 0: truth/instrumentation (best-effort)
try:
    from backend.services.swing_truth_store import append_swing_event, bump_swing_metric
except Exception:  # pragma: no cover
    def append_swing_event(_event: dict) -> None:  # type: ignore
        return

    def bump_swing_metric(_name: str, _amount: float = 1.0) -> None:  # type: ignore
        return

# Swing Phase 7: calibrated P(hit) helper (optional)
try:
    from backend.calibration.phit_calibrator_swing import get_phit as _get_phit  # type: ignore
except Exception:  # pragma: no cover
    _get_phit = None  # type: ignore

ROOT = Path(PATHS.get("root", "."))
ML_DATA = Path(PATHS["ml_data"])
STOCK_CACHE = Path(PATHS["stock_cache"])
INSIGHTS_DIR = Path(PATHS.get("insights", ROOT / "insights"))

# Runtime bot config file (edited via /api/eod/configs)
BOT_CONFIG_FILE = STOCK_CACHE / "master" / "bot" / "configs.json"


# UI overrides store (enabled/aggression) — bots page controls.
# Engine respects "enabled": false by skipping runs.
_UI_OVERRIDES_PATH = None

def _bot_ui_overrides_path() -> Path:
    global _UI_OVERRIDES_PATH
    if _UI_OVERRIDES_PATH is not None:
        return _UI_OVERRIDES_PATH
    try:
        p = PATHS.get("bots_ui_overrides")
        if p:
            _UI_OVERRIDES_PATH = Path(p)
            return _UI_OVERRIDES_PATH
    except Exception:
        pass
    _UI_OVERRIDES_PATH = ML_DATA / "config" / "bots_ui_overrides.json"
    return _UI_OVERRIDES_PATH


def _bot_is_enabled(bot_key: str) -> bool:
    p = _bot_ui_overrides_path()
    try:
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                node = obj.get(bot_key) or {}
                if isinstance(node, dict) and ("enabled" in node):
                    return bool(node.get("enabled"))
    except Exception:
        pass
    return True



# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------


def _days_held(pos: "Position", now: datetime | None = None) -> int:
    """Return integer number of full days held (best-effort)."""
    now = now or datetime.now(timezone.utc)
    try:
        # Try entry_ts first (if it exists), fallback to last_add_ts
        ts = (getattr(pos, "entry_ts", "") or getattr(pos, "last_add_ts", "") or "").strip()
        if not ts:
            return 0
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return max(0, int((now - dt).total_seconds() // 86400))
    except Exception:
        return 0


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_gz_json(path: Path) -> Optional[dict]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_gz_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------
# Config & state models
# ---------------------------------------------------------------------


@dataclass
class SwingBotConfig:
    """Hyper-parameters and identity for an EOD swing bot."""

    horizon: str              # "1w", "2w", "4w"
    bot_key: str              # e.g. "eod_1w"
    max_positions: int
    base_risk_pct: float      # reserved for future sizing refinements
    conf_threshold: float     # minimum policy confidence to act
    stop_loss_pct: float      # -0.05 → -5%
    take_profit_pct: float    # 0.10 → +10%
    max_weight_per_name: float  # max fraction of equity in one symbol

    # Phase 4: cap sizing for low-conviction "probe" names (Tier C).
    low_conf_max_fraction: float = 0.35

    # Phase 7: EV shaping (used in universe scoring + ranking helpers)
    ev_power: float = 1.0
    # Phase 4: position building (starter entries + adds)
    starter_fraction: float = 0.35          # fraction of desired size on first entry
    add_fraction: float = 0.33              # fraction of desired size per add
    max_build_stages: int = 3               # 1=starter, 2..N=adds
    min_days_between_adds: int = 1
    add_conf_extra: float = 0.03            # require conf >= conf_threshold + add_conf_extra to add
    allow_build_adds: bool = True
    initial_cash: float = 100.0

    # Phase 5: time-based holding / exit discipline (reduce churn)
    min_hold_days: int = 2            # ignore SELL flips for N days (stop-loss still applies)
    time_stop_days: int = 12          # if it's not working by then, free the capital
    exit_confirmations: int = 2       # consecutive SELL signals required
    exit_conf_buffer: float = 0.05    # SELL must exceed conf_threshold by this margin

    # Phase 7: calibrated P(hit) + EV-based sizing (optional; defaults ON)
    use_phit: bool = True
    min_phit: float = 0.52
    # Expected loss on a miss as a fraction of price (seatbelt for EV sizing).
    # If you prefer, set to your typical stop distance.
    loss_est_pct: float = 0.06
    # If True, require EV > 0 to allocate size; otherwise allow small "exploratory" size.
    require_positive_ev: bool = True


@dataclass
class Position:
    qty: float
    entry: float
    stop: float
    target: float

    # Phase 4: starter entries + adds (position build state)
    goal_qty: float = 0.0           # desired full size in shares at last rebalance
    build_stage: int = 0            # 0 = none, 1 = starter, 2..N = adds
    last_add_ts: str = ""           # ISO8601 Z


@dataclass
class BotState:
    cash: float
    positions: Dict[str, Position]
    last_equity: float
    last_updated: str

    @classmethod
    def from_dict(cls, d: dict, initial_cash: float) -> "BotState":
        cash = _safe_float(d.get("cash", initial_cash), initial_cash)
        last_equity = _safe_float(d.get("last_equity", cash), cash)
        last_updated = str(d.get("last_updated") or _now_iso())
        raw_pos = d.get("positions", {}) or {}
        positions: Dict[str, Position] = {}
        for sym, pd in raw_pos.items():
            positions[str(sym).upper()] = Position(
                qty=_safe_float(pd.get("qty"), 0.0),
                entry=_safe_float(pd.get("entry"), 0.0),
                stop=_safe_float(pd.get("stop"), 0.0),
                target=_safe_float(pd.get("target"), 0.0),
                goal_qty=_safe_float(pd.get("goal_qty"), 0.0),
                build_stage=int(_safe_float(pd.get("build_stage"), 0.0) or 0),
                last_add_ts=str(pd.get("last_add_ts") or ""),
            )
        return cls(
            cash=cash,
            positions=positions,
            last_equity=last_equity,
            last_updated=last_updated,
        )

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "last_equity": self.last_equity,
            "last_updated": self.last_updated,
            "positions": {s: asdict(p) for s, p in self.positions.items()},
        }


@dataclass
class Trade:
    t: str
    symbol: str
    side: str  # "BUY" / "SELL"
    qty: float
    price: float
    reason: str
    pnl: Optional[float] = None


# ---------------------------------------------------------------------
# SwingBot engine
# ---------------------------------------------------------------------


class SwingBot:
    """
    Generic AI-powered EOD swing bot (Policy + Regression aware, runtime-configurable).

    Usage:
        from backend.bots.base_swing_bot import SwingBot
        from backend.bots.strategy_1w import CONFIG as CONFIG_1W

        bot = SwingBot(CONFIG_1W)
        bot.run("full")  # or "loop"
    """

    def __init__(self, config: SwingBotConfig) -> None:
        self.cfg = config

        # Derived paths
        # NOTE: Rolling now comes from core data pipeline (ml_data/rolling.json.gz)
        self.bot_state_file = (
            STOCK_CACHE / "master" / "bot" / f"rolling_{config.bot_key}.json.gz"
        )
        self.bot_log_dir = ML_DATA / "bot_logs" / config.horizon
        self.insights_file = INSIGHTS_DIR / f"top50_{config.horizon}.json"

        # Initialize rejection tracking for alerts
        self._last_rejection_counts: Dict[str, int] = {}
        self._last_universe_analyzed: int = 0
        self._last_universe_qualified: int = 0

        # Apply runtime overrides from configs.json (if present)
        self._apply_runtime_overrides()

        # Apply env knobs (knobs.env) last so deploys can tune without code changes.
        apply_swing_env_overrides(self.cfg)

        log(
            f"[{self.cfg.bot_key}] SwingBot initialized — "
            f"horizon={self.cfg.horizon}, "
            f"max_positions={self.cfg.max_positions}, "
            f"conf_threshold={self.cfg.conf_threshold:.3f}, "
            f"stop_loss_pct={self.cfg.stop_loss_pct:.3f}, "
            f"take_profit_pct={self.cfg.take_profit_pct:.3f}, "
            f"max_weight_per_name={self.cfg.max_weight_per_name:.3f}, "
            f"initial_cash={self.cfg.initial_cash:.2f}"
        )

    # -------------------- Runtime config overrides -------------------- #

    def _load_bot_overrides(self) -> Dict[str, Any]:
        """
        Load overrides from BOT_CONFIG_FILE, shape:
            {
              "1w": {
                "max_positions": 20,
                "conf_threshold": 0.55,
                "stop_loss_pct": -0.05,
                "take_profit_pct": 0.10,
                "max_weight_per_name": 0.15,
                "base_risk_pct": 0.20,       # optional
                "initial_cash": 100.0        # optional
              },
              ...
            }
        """
        js = _read_json(BOT_CONFIG_FILE)
        if not isinstance(js, dict):
            return {}
        horizon_cfg = js.get(self.cfg.horizon)
        if not isinstance(horizon_cfg, dict):
            return {}
        return horizon_cfg

    def _apply_runtime_overrides(self) -> None:
        """
        Merge JSON overrides into this bot's config.
        Only applies to numeric hyper-params; never touches horizon or bot_key.
        """
        overrides = self._load_bot_overrides()
        if not overrides:
            return

        changed: Dict[str, Any] = {}

        # int field
        if "max_positions" in overrides:
            try:
                val = int(overrides["max_positions"])
                if val > 0 and val != self.cfg.max_positions:
                    self.cfg.max_positions = val
                    changed["max_positions"] = val
            except Exception:
                pass

        # float fields (with sane bounds where relevant)
        def _apply_float(key: str, attr: str | None = None):
            dest = attr or key
            if key not in overrides:
                return
            try:
                val = float(overrides[key])
            except Exception:
                return
            old = getattr(self.cfg, dest, None)
            # light validation for some keys
            if key in ("conf_threshold",):
                if not (0.0 <= val <= 1.0):
                    return
            if key in ("max_weight_per_name", "base_risk_pct"):
                if val <= 0.0 or val > 1.0:
                    return
            if key in ("stop_loss_pct", "take_profit_pct"):
                # allow negative for stop, positive for take; basic sanity
                if not (-1.0 <= val <= 1.0):
                    return
            if key == "initial_cash":
                if val <= 0:
                    return

            if old is None or float(old) != val:
                setattr(self.cfg, dest, val)
                changed[dest] = val

        _apply_float("conf_threshold")
        _apply_float("stop_loss_pct")
        _apply_float("take_profit_pct")
        _apply_float("max_weight_per_name")
        _apply_float("base_risk_pct")
        _apply_float("initial_cash")

        if changed:
            log(
                f"[{self.cfg.bot_key}] 🔧 Runtime overrides applied from {BOT_CONFIG_FILE}: "
                + ", ".join(f"{k}={v}" for k, v in changed.items())
            )

    # ------------------------ Data Loading ------------------------ #

    def load_rolling(self) -> Dict[str, dict]:
        js = _read_rolling() or {}
        if not isinstance(js, dict):
            log(f"[{self.cfg.bot_key}] ⚠️ rolling.json.gz missing or invalid.")
            return {}
        out: Dict[str, dict] = {}
        for sym, node in js.items():
            if sym.startswith("_"):
                continue
            out[str(sym).upper()] = node or {}
        log(f"[{self.cfg.bot_key}] rolling loaded for {len(out)} symbols.")
        return out

    def load_insights(self) -> List[dict]:
        js = _read_json(self.insights_file)
        if not isinstance(js, (dict, list)):
            log(
                f"[{self.cfg.bot_key}] ⚠️ Insights file {self.insights_file} "
                f"missing or invalid."
            )
            return []
        if isinstance(js, dict):
            arr = js.get("items") or js.get("insights") or []
        else:
            arr = js
        if not isinstance(arr, list):
            return []
        return arr

    # ---------------------- Feature Extractors -------------------- #

    @staticmethod
    def _extract_price(node: dict) -> Optional[float]:
        price = (
            node.get("price")
            or node.get("last")
            or node.get("close")
            or node.get("c")
        )
        if price is None:
            return None
        try:
            p = float(price)
            return p if p > 0 else None
        except Exception:
            return None

    def _extract_policy_signal(self, node: dict) -> Tuple[str, float, float]:
        """
        Use unified policy engine output:

            node["policy"] = {
                "intent": "BUY" | "SELL" | "HOLD",
                "score": float,          # ~[-0.3,0.3]
                "confidence": float,     # ~[0.5,0.97]
                ...
            }
        """
        pol = node.get("policy") or {}
        intent = str(pol.get("intent") or "").upper()
        conf = _safe_float(pol.get("confidence"), 0.0)
        score = _safe_float(pol.get("score"), 0.0)
        return intent, conf, score

    def _extract_horizon_pred(self, node: dict) -> float:
        """
        Get horizon-specific expected return from regression block:

            node["predictions"][horizon]["predicted_return"]
        """
        preds = node.get("predictions") or {}
        hblk = preds.get(self.cfg.horizon) or {}
        return _safe_float(hblk.get("predicted_return"), 0.0)

    @staticmethod
    def _extract_regime_bias(node: dict) -> float:
        """
        Simple, robust bias in [0.5, 1.5], based on context / regime.

        Currently:
          - Uses context.trend (if present)
          - Optionally uses policy.reasons.market_regime
        """
        ctx = node.get("context", {}) or {}
        trend = str(ctx.get("trend", "")).lower()

        pol = node.get("policy") or {}
        reasons = pol.get("reasons") or {}
        regime = str(reasons.get("market_regime", "chop")).lower()

        bias = 1.0
        if trend == "bullish":
            bias *= 1.05
        elif trend == "bearish":
            bias *= 0.95

        if regime == "bull":
            bias *= 1.05
        elif regime in ("bear", "panic"):
            bias *= 0.9

        return max(0.5, min(1.5, bias))


    def _regime_risk_profile(self, rolling: Dict[str, Any]) -> Dict[str, float]:
        """Phase 3: replace hard regime blocks with regime-based risk adjustments.

        We *never* forbid trading outright here. We adjust:
          - max_positions (portfolio breadth)
          - max_weight_per_name (per-name risk)
          - stop_loss_pct (stop width)
          - take_profit_pct (profit target realism)

        All are best-effort and env-overridable.
        """
        import os

        g = rolling.get("_GLOBAL") if isinstance(rolling, dict) else None
        g = g if isinstance(g, dict) else {}
        reg = g.get("regime") if isinstance(g.get("regime"), dict) else {}
        label = str((reg or {}).get("label") or "unknown").strip().lower()

        # Defaults: conservative in bear/stress, modestly freer in bull.
        if label in {"bull"}:
            pos_mult, w_mult, stop_mult, tp_mult = 1.15, 1.10, 1.00, 1.05
        elif label in {"bear", "risk_off"}:
            pos_mult, w_mult, stop_mult, tp_mult = 0.75, 0.80, 1.15, 0.90
        elif label in {"stress", "crash"}:
            pos_mult, w_mult, stop_mult, tp_mult = 0.60, 0.70, 1.25, 0.85
        elif label in {"chop"}:
            pos_mult, w_mult, stop_mult, tp_mult = 0.85, 0.90, 1.05, 0.95
        else:
            pos_mult, w_mult, stop_mult, tp_mult = 0.90, 0.95, 1.05, 0.95

        def _envf(name: str, default: float) -> float:
            try:
                raw = (os.getenv(name, "") or "").strip()
                return float(raw) if raw else float(default)
            except Exception:
                return float(default)

        # Optional env overrides (keep API keys separate — these are "knobs")
        pos_mult = _envf("SWING_REGIME_POS_MULT", pos_mult)
        w_mult = _envf("SWING_REGIME_WEIGHT_MULT", w_mult)
        stop_mult = _envf("SWING_REGIME_STOP_MULT", stop_mult)
        tp_mult = _envf("SWING_REGIME_TP_MULT", tp_mult)

        # Sane clamps
        pos_mult = max(0.25, min(2.0, float(pos_mult)))
        w_mult = max(0.25, min(2.0, float(w_mult)))
        stop_mult = max(0.75, min(2.0, float(stop_mult)))
        tp_mult = max(0.50, min(1.50, float(tp_mult)))

        return {
            "label": label,
            "pos_mult": float(pos_mult),
            "weight_mult": float(w_mult),
            "stop_mult": float(stop_mult),
            "tp_mult": float(tp_mult),
        }


    # ------------------------- State I/O -------------------------- #

    def load_bot_state(self) -> BotState:
        js = _read_gz_json(self.bot_state_file)
        if not isinstance(js, dict):
            log(f"[{self.cfg.bot_key}] ℹ️ No existing state — seeding new bot state.")
            return BotState(
                cash=self.cfg.initial_cash,
                positions={},
                last_equity=self.cfg.initial_cash,
                last_updated=_now_iso(),
            )
        return BotState.from_dict(js, initial_cash=self.cfg.initial_cash)

    def save_bot_state(self, state: BotState) -> None:
        self.bot_state_file.parent.mkdir(parents=True, exist_ok=True)
        _write_gz_json(self.bot_state_file, state.to_dict())
        log(
            f"[{self.cfg.bot_key}] 💾 State saved — "
            f"positions={len(state.positions)} cash={state.cash:.2f} "
            f"equity={state.last_equity:.2f}"
        )

    # -------------------- AI Ranking & Weights -------------------- #

    def build_ai_ranked_universe(
        self, rolling: Dict[str, dict], insights: List[dict]
    ) -> List[Tuple[str, float]]:
        """
        Combine:
            • policy intent/score/confidence
            • horizon predicted_return (regression)
            • insight rank
            • simple regime bias

        into a composite score for each symbol.
        Long-only for now (BUY-only).
        """
        # Extract global regime label from rolling data
        g = rolling.get("_GLOBAL") if isinstance(rolling, dict) else None
        g = g if isinstance(g, dict) else {}
        reg = g.get("regime") if isinstance(g.get("regime"), dict) else {}
        regime = str((reg or {}).get("label") or "unknown").strip().lower()
        
        # insight rank: lower index = stronger rank
        insight_rank: Dict[str, int] = {}
        for idx, row in enumerate(insights):
            sym = str(row.get("symbol") or row.get("ticker") or "").upper()
            if not sym:
                continue
            insight_rank[sym] = idx

        universe_scores: Dict[str, float] = {}

        # Phase S0: optional rejection logging for "missed opportunities" analytics.
        log_reject = str(os.getenv("SWING_LOG_REJECTIONS", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
        max_reject_events = int(os.getenv("SWING_LOG_REJECTIONS_MAX", "250"))
        # Slack rejection alerts (separate from logging)
        send_reject_alerts = str(os.getenv("SWING_SEND_REJECTIONS", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
        max_reject_alerts = int(os.getenv("SWING_SEND_REJECTIONS_MAX", "20"))
        reject_events = 0
        reject_alerts_sent = 0
        reject_counts: Dict[str, int] = {}

        for sym, node in rolling.items():
            price = self._extract_price(node)
            if price is None or price <= 0:
                if log_reject and reject_events < max_reject_events:
                    append_swing_event({
                        "type": "swing_reject",
                        "bot": self.cfg.bot_key,
                        "symbol": str(sym).upper(),
                        "reason": "no_price",
                    })
                    reject_events += 1
                    reject_counts["no_price"] = reject_counts.get("no_price", 0) + 1
                continue

            # Policy + regression signal
            intent, pol_conf, pol_score = self._extract_policy_signal(node)
            exp_ret = self._extract_horizon_pred(node)

            # Long-only + minimum confidence
            if intent != "BUY":
                if log_reject and reject_events < max_reject_events:
                    append_swing_event({
                        "type": "swing_reject",
                        "bot": self.cfg.bot_key,
                        "symbol": str(sym).upper(),
                        "reason": "intent_not_buy",
                        "intent": intent,
                        "confidence": float(pol_conf),
                        "expected_return": float(exp_ret),
                    })
                    reject_events += 1
                reject_counts["intent_not_buy"] = reject_counts.get("intent_not_buy", 0) + 1
                continue
            if pol_conf < self.cfg.conf_threshold:
                if log_reject and reject_events < max_reject_events:
                    append_swing_event({
                        "type": "swing_reject",
                        "bot": self.cfg.bot_key,
                        "symbol": str(sym).upper(),
                        "reason": "conf_below_threshold",
                        "intent": intent,
                        "confidence": float(pol_conf),
                        "conf_threshold": float(self.cfg.conf_threshold),
                        "expected_return": float(exp_ret),
                    })
                    reject_events += 1
                reject_counts["conf_below_threshold"] = reject_counts.get("conf_below_threshold", 0) + 1
                # Send Slack rejection alert if enabled
                if send_reject_alerts and reject_alerts_sent < max_reject_alerts:
                    self._send_rejection_alert(
                        symbol=sym,
                        price=price,
                        reason="conf_below_threshold",
                        details={
                            "confidence": pol_conf,
                            "conf_threshold": self.cfg.conf_threshold,
                            "expected_return": exp_ret,
                        },
                    )
                    reject_alerts_sent += 1
                continue
            # Phase 7: calibrated probability-of-hit + EV (expected value) gating.
            p_hit = float(pol_conf)
            if self.cfg.use_phit and callable(_get_phit):
                try:
                    p_hit = float(_get_phit(
                        base_conf=float(pol_conf),
                        expected_return=float(exp_ret),
                        regime_label=str(regime),
                    ))
                except Exception:
                    p_hit = float(pol_conf)
            p_hit = max(0.0, min(0.999, p_hit))

            # Optional gate: if P(hit) isn't above coin-flip + margin, don't allocate.
            if self.cfg.use_phit and p_hit < float(self.cfg.min_phit):
                if log_reject and reject_events < max_reject_events:
                    append_swing_event({
                        "type": "swing_reject",
                        "bot": self.cfg.bot_key,
                        "symbol": str(sym).upper(),
                        "reason": "phit_below_threshold",
                        "intent": intent,
                        "confidence": float(pol_conf),
                        "p_hit": float(p_hit),
                        "min_phit": float(self.cfg.min_phit),
                        "expected_return": float(exp_ret),
                        "price": float(price),
                    })
                    reject_events += 1
                reject_counts["phit_below_threshold"] = reject_counts.get("phit_below_threshold", 0) + 1
                # Send Slack rejection alert if enabled
                if send_reject_alerts and reject_alerts_sent < max_reject_alerts:
                    self._send_rejection_alert(
                        symbol=sym,
                        price=price,
                        reason="phit_below_threshold",
                        details={
                            "confidence": pol_conf,
                            "p_hit": p_hit,
                            "min_phit": self.cfg.min_phit,
                            "expected_return": exp_ret,
                        },
                    )
                    reject_alerts_sent += 1
                continue

            # EV model: hit → +expected_return, miss → -loss_est_pct
            loss_est = max(0.0, float(self.cfg.loss_est_pct))
            ev = (p_hit * float(exp_ret)) - ((1.0 - p_hit) * loss_est)

            if self.cfg.require_positive_ev and ev <= 0.0:
                if log_reject and reject_events < max_reject_events:
                    append_swing_event({
                        "type": "swing_missed_candidate",
                        "bot": self.cfg.bot_key,
                        "symbol": str(sym).upper(),
                        "reason": "non_positive_ev",
                        "intent": intent,
                        "confidence": float(pol_conf),
                        "p_hit": float(p_hit),
                        "expected_return": float(exp_ret),
                        "loss_est_pct": float(loss_est),
                        "ev": float(ev),
                        "price": float(price),
                    })
                    reject_events += 1
                reject_counts["non_positive_ev"] = reject_counts.get("non_positive_ev", 0) + 1
                # Send Slack rejection alert if enabled
                if send_reject_alerts and reject_alerts_sent < max_reject_alerts:
                    self._send_rejection_alert(
                        symbol=sym,
                        price=price,
                        reason="non_positive_ev",
                        details={
                            "confidence": pol_conf,
                            "p_hit": p_hit,
                            "expected_return": exp_ret,
                            "loss_est_pct": loss_est,
                            "ev": ev,
                        },
                    )
                    reject_alerts_sent += 1
                continue

            # Base AI score (Phase 7): EV * confidence
            ai_score = (max(0.0, float(ev)) ** float(self.cfg.ev_power)) * (0.5 + 0.5 * pol_conf)

            # Add a bit of raw policy score (directional conviction)
            ai_score += 0.5 * pol_score

            # insight rank bonus (if present)
            if sym in insight_rank:
                rank = insight_rank[sym]
                n = max(1, len(insights))
                rank_score = 1.0 - (rank / n)
                ai_score += 0.2 * rank_score

            # regime/context bias
            bias = self._extract_regime_bias(node)
            score = ai_score * bias

            # light penalty for penny-ish names
            if price < 3:
                score *= 0.7

            if score > 0:
                universe_scores[sym] = score

        ranked = sorted(universe_scores.items(), key=lambda kv: kv[1], reverse=True)
        if log_reject:
            try:
                bump_swing_metric(f"{self.cfg.bot_key}.rejections_logged", float(reject_events))
                for k, v in reject_counts.items():
                    bump_swing_metric(f"{self.cfg.bot_key}.reject.{k}", float(v))
            except Exception:
                pass
        log(f"[{self.cfg.bot_key}] AI-ranked universe size={len(ranked)}")
        
        # Store rejection counts for summary alert
        self._last_rejection_counts = reject_counts
        self._last_universe_analyzed = len(rolling) - 1 if "_GLOBAL" in rolling else len(rolling)  # Exclude _GLOBAL
        self._last_universe_qualified = len(ranked)
        
        return ranked

    def construct_target_weights(self, ranked: List[Tuple]) -> Dict[str, float]:
        """
        From ranked (sym, score), produce target weights.
        - keep top max_positions
        - normalize
        - cap each name at max_weight_per_name
        """
        if not ranked:
            return {}

        top = ranked[: self.cfg.max_positions]

        # ranked items may be (sym, score) or (sym, score, tier)
        def _score(it: Tuple) -> float:
            try:
                return max(0.0, float(it[1]))
            except Exception:
                return 0.0

        def _tier(it: Tuple) -> str:
            try:
                return str(it[2]).upper() if len(it) >= 3 else "A"
            except Exception:
                return "A"

        scores = [_score(it) for it in top]
        total = sum(scores)
        if total <= 0:
            return {}

        weights: Dict[str, float] = {}
        for it in top:
            sym = str(it[0]).upper()
            s = _score(it)
            tier = _tier(it)

            w = s / total

            # Phase 4: tier-based caps (C ≈ probe).
            tier_mult = 1.0
            if tier == "B":
                tier_mult = 0.75
            elif tier == "C":
                tier_mult = float(getattr(self.cfg, "low_conf_max_fraction", 0.35))
            cap = float(self.cfg.max_weight_per_name) * max(0.0, min(1.0, tier_mult))
            w = min(w, cap)
            weights[sym] = w

        # renormalize after capping
        total = sum(weights.values())
        if total > 0:
            for sym in list(weights.keys()):
                weights[sym] /= total

        log(f"[{self.cfg.bot_key}] Constructed {len(weights)} target weights.")
        return weights

    # ---------------------- Alert Methods ----------------------- #

    def _send_buy_alert(
        self,
        symbol: str,
        qty: float,
        price: float,
        rolling: Dict[str, dict],
    ) -> None:
        """Send Slack alert for BUY trade."""
        try:
            # Extract signal data
            node = rolling.get(symbol, {})
            intent, pol_conf, _ = self._extract_policy_signal(node)
            exp_ret = self._extract_horizon_pred(node)
            
            # Calculate P(Hit) if calibration available
            p_hit = pol_conf
            if self.cfg.use_phit and callable(_get_phit):
                try:
                    g = rolling.get("_GLOBAL", {})
                    reg = g.get("regime", {}) if isinstance(g, dict) else {}
                    regime = str((reg or {}).get("label") or "unknown")
                    p_hit = float(_get_phit(
                        base_conf=float(pol_conf),
                        expected_return=float(exp_ret),
                        regime_label=regime,
                    ))
                except Exception:
                    pass
            
            # Format alert message
            title = f"📈 Swing Bot {self.cfg.horizon} - BUY"
            message = f"""Symbol: {symbol}
Qty: {qty:.2f} shares @ ${price:.2f}
Confidence: {pol_conf*100:.1f}% (> threshold: {self.cfg.conf_threshold*100:.1f}%)
Expected Return: {exp_ret*100:+.2f}%
P(Hit): {p_hit*100:.1f}%
Why Selected: Top rank in AI universe, strong signal + positive EV
Entry Time: {_now_iso()}"""
            
            alert_swing(title, message, level="info", skip_rate_limit=True)
            log(f"[{self.cfg.bot_key}] 📤 Sent BUY alert for {symbol}")
        except Exception as e:
            log(f"[{self.cfg.bot_key}] ⚠️ Failed to send BUY alert: {e}")

    def _send_sell_alert(
        self,
        symbol: str,
        qty: float,
        price: float,
        entry_price: float,
        reason: str,
        entry_ts: Optional[str] = None,
    ) -> None:
        """Send Slack alert for SELL trade."""
        try:
            # Calculate PnL
            pnl_dollars = (price - entry_price) * qty
            pnl_pct = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
            
            # Format PnL with proper sign
            if pnl_dollars >= 0:
                pnl_str = f"+${pnl_dollars:.2f}"
                pct_str = f"+{pnl_pct:.2f}%"
            else:
                pnl_str = f"-${abs(pnl_dollars):.2f}"
                pct_str = f"{pnl_pct:.2f}%"  # pnl_pct is already negative
            
            # Calculate hold duration
            hold_duration = "unknown"
            if entry_ts:
                try:
                    entry_dt = datetime.fromisoformat(entry_ts.replace("Z", "+00:00"))
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                    now_dt = datetime.now(timezone.utc)
                    delta = now_dt - entry_dt
                    days = delta.days
                    hours = delta.seconds // 3600
                    hold_duration = f"{days} days, {hours} hours"
                except Exception:
                    pass
            
            # Format reason
            reason_display = reason.replace("_", " ").title()
            
            # Format alert message
            title = f"📉 Swing Bot {self.cfg.horizon} - SELL"
            message = f"""Symbol: {symbol}
Qty: {qty:.2f} shares @ ${price:.2f}
PnL: {pnl_str} ({pct_str} return)
Reason: {reason_display}
Hold Duration: {hold_duration}
Exit Time: {_now_iso()}"""
            
            alert_swing(title, message, level="info", skip_rate_limit=True)
            log(f"[{self.cfg.bot_key}] 📤 Sent SELL alert for {symbol}")
        except Exception as e:
            log(f"[{self.cfg.bot_key}] ⚠️ Failed to send SELL alert: {e}")

    def _send_rejection_alert(
        self,
        symbol: str,
        price: float,
        reason: str,
        details: Dict[str, Any],
    ) -> None:
        """Send Slack alert for rejected symbol (optional, controlled by env var)."""
        try:
            # Format reason
            reason_display = reason.replace("_", " ").title()
            
            # Format details
            details_lines = []
            if "confidence" in details:
                details_lines.append(f"  • Confidence: {details['confidence']*100:.1f}%")
            if "conf_threshold" in details:
                details_lines.append(f"  • Required: {details['conf_threshold']*100:.1f}%")
                if "confidence" in details:
                    gap = (details["confidence"] - details["conf_threshold"]) * 100
                    details_lines.append(f"  • Gap: {gap:+.1f}%")
            if "p_hit" in details:
                details_lines.append(f"  • P(Hit): {details['p_hit']*100:.1f}%")
            if "min_phit" in details:
                details_lines.append(f"  • Min P(Hit): {details['min_phit']*100:.1f}%")
            if "expected_return" in details:
                details_lines.append(f"  • Expected Return: {details['expected_return']*100:+.2f}%")
            if "ev" in details:
                details_lines.append(f"  • EV: {details['ev']:.3f}")
            if "loss_est_pct" in details:
                details_lines.append(f"  • Loss Estimate: {details['loss_est_pct']*100:.1f}%")
            
            details_text = "\n".join(details_lines) if details_lines else "  • No additional details"
            
            # Format alert message
            title = f"⛔ Swing Bot {self.cfg.horizon} - REJECTED: {symbol}"
            message = f"""Price: ${price:.2f}
Reason: {reason_display}
Details:
{details_text}"""
            
            alert_swing(title, message, level="warning", skip_rate_limit=True)
            log(f"[{self.cfg.bot_key}] 📤 Sent REJECTION alert for {symbol}")
        except Exception as e:
            log(f"[{self.cfg.bot_key}] ⚠️ Failed to send REJECTION alert: {e}")

    def _send_rebalance_summary(
        self,
        trades: List[Trade],
        universe_analyzed: int,
        universe_qualified: int,
        rejection_counts: Dict[str, int],
        state: BotState,
    ) -> None:
        """Send Slack alert with rebalance summary."""
        try:
            # Count trades by type
            buys = sum(1 for t in trades if t.side == "BUY")
            sells = sum(1 for t in trades if t.side == "SELL")
            total_trades = len(trades)
            
            # Calculate rejection stats
            total_rejected = universe_analyzed - universe_qualified
            
            # Format top rejections
            rejection_lines = []
            if rejection_counts:
                sorted_rejections = sorted(rejection_counts.items(), key=lambda x: x[1], reverse=True)
                for reason, count in sorted_rejections[:5]:  # Top 5
                    pct = (count / total_rejected * 100) if total_rejected > 0 else 0
                    reason_display = reason.replace("_", " ").title()
                    rejection_lines.append(f"  • {reason_display}: {count} ({pct:.0f}%)")
            
            rejection_text = "\n".join(rejection_lines) if rejection_lines else "  • None"
            
            # Format portfolio state
            positions_count = len(state.positions)
            equity = state.last_equity if state.last_equity > 0 else state.cash
            
            # Format alert message
            title = f"📊 Swing Bot {self.cfg.horizon} - Rebalance Complete"
            message = f"""Trades Executed: {total_trades} total
  ✅ Buys: {buys} positions entered
  ✅ Sells: {sells} positions closed
  
Universe Analyzed: {universe_analyzed} symbols
  ✅ Qualified: {universe_qualified} symbols
  ❌ Rejected: {total_rejected} symbols

Top Rejections:
{rejection_text}

Portfolio:
  • Positions: {positions_count} open
  • Cash: ${state.cash:,.2f}
  • Equity: ${equity:,.2f}"""
            
            alert_swing(title, message, level="info", skip_rate_limit=True)
            log(f"[{self.cfg.bot_key}] 📤 Sent REBALANCE SUMMARY alert")
        except Exception as e:
            log(f"[{self.cfg.bot_key}] ⚠️ Failed to send REBALANCE SUMMARY alert: {e}")

    # ---------------------- Portfolio Logic ----------------------- #

    @staticmethod
    def compute_equity(state: BotState, prices: Dict[str, float]) -> float:
        equity = state.cash
        for sym, pos in state.positions.items():
            px = prices.get(sym)
            if isinstance(px, (int, float)):
                equity += pos.qty * px
        return equity

    def rebalance_full(
        self,
        state: BotState,
        rolling: Dict[str, dict],
        target_weights: Dict[str, float],
    ) -> List[Trade]:
        """
        FULL pre-market rebalance:
            - Sell anything outside universe
            - Adjust positions toward target weights
        """
        trades: List[Trade] = []

        prices: Dict[str, float] = {}
        for sym, node in rolling.items():
            px = self._extract_price(node)
            if px is not None:
                prices[sym] = px

        equity = self.compute_equity(state, prices)
        if equity <= 0:
            equity = state.cash if state.cash > 0 else self.cfg.initial_cash

        log(
            f"[{self.cfg.bot_key}] Starting FULL rebalance — "
            f"equity={equity:.2f}, positions={len(state.positions)}"
        )

        # 1) Sell names not in target
        for sym in list(state.positions.keys()):
            if sym not in target_weights:
                px = prices.get(sym)
                pos = state.positions[sym]
                if isinstance(px, (int, float)) and pos.qty > 0:
                    proceeds = pos.qty * px
                    state.cash += proceeds
                    pnl = (px - pos.entry) * pos.qty
                    trades.append(
                        Trade(
                            t=_now_iso(),
                            symbol=sym,
                            side="SELL",
                            qty=pos.qty,
                            price=px,
                            reason="REMOVE_FROM_UNIVERSE",
                            pnl=pnl,
                        )
                    )
                    log(
                        f"[{self.cfg.bot_key}] SELL {pos.qty:.4f} {sym} @ {px:.4f} "
                        f"(universe exit) PnL={pnl:.2f}"
                    )
                    # Send SELL alert
                    self._send_sell_alert(
                        symbol=sym,
                        qty=pos.qty,
                        price=px,
                        entry_price=pos.entry,
                        reason="REMOVE_FROM_UNIVERSE",
                        entry_ts=getattr(pos, "last_add_ts", None),
                    )
                del state.positions[sym]

        # 2) Recompute equity from target universe only
        prices_now = {
            s: prices.get(s)
            for s in target_weights.keys()
            if isinstance(prices.get(s), (int, float))
        }
        equity = self.compute_equity(state, prices_now)
        state.last_equity = equity

        # 3) Move toward target weights
        for sym, target_w in target_weights.items():
            px = prices_now.get(sym)
            if px is None or px <= 0:
                continue

            target_value = equity * target_w
            pos = state.positions.get(sym)
            current_value = pos.qty * px if pos else 0.0
            diff_value = target_value - current_value

            # ignore tiny moves
            if abs(diff_value) < max(10.0, 0.01 * equity):
                continue

            qty_delta = diff_value / px

            if qty_delta > 0:
                # BUY / increase (Phase 4: starter entry + adds)
                desired_qty = float(target_value / max(px, 1e-9))
                node = rolling.get(sym) if isinstance(rolling, dict) else None
                if node:
                    intent, sig_conf, _ = self._extract_policy_signal(node)
                else:
                    sig_conf = 0.0

                buy_qty = float(qty_delta)

                # New position → starter entry only
                if pos is None or pos.qty <= 0:
                    buy_qty = min(buy_qty, max(0.0, desired_qty * float(self.cfg.starter_fraction)))
                    if buy_qty > 0:
                        state.positions[sym] = Position(
                            qty=float(buy_qty),
                            entry=float(px),
                            stop=float(px * 0.90),
                            target=float(px * 1.10),
                            goal_qty=float(desired_qty),
                            build_stage=1,
                            last_add_ts=_now_iso(),
                        )
                        pos = state.positions[sym]

                # Existing position → optional adds toward desired size
                elif bool(getattr(self.cfg, "allow_build_adds", True)) and pos.qty < desired_qty and int(getattr(self.cfg, "max_build_stages", 3)) > 1:
                    # Gate adds: confidence and time spacing
                    can_add = True
                    try:
                        if pos.last_add_ts:
                            last_dt = datetime.fromisoformat(str(pos.last_add_ts).replace("Z", "+00:00"))
                            if last_dt.tzinfo is None:
                                last_dt = last_dt.replace(tzinfo=timezone.utc)
                            days = (datetime.now(timezone.utc).date() - last_dt.astimezone(timezone.utc).date()).days
                            if days < int(getattr(self.cfg, "min_days_between_adds", 1)):
                                can_add = False
                    except Exception:
                        pass

                    if sig_conf < float(self.cfg.conf_threshold) + float(getattr(self.cfg, "add_conf_extra", 0.03)):
                        can_add = False

                    if int(getattr(pos, "build_stage", 0)) >= int(getattr(self.cfg, "max_build_stages", 3)):
                        can_add = False

                    if can_add:
                        step_qty = max(0.0, desired_qty * float(getattr(self.cfg, "add_fraction", 0.33)))
                        remaining = max(0.0, desired_qty - float(pos.qty))
                        buy_qty = min(buy_qty, step_qty, remaining)
                        if buy_qty > 0:
                            pos.build_stage = int(getattr(pos, "build_stage", 0)) + 1
                            pos.last_add_ts = _now_iso()

                # If we didn't create/add under build rules, fall back to classic rebalance qty.
                if buy_qty > 0:
                    state.cash -= buy_qty * px
                    if pos:
                        # Weighted-average entry
                        total = float(pos.qty) + float(buy_qty)
                        pos.entry = float((pos.entry * float(pos.qty) + px * float(buy_qty)) / max(total, 1e-9))
                        pos.qty = float(total)
                        pos.goal_qty = float(desired_qty)
                        state.positions[sym] = pos
                    else:
                        state.positions[sym] = Position(qty=float(buy_qty), entry=float(px), stop=float(px * 0.90), target=float(px * 1.10), goal_qty=float(desired_qty), build_stage=1, last_add_ts=_now_iso())

                    trades.append({
                        "symbol": sym,
                        "side": "BUY",
                        "qty": float(buy_qty),
                        "price": float(px),
                        "ts": _now_iso(),
                    })
                    log(f"[{self.cfg.bot_key}] BUY {sym} qty={buy_qty:.4f} px={px:.2f} cash={state.cash:.2f}")
                    # Send BUY alert
                    self._send_buy_alert(
                        symbol=sym,
                        qty=buy_qty,
                        price=px,
                        rolling=rolling,
                    )
            else:
                # SELL / decrease
                qty = abs(qty_delta)
                if not pos or pos.qty <= 0:
                    continue
                sell_qty = min(qty, pos.qty)
                state.cash += sell_qty * px
                pnl = (px - pos.entry) * sell_qty
                trades.append(
                    Trade(
                        t=_now_iso(),
                        symbol=sym,
                        side="SELL",
                        qty=sell_qty,
                        price=px,
                        reason="TARGET_REBALANCE",
                        pnl=pnl,
                    )
                )
                log(
                    f"[{self.cfg.bot_key}] SELL {sell_qty:.4f} {sym} @ {px:.4f} "
                    f"(target_w={target_w:.3f}) PnL={pnl:.2f}"
                )
                # Send SELL alert
                self._send_sell_alert(
                    symbol=sym,
                    qty=sell_qty,
                    price=px,
                    entry_price=pos.entry,
                    reason="TARGET_REBALANCE",
                    entry_ts=getattr(pos, "last_add_ts", None),
                )
                pos.qty -= sell_qty
                if pos.qty <= 0:
                    del state.positions[sym]

        state.last_equity = self.compute_equity(state, prices_now)
        state.last_updated = _now_iso()
        return trades

    def apply_loop_risk_checks(
        self, state: BotState, rolling: Dict[str, dict]
    ) -> List[Trade]:
        """
        Intraday loop:
            - enforce SL/TP
            - AI SELL exits for strong convictions
        """
        trades: List[Trade] = []
        prices: Dict[str, float] = {}

        for sym, node in rolling.items():
            px = self._extract_price(node)
            if px is not None:
                prices[sym] = px

        for sym in list(state.positions.keys()):
            pos = state.positions[sym]
            px = prices.get(sym)
            if px is None:
                continue

            # Stop-loss
            if px <= pos.stop:
                pnl = (px - pos.entry) * pos.qty
                state.cash += pos.qty * px
                trades.append(
                    Trade(
                        t=_now_iso(),
                        symbol=sym,
                        side="SELL",
                        qty=pos.qty,
                        price=px,
                        reason="STOP_LOSS",
                        pnl=pnl,
                    )
                )
                log(
                    f"[{self.cfg.bot_key}] STOP_LOSS SELL {pos.qty:.4f} {sym} @ {px:.4f} "
                    f"PnL={pnl:.2f}"
                )
                # Send SELL alert
                self._send_sell_alert(
                    symbol=sym,
                    qty=pos.qty,
                    price=px,
                    entry_price=pos.entry,
                    reason="STOP_LOSS",
                    entry_ts=getattr(pos, "last_add_ts", None),
                )
                del state.positions[sym]
                continue

            # Take-profit
            if px >= pos.target:
                pnl = (px - pos.entry) * pos.qty
                state.cash += pos.qty * px
                trades.append(
                    Trade(
                        t=_now_iso(),
                        symbol=sym,
                        side="SELL",
                        qty=pos.qty,
                        price=px,
                        reason="TAKE_PROFIT",
                        pnl=pnl,
                    )
                )
                log(
                    f"[{self.cfg.bot_key}] TAKE_PROFIT SELL {pos.qty:.4f} {sym} @ {px:.4f} "
                    f"PnL={pnl:.2f}"
                )
                # Send SELL alert
                self._send_sell_alert(
                    symbol=sym,
                    qty=pos.qty,
                    price=px,
                    entry_price=pos.entry,
                    reason="TAKE_PROFIT",
                    entry_ts=getattr(pos, "last_add_ts", None),
                )
                del state.positions[sym]
                continue            # Phase 5: time stop (let time work, but not forever)
            held_days = _days_held(pos)
            if held_days >= self.cfg.time_stop_days:
                pnl = (px - pos.entry) * pos.qty
                if pnl <= (0.01 * pos.entry * pos.qty):
                    state.cash += pos.qty * px
                    trades.append(
                        Trade(
                            t=_now_iso(),
                            symbol=sym,
                            side="SELL",
                            qty=pos.qty,
                            price=px,
                            reason=f"time_stop(days={held_days})",
                            confidence=1.0,
                            pnl=pnl,
                        )
                    )
                    log(f"[{self.cfg.bot_key}] TIME_STOP SELL {pos.qty:.4f} {sym} @ {px:.4f} PnL={pnl:.2f}")
                    # Send SELL alert
                    self._send_sell_alert(
                        symbol=sym,
                        qty=pos.qty,
                        price=px,
                        entry_price=pos.entry,
                        reason=f"time_stop(days={held_days})",
                        entry_ts=getattr(pos, "last_add_ts", None),
                    )
                    del state.positions[sym]
                    continue            # Phase 5: less twitchy exits — require hold time + confirmation
            node = rolling.get(sym) or {}
            intent, pol_conf, _ = self._extract_policy_signal(node)
            if intent == "SELL" and pol_conf >= self.cfg.conf_threshold:
                held_days = _days_held(pos)
                if held_days < self.cfg.min_hold_days:
                    pos.pending_exit = 0
                elif pol_conf < (self.cfg.conf_threshold + self.cfg.exit_conf_buffer):
                    pos.pending_exit = 0
                else:
                    pos.pending_exit = int(getattr(pos, "pending_exit", 0) or 0) + 1
                    pos.last_exit_signal_ts = _now_iso()
                    if pos.pending_exit >= self.cfg.exit_confirmations:
                        pnl = (px - pos.entry) * pos.qty
                        state.cash += pos.qty * px
                        trades.append(
                            Trade(
                                t=_now_iso(),
                                symbol=sym,
                                side="SELL",
                                qty=pos.qty,
                                price=px,
                                reason=f"ai_sell_confirmed({pos.pending_exit}/{self.cfg.exit_confirmations})",
                                confidence=pol_conf,
                                pnl=pnl,
                            )
                        )
                        log(f"[{self.cfg.bot_key}] AI_CONFIRM SELL {pos.qty:.4f} {sym} @ {px:.4f} PnL={pnl:.2f}")
                        # Send SELL alert
                        self._send_sell_alert(
                            symbol=sym,
                            qty=pos.qty,
                            price=px,
                            entry_price=pos.entry,
                            reason=f"ai_sell_confirmed({pos.pending_exit}/{self.cfg.exit_confirmations})",
                            entry_ts=getattr(pos, "last_add_ts", None),
                        )
                        del state.positions[sym]
                        continue
            else:
                pos.pending_exit = 0

        state.last_equity = self.compute_equity(state, prices)
        state.last_updated = _now_iso()
        return trades

    # --------------------------- Logging -------------------------- #

    def append_trades_to_daily_log(self, trades: List[Trade]) -> None:
        if not trades:
            return
        self.bot_log_dir.mkdir(parents=True, exist_ok=True)
        day = _today()
        path = self.bot_log_dir / f"bot_activity_{day}.json"

        data = _read_json(path) or {}
        arr = data.get(self.cfg.bot_key, [])
        if not isinstance(arr, list):
            arr = []
        for t in trades:
            arr.append(asdict(t))
        data[self.cfg.bot_key] = arr

        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
        log(f"[{self.cfg.bot_key}] 📓 Logged {len(trades)} trades to {path}")

    # --------------------------- Orchestration -------------------- #

    def run_full(self) -> None:
        """Premarket FULL rebalance."""
        rolling = self.load_rolling()
        if not rolling:
            log(f"[{self.cfg.bot_key}] ⚠️ No rolling data — aborting FULL rebalance.")
            return

        # Phase 3: regime-based risk adjustments (no hard blocks).
        # We tune sizing + risk rails based on the global regime label.
        try:
            prof = self._regime_risk_profile(rolling)
            # Apply multiplicatively each cycle (idempotent-ish because we base off the original defaults).
            base_max_pos = int(getattr(self.cfg, "_base_max_positions", self.cfg.max_positions))
            base_max_w = float(getattr(self.cfg, "_base_max_weight_per_name", self.cfg.max_weight_per_name))
            base_sl = float(getattr(self.cfg, "_base_stop_loss_pct", self.cfg.stop_loss_pct))
            base_tp = float(getattr(self.cfg, "_base_take_profit_pct", self.cfg.take_profit_pct))

            # Stash originals once so repeated cycles don't drift.
            setattr(self.cfg, "_base_max_positions", base_max_pos)
            setattr(self.cfg, "_base_max_weight_per_name", base_max_w)
            setattr(self.cfg, "_base_stop_loss_pct", base_sl)
            setattr(self.cfg, "_base_take_profit_pct", base_tp)

            self.cfg.max_positions = max(1, int(round(base_max_pos * float(prof.get("pos_mult") or 1.0))))
            self.cfg.max_weight_per_name = max(0.01, min(1.0, base_max_w * float(prof.get("weight_mult") or 1.0)))

            # Wider stops in hostile regimes; slightly smaller targets when conditions are ugly.
            self.cfg.stop_loss_pct = float(base_sl) * float(prof.get("stop_mult") or 1.0)
            self.cfg.take_profit_pct = float(base_tp) * float(prof.get("tp_mult") or 1.0)

            # Optional: drop a tiny breadcrumb for debugging/telemetry.
            g = rolling.get("_GLOBAL") if isinstance(rolling.get("_GLOBAL"), dict) else {}
            gprof = {k: v for k, v in prof.items() if k != "label"}
            gprof["label"] = prof.get("label")
            g["swing_risk_profile"] = gprof
            rolling["_GLOBAL"] = g
        except Exception:
            pass

        insights = self.load_insights()
        # Phase 5: regime playbook profile (soft gates + tier thresholds)
        tier_params: dict = {}
        try:
            g = rolling.get("_GLOBAL") if isinstance(rolling.get("_GLOBAL"), dict) else {}
            reg_label = str(
                g.get("regime_label")
                or g.get("market_regime")
                or g.get("regime")
                or (g.get("regime_state") or {}).get("label")
                or ""
            )
            playbook = load_swing_profile(reg_label)

            # Avoid permanent drift across cycles
            base_conf_th = float(getattr(self.cfg, "_base_conf_threshold", self.cfg.conf_threshold))
            base_low_frac = float(getattr(self.cfg, "_base_low_conf_max_fraction", self.cfg.low_conf_max_fraction))
            setattr(self.cfg, "_base_conf_threshold", base_conf_th)
            setattr(self.cfg, "_base_low_conf_max_fraction", base_low_frac)

            if isinstance(playbook, dict):
                if playbook.get("conf_threshold") is not None:
                    self.cfg.conf_threshold = float(playbook.get("conf_threshold"))
                if playbook.get("low_conf_max_fraction") is not None:
                    self.cfg.low_conf_max_fraction = float(playbook.get("low_conf_max_fraction"))

                g["swing_playbook_profile"] = {"name": str(playbook.get("label") or ""), "ts": _now_iso()}
                rolling["_GLOBAL"] = g

                if isinstance(playbook.get("tier_overrides"), dict):
                    tier_params["tier_overrides"] = playbook.get("tier_overrides")
                if playbook.get("max_vol") is not None:
                    tier_params["max_vol"] = float(playbook.get("max_vol"))
        except Exception:
            tier_params = {}

        ranked = rank_universe_v2(
            rolling=rolling,
            insights=insights,
            horizon=self.cfg.horizon,
            conf_threshold=self.cfg.conf_threshold,
            tier_params=tier_params if isinstance(tier_params, dict) and tier_params else None,
        )
        
        # Track universe stats for summary (approximate since rank_universe_v2 doesn't return rejection details)
        self._last_universe_analyzed = len(rolling) - 1 if "_GLOBAL" in rolling else len(rolling)  # Exclude _GLOBAL
        self._last_universe_qualified = len(ranked)
        # Note: Detailed rejection counts come from build_ai_ranked_universe if used,
        # otherwise we'll show simplified stats in the summary
        
        target_weights = self.construct_target_weights(ranked)

        state = self.load_bot_state()
        trades = self.rebalance_full(state, rolling, target_weights)
        self.save_bot_state(state)
        self.append_trades_to_daily_log(trades)
        
        # Send rebalance summary alert to Slack
        self._send_rebalance_summary(
            trades=trades,
            universe_analyzed=self._last_universe_analyzed,
            universe_qualified=self._last_universe_qualified,
            rejection_counts=getattr(self, "_last_rejection_counts", {}),
            state=state,
        )
        
        log(
            f"[{self.cfg.bot_key}] ✅ FULL rebalance complete. "
            f"Trades={len(trades)}"
        )
        
        # SSE Broadcast Note:
        # Rebalance completion triggers file updates:
        # - rolling_<bot_key>.json.gz (bot state)
        # - bot_logs/<horizon>/bot_activity_*.json (trade log)
        # SSE polling in events_router.py (/events/bots) picks up changes every 5 seconds
        # and pushes fresh data to connected clients. Client-side cache auto-invalidates
        # on SSE push for instant UI updates without backend caching.

    def run_loop(self) -> None:
        """Intraday LOOP — risk checks only."""
        rolling = self.load_rolling()
        if not rolling:
            log(f"[{self.cfg.bot_key}] ⚠️ No rolling data — aborting LOOP check.")
            return

        state = self.load_bot_state()
        trades = self.apply_loop_risk_checks(state, rolling)
        self.save_bot_state(state)
        self.append_trades_to_daily_log(trades)
        log(
            f"[{self.cfg.bot_key}] ✅ LOOP risk-check complete. "
            f"Trades={len(trades)}"
        )

    def run(self, mode: str = "full") -> None:
        """
        Dispatch method used by runners.
        mode ∈ {"full", "loop"}
        """
        if not _bot_is_enabled(self.cfg.bot_key):
            log(f"[{self.cfg.bot_key}] ⏸ Bot disabled via UI — skipping run.")
            return

        if mode == "loop":
            self.run_loop()
        else:
            self.run_full()