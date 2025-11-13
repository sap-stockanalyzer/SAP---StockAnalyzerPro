# backend/trading_bot_simulator.py
# SimTrader v2.5 — Intraday multi-bot simulator with risk control + cumulative PnL
# - Reads 1-min bars from: data_dt/rolling_intraday.json.gz
# - Reads AI signals (best-effort) from: ml_data_dt/signals/
# - Simulates 5 bots: momentum, mean-revert, signal-follow, breakout, hybrid
# - Enforces stop-loss / take-profit per position
# - Logs daily trades and updates ml_data_dt/sim_summary.json
# - Can run standalone or be called from daytrading_job

from __future__ import annotations
import os, json, gzip, time, glob, threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# Reuse your logger + DT paths
try:
    from backend.data_pipeline import log  # unified logger
except Exception:
    def log(msg: str):  # tiny fallback
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

try:
    from dt_backend.config_dt import DT_PATHS
except Exception:
    # Fallback if imported differently
    from .config_dt import DT_PATHS  # type: ignore

# ==============================================================
# ---------- Safe Symbol Extractor ------------------------------
# ==============================================================

def _get_symbol(val: str | dict) -> str:
    """Normalize symbol/ticker field for consistency."""
    if isinstance(val, str):
        return val.strip().upper()
    if isinstance(val, dict):
        sym = val.get("symbol") or val.get("ticker") or val.get("name") or ""
        return str(sym).strip().upper()
    return ""

# ----------------------------- CONFIG --------------------------------

# Risk & money management (tune freely)
START_CASH          = 100.0
STOP_LOSS_PCT       = -0.02   # -2%
TAKE_PROFIT_PCT     =  0.04   # +4%
MAX_POSITIONS       = 10
POSITION_SIZE_PCT   = 0.20    # 20% of equity per trade
LOOP_SECONDS        = 60      # main loop interval for bar-driven checks

# Paths (DT_PATHS-aware)
BARS_PATH           = os.path.join("data_dt", "rolling_intraday.json.gz")  # your file
SIGNALS_DIR         = os.path.join("ml_data_dt", "signals")
SIM_LOG_DIR       = str(PATHS["ml_data_dt"] / "sim_logs")
SIM_SUMMARY_FILE  = str(PATHS["ml_data_dt"] / "sim_summary.json")

# Universe cap (optional — to keep sim light)
MAX_UNIVERSE        = 2000  # consider only the first N symbols that have bars

# ----------------------------------------------------------------------

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _today_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _ensure_dirs():
    os.makedirs(SIM_LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SIM_SUMMARY_FILE), exist_ok=True)

def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_latest_bars(path: str = BARS_PATH) -> Dict[str, dict]:
    """
    Loads your rolling intraday bars and returns a dict:
    { "AAPL": {"price": 188.05, "volume": 412562, "time": "...", "last_n": [closes...]}, ... }
    """
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            js = json.load(f)
    except Exception as e:
        log(f"⚠️ Failed to load intraday bars: {e}")
        return {}

    bars = js.get("bars", {}) or {}
    out = {}
    for sym, arr in bars.items():
        if not arr:
            continue
        last = arr[-1]
        # collect a small window of closes/highs/lows for bar logic
        closes = [float(x.get("c", 0.0)) for x in arr[-60:]]  # last 60 mins
        highs  = [float(x.get("h", 0.0)) for x in arr[-60:]]
        lows   = [float(x.get("l", 0.0)) for x in arr[-60:]]
        out[sym] = {
            "price": float(last.get("c", 0.0)),
            "volume": int(last.get("v", 0) or 0),
            "time": last.get("t"),
            "closes": closes,
            "highs": highs,
            "lows": lows
        }
        if len(out) >= MAX_UNIVERSE:
            break
    return out

def _latest_file(pattern: str) -> Optional[str]:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

def load_latest_signals(signals_dir: str = SIGNALS_DIR) -> Dict[str, dict]:
    """
    Best-effort loader that tolerates different shapes:
    - dict: {"AAPL": {"signal": "BUY", "confidence": 0.74, "price": 189.2}, ...}
    - list: [{"ticker":"AAPL","signal":"BUY","confidence":0.7,...}, ...]
    - top-picks array with "ticker"/"rankingScore" fields
    """
    try:
        # try common names in descending preference
        candidates = [
            _latest_file(os.path.join(signals_dir, "*.json")),
            _latest_file(os.path.join(signals_dir, "*.gz")),  # if compressed json
        ]
        path = next((p for p in candidates if p), None)
        if not path:
            return {}
        data = None
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        out = {}
        if isinstance(data, dict):
            # shape: { "AAPL": {...}, "NVDA": {...} }
            for k, v in data.items():
                if isinstance(v, dict):
                    sig = (v.get("signal") or v.get("action") or "").upper()
                    out[k.upper()] = {
                        "signal": sig if sig in ("BUY","SELL") else None,
                        "confidence": float(v.get("confidence") or v.get("rankScore") or 0.0),
                        "price": float(v.get("price") or v.get("currentPrice") or 0.0),
                    }
        elif isinstance(data, list):
            for r in data:
                if not isinstance(r, dict):
                    continue
                t = (r.get("ticker") or r.get("symbol") or "").upper()
                sig = (r.get("signal") or r.get("action") or "").upper()
                out[t] = {
                    "signal": sig if sig in ("BUY","SELL") else None,
                    "confidence": float(r.get("confidence") or r.get("rankingScore") or 0.0),
                    "price": float(r.get("price") or r.get("currentPrice") or 0.0),
                }
        return out
    except Exception:
        return {}

# ----------------------------- TECHNICAL UTILS -------------------------------

def _sma(vals: List[float], n: int) -> Optional[float]:
    vals = [v for v in vals if isinstance(v, (int,float))]
    if len(vals) < n or n <= 0:
        return None
    return sum(vals[-n:]) / float(n)

def _ema(vals: List[float], n: int) -> Optional[float]:
    vals = [v for v in vals if isinstance(v, (int,float))]
    if len(vals) < n or n <= 0:
        return None
    k = 2 / (n + 1.0)
    ema = vals[-n]
    for v in vals[-n+1:]:
        ema = v * k + ema * (1 - k)
    return ema

def _pct_chg(a: float, b: float) -> Optional[float]:
    try:
        if b == 0:
            return None
        return (a - b) / b
    except Exception:
        return None

# ----------------------------- CORE SIM ENGINE -------------------------------

class BaseSimBot:
    def __init__(self, name: str):
        self.name = name
        self.cash = START_CASH
        self.positions: Dict[str, dict] = {}  # {sym: {"qty": float, "entry": float, "stop": float, "target": float}}
        self.trades: List[dict] = []
        self.daily_equity: List[Tuple[str, float]] = []  # (iso_ts, equity)
        self.max_positions = MAX_POSITIONS

    # ------- portfolio helpers -------
    def equity(self, price_map: Dict[str, float]) -> float:
        value = self.cash
        for sym, pos in self.positions.items():
            px = price_map.get(sym)
            if isinstance(px, (int,float)):
                value += pos["qty"] * px
        return float(value)

    def _position_size(self, equity_now: float, price: float) -> float:
        alloc_cash = max(0.0, equity_now * POSITION_SIZE_PCT)
        return round(alloc_cash / max(price, 1e-6), 6)

    def _enter(self, sym: str, price: float, reason: str):
        # Restrict trading to regular market hours
        import pytz
        from datetime import datetime, time as dtime
        ny = pytz.timezone("America/New_York")
        now = datetime.now(ny)
        if now.weekday() >= 5 or not (dtime(9,30) <= now.time() <= dtime(16,0)):
            log(f"[{self.name}] ⏸ Market closed — skipping buy for {sym}.")
            return

        if len(self.positions) >= self.max_positions:
            return
        eq = self.equity({})
        qty = self._position_size(self.equity({}), price)
        cost = qty * price
        if qty <= 0 or cost > self.cash:
            return
        stop = price * (1.0 + STOP_LOSS_PCT)
        target = price * (1.0 + TAKE_PROFIT_PCT)
        self.cash -= cost
        self.positions[sym] = {"qty": qty, "entry": price, "stop": stop, "target": target}
        self.trades.append({
            "t": _now_utc_iso(),
            "ticker": sym,
            "action": "BUY",
            "qty": qty,
            "price": price,
            "reason": reason
        })

    def _exit(self, sym: str, price: float, reason: str):
        # Restrict trading to regular market hours
        import pytz
        from datetime import datetime, time as dtime
        ny = pytz.timezone("America/New_York")
        now = datetime.now(ny)
        if now.weekday() >= 5 or not (dtime(9,30) <= now.time() <= dtime(16,0)):
            log(f"[{self.name}] ⏸ Market closed — skipping sell for {sym}.")
            return

        pos = self.positions.get(sym)
        if not pos:
            return
        proceeds = pos["qty"] * price
        self.cash += proceeds
        pnl = (price - pos["entry"]) * pos["qty"]
        self.trades.append({
            "t": _now_utc_iso(),
            "ticker": sym,
            "action": "SELL",
            "qty": pos["qty"],
            "price": price,
            "reason": reason,
            "pnl": round(pnl, 6)
        })
        del self.positions[sym]

    def risk_checks(self, sym: str, price: float):
        pos = self.positions.get(sym)
        if not pos:
            return
        if price <= pos["stop"]:
            self._exit(sym, price, "STOP_LOSS")
        elif price >= pos["target"]:
            self._exit(sym, price, "TAKE_PROFIT")

    # ------- strategy hooks (override) -------
    def maybe_enter(self, sym: str, bar: dict, signals: dict):  # decide opens
        pass

    def maybe_exit(self, sym: str, bar: dict, signals: dict):   # decide discretionary exits
        pass

    # ------- step with latest bars -------
    def step(self, market: Dict[str, dict], signals: Dict[str, dict]):
        # price_map for equity calc
        price_map = {s: d.get("price") for s, d in market.items() if isinstance(d.get("price"), (int,float))}
        ts = _now_utc_iso()
        # exits/SL-TP
        for sym in list(self.positions.keys()):
            px = price_map.get(sym)
            if not isinstance(px, (int,float)):
                continue
            self.risk_checks(sym, px)
            # discretionary exit
            self.maybe_exit(sym, market.get(sym) or {}, signals)

        # entries
        for sym, d in market.items():
            # skip if already long
            if sym in self.positions:
                continue
            self.maybe_enter(sym, d, signals)

        # record equity snapshot
        self.daily_equity.append((ts, self.equity(price_map)))

# ----------------------------- STRATEGIES ------------------------------------

class MomentumBot(BaseSimBot):
    def __init__(self):
        super().__init__("momentum_bot")

    def maybe_enter(self, sym, bar, signals):
        closes = bar.get("closes") or []
        if len(closes) < 20:
            return
        ema5 = _ema(closes, 5)
        ema20 = _ema(closes, 20)
        price = float(closes[-1])
        if ema5 is None or ema20 is None:
            return
        # bullish momentum + above ema20
        if ema5 > ema20 and price > ema20:
            self._enter(sym, price, "EMA5>EMA20")

    def maybe_exit(self, sym, bar, signals):
        closes = bar.get("closes") or []
        if len(closes) < 20 or sym not in self.positions:
            return
        ema5 = _ema(closes, 5)
        ema20 = _ema(closes, 20)
        price = float(closes[-1])
        if ema5 is None or ema20 is None:
            return
        # momentum fade
        if ema5 < ema20:
            self._exit(sym, price, "MOMENTUM_FADE")

class MeanRevertBot(BaseSimBot):
    def __init__(self):
        super().__init__("mean_revert_bot")

    def maybe_enter(self, sym, bar, signals):
        closes = bar.get("closes") or []
        if len(closes) < 30:
            return
        ma20 = _sma(closes, 20)
        price = float(closes[-1])
        if ma20 is None or ma20 == 0:
            return
        drop = _pct_chg(price, ma20)  # current vs mean
        # buy ~1–2% below mean
        if isinstance(drop, float) and drop <= -0.012:
            self._enter(sym, price, "MEAN_REVERT_BUY")

    def maybe_exit(self, sym, bar, signals):
        closes = bar.get("closes") or []
        if len(closes) < 20 or sym not in self.positions:
            return
        ma20 = _sma(closes, 20)
        price = float(closes[-1])
        if ma20 is None:
            return
        # mean reversion realized
        if price >= ma20:
            self._exit(sym, price, "MEAN_TOUCHED")

class SignalFollowBot(BaseSimBot):
    def __init__(self):
        super().__init__("signal_follow_bot")

    def maybe_enter(self, sym, bar, signals):
        rec = signals.get(sym) or {}
        sig = (rec.get("signal") or "").upper()
        price = float(bar.get("closes", [bar.get("price")])[-1])
        if sig == "BUY":
            self._enter(sym, price, "AI_SIGNAL_BUY")

    def maybe_exit(self, sym, bar, signals):
        rec = signals.get(sym) or {}
        sig = (rec.get("signal") or "").upper()
        price = float(bar.get("closes", [bar.get("price")])[-1])
        if sig == "SELL":
            self._exit(sym, price, "AI_SIGNAL_SELL")

class BreakoutBot(BaseSimBot):
    def __init__(self):
        super().__init__("breakout_bot")

    def maybe_enter(self, sym, bar, signals):
        highs = bar.get("highs") or []
        lows = bar.get("lows") or []
        closes = bar.get("closes") or []
        if len(highs) < 21 or len(lows) < 21 or not closes:
            return
        price = float(closes[-1])
        hi20 = max(highs[-21:-1]) if len(highs) >= 21 else None
        lo20 = min(lows[-21:-1]) if len(lows) >= 21 else None
        if hi20 is None or lo20 is None:
            return
        if price > hi20:
            self._enter(sym, price, "BREAKOUT_UP")

    def maybe_exit(self, sym, bar, signals):
        lows = bar.get("lows") or []
        closes = bar.get("closes") or []
        if len(lows) < 20 or not closes or sym not in self.positions:
            return
        price = float(closes[-1])
        lo20 = min(lows[-20:])
        # break below recent floor
        if price < lo20:
            self._exit(sym, price, "BREAKDOWN_EXIT")

class HybridBot(BaseSimBot):
    """
    Fusion of the 4 strategies:
      - Requires AI BUY OR strong momentum filter
      - Bias long only when price above MA50
      - Avoids entries during high micro drawdown vs MA20
      - Discretionary exit on momentum fade OR AI SELL
    """
    def __init__(self):
        super().__init__("hybrid_bot")

    def maybe_enter(self, sym, bar, signals):
        closes = bar.get("closes") or []
        if len(closes) < 50:
            return
        price = float(closes[-1])
        ema5 = _ema(closes, 5)
        ema20 = _ema(closes, 20)
        ma50 = _sma(closes, 50)
        ai = (signals.get(sym) or {}).get("signal")
        ai_buy = (isinstance(ai, str) and ai.upper() == "BUY")
        if None in (ema5, ema20, ma50):
            return

        # momentum strength
        mom_ok = (ema5 > ema20) and (price > ema20)
        # trend bias
        trend_ok = price > ma50
        # avoid entries if sharply extended below mean (drawdown filter)
        dd = _pct_chg(price, ema20)
        safe = True if dd is None else (dd > -0.03)

        if (ai_buy or mom_ok) and trend_ok and safe:
            self._enter(sym, price, "HYBRID_ENTRY")

    def maybe_exit(self, sym, bar, signals):
        closes = bar.get("closes") or []
        if len(closes) < 20 or sym not in self.positions:
            return
        price = float(closes[-1])
        ema5 = _ema(closes, 5)
        ema20 = _ema(closes, 20)
        ai = (signals.get(sym) or {}).get("signal")
        ai_sell = (isinstance(ai, str) and ai.upper() == "SELL")
        if None in (ema5, ema20):
            return
        if ai_sell or ema5 < ema20:
            self._exit(sym, price, "HYBRID_EXIT")

# ----------------------------- LOGGING / SUMMARY -----------------------------

def _save_daily_log(bot: BaseSimBot, date_tag: str):
    path = os.path.join(SIM_LOG_DIR, f"trading_log_{bot.name}_{date_tag}.json")
    out = {
        "date": date_tag,
        "bot": bot.name,
        "start_cash": START_CASH,
        "end_equity": bot.daily_equity[-1][1] if bot.daily_equity else bot.cash,
        "trades": bot.trades,
        "positions_end": bot.positions,
        "equity_series": bot.daily_equity[-200:],  # last 200 snaps
    }
    _write_json(path, out)

def _update_summary(bots: List[BaseSimBot], date_tag: str):
    summary = _read_json(SIM_SUMMARY_FILE) or {}
    for bot in bots:
        end_equity = bot.daily_equity[-1][1] if bot.daily_equity else bot.cash
        pnl = end_equity - START_CASH
        pnl_pct = (pnl / START_CASH) * 100.0 if START_CASH else 0.0
        node = summary.get(bot.name) or {"total_days": 0, "cumulative_pnl": 0.0, "avg_pnl_pct": 0.0, "total_trades": 0}
        node["total_days"] = int(node.get("total_days", 0)) + 1
        node["cumulative_pnl"] = float(node.get("cumulative_pnl", 0.0)) + float(pnl)
        # rolling average
        node["avg_pnl_pct"] = round((node["avg_pnl_pct"] * (node["total_days"] - 1) + pnl_pct) / node["total_days"], 4)
        node["last_day"] = {"date": date_tag, "pnl": round(pnl, 4), "pnl_pct": round(pnl_pct, 4)}
        node["total_trades"] = int(node.get("total_trades", 0)) + len(bot.trades)
        summary[bot.name] = node
    _write_json(SIM_SUMMARY_FILE, summary)

# ----------------------------- RUNNER ----------------------------------------

def _loop_once(bots: List[BaseSimBot]) -> bool:
    """
    One iteration: read bars + signals, step all bots. Return True if market present.
    """
    market = {_get_symbol(s): d for s, d in load_latest_bars(BARS_PATH).items()}
    signals = {_get_symbol(s): d for s, d in load_latest_signals(SIGNALS_DIR).items()}

    market = load_latest_bars(BARS_PATH)
    if not market:
        log("ℹ️ No bars found — skipping loop.")
        return False
    signals = load_latest_signals(SIGNALS_DIR)

    # Step bots
    for bot in bots:
        try:
            bot.step(market, signals)
        except Exception as e:
            log(f"⚠️ Bot step failed ({bot.name}): {e}")

    return True

def run_all_bots(run_minutes: Optional[int] = None):
    """
    Main loop. If run_minutes provided, loops that many minutes; else runs forever.
    Use from daytrading_job or standalone.
    """
    _ensure_dirs()
    date_tag = _today_tag()

    bots: List[BaseSimBot] = [
        MomentumBot(),
        MeanRevertBot(),
        SignalFollowBot(),
        BreakoutBot(),
        HybridBot(),
    ]
    log(f"🤖 SimTrader started — bots: {[b.name for b in bots]} | start_cash=${START_CASH:.2f}")

    loops = 0
try:
    while True:
        # stop running completely outside market hours
        if not is_market_open():
            log("⏹ Market closed — stopping day-trading bots until next session.")
            break

        loops += 1
        market_ok = _loop_once(bots)

        # Save equity checkpoint after each loop to keep logs responsive
        if market_ok:
            for bot in bots:
                # top-of-loop equity snapshot already appended inside bot.step()
                pass

        # bounded run (useful for tests) or forever
        if run_minutes is not None and loops >= max(1, int(run_minutes)):
            break

        time.sleep(LOOP_SECONDS)
except KeyboardInterrupt:
    log("🛑 SimTrader interrupted by user.")
except Exception as e:
    log(f"⚠️ SimTrader crashed: {e}")
finally:
    log("✅ SimTrader loop ended (market closed or run complete).")

    finally:
        # End-of-day (or end-of-run) logging + summary
        for bot in bots:
            if not bot.daily_equity:
                # ensure at least one equity point
                market = load_latest_bars(BARS_PATH)
                price_map = {s: d.get("price") for s, d in (market or {}).items() if isinstance(d.get("price"), (int,float))}
                bot.daily_equity.append((_now_utc_iso(), bot.equity(price_map)))
            _save_daily_log(bot, date_tag)

        _update_summary(bots, date_tag)
        log("✅ SimTrader logs & summary updated.")

# ----------------------------- ENTRYPOINT ------------------------------------

if __name__ == "__main__":
    # Run indefinitely; press Ctrl+C to stop.
    run_all_bots(run_minutes=None)
