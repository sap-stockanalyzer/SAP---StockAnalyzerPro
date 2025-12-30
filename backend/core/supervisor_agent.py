# backend/core/supervisor_agent.py — v1.2
"""
Supervisor Agent — AION Analytics (System Health + Truth Loop + Overrides)

Responsibilities:
    • Inspect health of:
        - ML dataset (training_data_daily.parquet + feature_list_daily.json)
        - Models per horizon (regressor_{h}.pkl preferred; legacy model_{h}.pkl supported)
        - Macro / news / social intel
        - Insights artifacts (top50_*.json)
        - Drift (continuous_learning meta + brain horizon drift)
        - Rolling coverage (predictions/context/news/social)
        - Prediction dispersion (predicted_return variability)
        - Sector performance snapshots (from continuous_learning brain meta)
        - Execution performance snapshot (from performance_aggregator)
        - AION brain freshness + meta sanity (aion_brain.json.gz)

    • Maintain supervisor_overrides.json:
        - kill_switch
        - conf_min
        - exposure_cap

Public:
    - supervisor_verdict()     → main status payload (for API/routers)
    - run_supervisor_agent()   → nightly_job wrapper
    - update_overrides(metrics) → back-compat alias to compute_overrides_from_metrics
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import json
import datetime
import os
import time
from pathlib import Path
from statistics import mean, pstdev

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import _read_rolling, _read_brain, _read_aion_brain, safe_float
from utils.logger import log

# -------------------------------------------------------------
# Logging controls (to avoid log spam when UI polls /api/system/status)
# -------------------------------------------------------------

# Set AION_SUPERVISOR_VERBOSE=1 to log every evaluation/verdict call
AION_SUPERVISOR_VERBOSE = os.getenv('AION_SUPERVISOR_VERBOSE', '0').strip().lower() in {'1','true','yes','y','on'}

# By default, only log when the overall verdict / overrides change (or once per minute)
AION_SUPERVISOR_LOG_ON_CHANGE = os.getenv('AION_SUPERVISOR_LOG_ON_CHANGE', '1').strip().lower() in {'1','true','yes','y','on'}
AION_SUPERVISOR_LOG_EVERY_SECS = float(os.getenv('AION_SUPERVISOR_LOG_EVERY_SECS', '0') or '0')

_LAST_SUPERVISOR_LOG_TS = 0.0
_LAST_SUPERVISOR_SIG = ''


def _maybe_log_supervisor(msg: str, sig: str = '') -> None:
    """Throttled supervisor logging to keep stdout usable."""
    global _LAST_SUPERVISOR_LOG_TS, _LAST_SUPERVISOR_SIG
    if AION_SUPERVISOR_VERBOSE:
        log(msg)
        _LAST_SUPERVISOR_LOG_TS = time.time()
        _LAST_SUPERVISOR_SIG = sig or _LAST_SUPERVISOR_SIG
        return

    now = time.time()
    changed = bool(sig) and (sig != _LAST_SUPERVISOR_SIG)

    if AION_SUPERVISOR_LOG_ON_CHANGE and changed:
        log(msg)
        _LAST_SUPERVISOR_LOG_TS = now
        _LAST_SUPERVISOR_SIG = sig
        return

    # Optional heartbeat log (default: once per 60s). Set to 0 to fully silence.
    if AION_SUPERVISOR_LOG_EVERY_SECS > 0 and (now - _LAST_SUPERVISOR_LOG_TS) >= AION_SUPERVISOR_LOG_EVERY_SECS:
        log(msg)
        _LAST_SUPERVISOR_LOG_TS = now
        if sig:
            _LAST_SUPERVISOR_SIG = sig


def _latest_with_prefix(dirpath: Path, prefix: str) -> Path | None:
    try:
        if not dirpath.exists():
            return None
        matches = sorted(dirpath.glob(prefix + "*"), key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0] if matches else None
    except Exception:
        return None


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ML_DATA_ROOT: Path = PATHS.get("ml_data", Path("ml_data"))

# Dataset (aligned with ml_data_builder v6+)
DATASET_DIR: Path = ML_DATA_ROOT / "nightly" / "dataset"
DATASET_FILE: Path = DATASET_DIR / "training_data_daily.parquet"
FEATURE_LIST_FILE: Path = DATASET_DIR / "feature_list_daily.json"

# Models (aligned with ai_model.py)
MODEL_ROOT: Path = PATHS.get("ml_models", ML_DATA_ROOT / "nightly" / "models")

# Intel
INSIGHTS_DIR: Path = PATHS.get("insights", PATHS.get("root", Path(".")) / "insights")
MACRO_STATE_FILE: Path = PATHS.get("macro_state", ML_DATA_ROOT / "macro_state.json")
NEWS_INTEL_FILE: Path = PATHS.get("news_intel", (ML_DATA_ROOT / "news_features" / "news_features_latest.json"))
SOCIAL_INTEL_FILE: Path = PATHS.get("social_intel", PATHS.get("analytics", Path("analytics")) / "social_intel.json")

# Drift / metrics
METRICS_ROOT: Path = ML_DATA_ROOT / "metrics"
DRIFT_DIR: Path = METRICS_ROOT / "drift"
DRIFT_REPORT_FILE: Path = DRIFT_DIR / "drift_report.json"

# Execution performance (new)
PERF_FILE: Path = ML_DATA_ROOT / "performance" / "system_perf.json"

# AION brain (new)
AION_BRAIN_FILE: Optional[Path] = PATHS.get("aion_brain")

# Overrides file
OVERRIDES_PATH: Path = PATHS["ml_data"] / "supervisor_overrides.json"

# Nightly horizons (must match ai_model / policy_engine)
HORIZONS: List[str] = ["1d", "3d", "1w", "2w", "4w", "13w", "26w", "52w"]


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _file_age_minutes(path: Path) -> float:
    """Age of file in minutes; large value if missing."""
    try:
        if not path.exists():
            return 9999.0
        m = path.stat().st_mtime
        dt = datetime.datetime.fromtimestamp(m, tz=TIMEZONE)
        diff = datetime.datetime.now(TIMEZONE) - dt
        return diff.total_seconds() / 60.0
    except Exception:
        return 9999.0


def _status_from_age(age_min: float, warn: float, crit: float) -> str:
    if age_min >= crit:
        return "critical"
    if age_min >= warn:
        return "warning"
    return "ok"


def _save_overrides(js: dict) -> None:
    """Write overrides, but avoid needless rewrites (helps when status is polled frequently)."""
    try:
        OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Skip write if unchanged
        if OVERRIDES_PATH.exists():
            try:
                existing = json.loads(OVERRIDES_PATH.read_text(encoding='utf-8'))
                if existing == js:
                    return
            except Exception:
                pass

        with open(OVERRIDES_PATH, 'w', encoding='utf-8') as f:
            json.dump(js, f, indent=2)
    except Exception as e:
        log(f"[supervisor_agent] ⚠️ Failed to save overrides: {e}")


def load_overrides() -> Dict[str, Any]:
    """Load the last written supervisor overrides (if any)."""
    try:
        if not OVERRIDES_PATH.exists():
            return {}
        return json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _model_path_candidates(h: str) -> List[Path]:
    """
    Preferred naming matches backend/core/ai_model.py training outputs:
      - regressor_{h}.pkl
    Legacy naming supported for older runs:
      - model_{h}.pkl
    """
    return [
        MODEL_ROOT / f"regressor_{h}.pkl",
        MODEL_ROOT / f"model_{h}.pkl",
    ]


def _resolve_existing_model_path(h: str) -> Tuple[bool, Path]:
    """
    Return (exists, chosen_path). Chooses first existing candidate, else preferred candidate.
    """
    candidates = _model_path_candidates(h)
    for p in candidates:
        if p.exists():
            return True, p
    return False, candidates[0]


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------
# Core health checks
# ---------------------------------------------------------------------

def _check_dataset_health() -> Dict[str, Any]:
    age = _file_age_minutes(DATASET_FILE)
    feat_age = _file_age_minutes(FEATURE_LIST_FILE)

    # Expect dataset to refresh at least once per day
    if age >= 1440 or feat_age >= 1440:  # 24h
        status = "critical"
    elif age >= 720 or feat_age >= 720:  # 12h
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "dataset_age_min": age,
        "feature_list_age_min": feat_age,
        "dataset_path": str(DATASET_FILE),
        "feature_list_path": str(FEATURE_LIST_FILE),
    }


def _check_model_health() -> Dict[str, Any]:
    """Check that each horizon has a model and how old they are."""
    horizon_info: Dict[str, Any] = {}
    ages: List[float] = []

    for h in HORIZONS:
        exists, mp = _resolve_existing_model_path(h)
        age = _file_age_minutes(mp)
        ages.append(age if exists else 9999.0)

        horizon_info[h] = {
            "exists": exists,
            "age_min": age,
            "path": str(mp),
            "candidates": [str(p) for p in _model_path_candidates(h)],
        }

    missing = [h for h, info in horizon_info.items() if not info["exists"]]
    oldest_age = max(ages) if ages else 9999.0

    # Expect models to be refreshed at least weekly, ideally daily
    if missing or oldest_age >= 10080:  # >= 7 days
        status = "critical"
    elif oldest_age >= 4320:  # >= 3 days
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "missing_horizons": missing,
        "oldest_model_age_min": oldest_age,
        "horizons": horizon_info,
    }


def _check_intel_files() -> Dict[str, Any]:
    macro_age = _file_age_minutes(MACRO_STATE_FILE)
    news_age = _file_age_minutes(NEWS_INTEL_FILE)
    soc_age = _file_age_minutes(SOCIAL_INTEL_FILE)

    macro_status = _status_from_age(macro_age, warn=720, crit=1440)
    news_status = _status_from_age(news_age, warn=240, crit=720)
    soc_status = _status_from_age(soc_age, warn=240, crit=720)

    return {
        "macro": {"status": macro_status, "age_min": macro_age, "path": str(MACRO_STATE_FILE)},
        "news_intel": {"status": news_status, "age_min": news_age, "path": str(NEWS_INTEL_FILE)},
        "social_intel": {"status": soc_status, "age_min": soc_age, "path": str(SOCIAL_INTEL_FILE)},
    }


def _check_insights_health() -> Dict[str, Any]:
    """Look at canonical insights files."""
    files = [
        "top50_1w.json",
        "top50_2w.json",
        "top50_4w.json",
        "top50_52w.json",
        "top50_social_heat.json",
        "top50_news_novelty.json",
    ]
    info: Dict[str, Any] = {}
    ages: List[float] = []

    for name in files:
        p = INSIGHTS_DIR / name
        age = _file_age_minutes(p)
        ages.append(age if p.exists() else 9999.0)
        info[name] = {"exists": p.exists(), "age_min": age, "path": str(p)}

    oldest_age = max(ages) if ages else 9999.0
    if oldest_age >= 10080:  # 7 days
        status = "critical"
    elif oldest_age >= 1440:  # 1 day
        status = "warning"
    else:
        status = "ok"

    return {"status": status, "oldest_insight_age_min": oldest_age, "files": info}


def _check_drift_health() -> Dict[str, Any]:
    """
    Drift is sourced from:
      - brain["_meta"]["horizon_drift"][h] (continuous_learning global summary, preferred)
      - drift_report.json drift_by_horizon (optional, supplemental)

    We DO NOT rely on legacy per-symbol drift_score.
    """
    try:
        brain = _read_brain() or {}
    except Exception:
        brain = {}

    meta = brain.get("_meta", {}) if isinstance(brain, dict) else {}
    horizon_drift = meta.get("horizon_drift") if isinstance(meta, dict) else {}
    if not isinstance(horizon_drift, dict):
        horizon_drift = {}

    # Compute global max/avg drift magnitude (negative is bad)
    vals = []
    worst_neg = 0.0
    for h, stats in horizon_drift.items():
        if not isinstance(stats, dict):
            continue
        d = safe_float(stats.get("avg_drift", 0.0))
        vals.append(d)
        worst_neg = min(worst_neg, d)

    avg_d = float(mean(vals)) if vals else 0.0
    worst_neg = float(worst_neg)

    # classify: degradation beyond ~1% is warning; beyond ~2% critical
    if worst_neg <= -0.02:
        status = "critical"
    elif worst_neg <= -0.01:
        status = "warning"
    else:
        status = "ok"

    drift_report = {}
    try:
        if DRIFT_REPORT_FILE.exists():
            drift_report = json.loads(DRIFT_REPORT_FILE.read_text(encoding="utf-8"))
    except Exception:
        drift_report = {}

    return {
        "status": status,
        "brain_horizon_drift": horizon_drift,
        "avg_drift": avg_d,
        "worst_negative_drift": worst_neg,
        "report_horizon_drift": drift_report.get("drift_by_horizon", {}),
    }


def _check_rolling_coverage() -> Dict[str, Any]:
    """
    Basic sanity: rolling present, predictions/context/news/social filled.
    """
    rolling = _read_rolling() or {}
    if not rolling:
        return {
            "status": "critical",
            "symbols": 0,
            "missing_predictions": 0,
            "missing_context": 0,
            "missing_news": 0,
            "missing_social": 0,
        }

    total = 0
    missing_preds = 0
    missing_ctx = 0
    missing_news = 0
    missing_social = 0

    for sym, node in rolling.items():
        if str(sym).startswith("_"):
            continue
        if not isinstance(node, dict):
            continue
        total += 1
        if not node.get("predictions"):
            missing_preds += 1
        if not node.get("context"):
            missing_ctx += 1
        if not node.get("news"):
            missing_news += 1
        if not node.get("social"):
            missing_social += 1

    ratio_missing_preds = missing_preds / max(total, 1)
    if ratio_missing_preds > 0.5:
        status = "critical"
    elif ratio_missing_preds > 0.1:
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "symbols": total,
        "missing_predictions": missing_preds,
        "missing_context": missing_ctx,
        "missing_news": missing_news,
        "missing_social": missing_social,
        "missing_preds_ratio": round(ratio_missing_preds, 4),
    }


def _check_prediction_dispersion() -> Dict[str, Any]:
    """
    Look at dispersion of predicted_return across symbols.
    If everything is near 0, something is wrong (flat model).
    """
    rolling = _read_rolling() or {}
    if not rolling:
        return {"status": "critical", "note": "no rolling", "per_horizon": {}}

    per_h: Dict[str, List[float]] = {h: [] for h in HORIZONS}
    for sym, node in rolling.items():
        if str(sym).startswith("_"):
            continue
        if not isinstance(node, dict):
            continue
        preds = node.get("predictions") or {}
        if not isinstance(preds, dict):
            continue
        for h in HORIZONS:
            blk = preds.get(h)
            if not isinstance(blk, dict):
                continue
            r = safe_float(blk.get("predicted_return", 0.0))
            per_h[h].append(r)

    out: Dict[str, Any] = {}
    worst_status = "ok"

    for h, vals in per_h.items():
        if not vals:
            out[h] = {"std": 0.0, "status": "warning", "note": "no predictions"}
            worst_status = "warning" if worst_status == "ok" else worst_status
            continue
        std = float(pstdev(vals)) if len(vals) > 1 else 0.0

        # If dispersion is tiny, model might be degenerate/flat
        if std < 0.002:  # <0.2% std in predicted_return
            status = "warning"
        else:
            status = "ok"
        out[h] = {"std": std, "status": status, "n": len(vals)}
        if status == "warning" and worst_status == "ok":
            worst_status = "warning"

    return {"status": worst_status, "per_horizon": out}


def _check_sector_perf() -> Dict[str, Any]:
    """
    Inspect sector_perf snapshot written by continuous_learning.
    Expected at: brain["_meta"]["sector_perf"][SECTOR][HORIZON] = {n, hit_ratio, mae, ...}
    """
    try:
        brain = _read_brain() or {}
    except Exception:
        brain = {}

    meta = brain.get("_meta", {}) if isinstance(brain, dict) else {}
    sector_perf = meta.get("sector_perf") if isinstance(meta, dict) else {}
    if not isinstance(sector_perf, dict) or not sector_perf:
        return {"status": "warning", "note": "no sector_perf meta", "sectors": {}}

    sectors_out: Dict[str, Any] = {}
    low_n_sectors = 0

    for sector, per_h in sector_perf.items():
        if not isinstance(per_h, dict):
            continue
        sector_info: Dict[str, Any] = {}
        for h, stats in per_h.items():
            if not isinstance(stats, dict):
                continue
            n = int(stats.get("n", 0))
            hit = float(stats.get("hit_ratio", 0.5))
            mae = float(stats.get("mae", 0.0))
            sector_info[h] = {"n": n, "hit_ratio": hit, "mae": mae}
            if n < 20:
                low_n_sectors += 1
        sectors_out[str(sector)] = sector_info

    if not sectors_out:
        return {"status": "warning", "note": "sector_perf empty after parse", "sectors": {}}

    # If half+ of sectors are low sample, warn (but don't kill system)
    if low_n_sectors > 0 and low_n_sectors >= max(1, len(sectors_out) // 2):
        status = "warning"
    else:
        status = "ok"

    return {"status": status, "sectors": sectors_out}


def _check_execution_performance() -> Dict[str, Any]:
    """
    Reads performance snapshot produced by backend/services/performance_aggregator.py
    """
    if not PERF_FILE.exists():
        return {"status": "warning", "note": "system_perf missing", "path": str(PERF_FILE)}

    perf = _load_json(PERF_FILE)
    metrics = perf.get("metrics", {}) if isinstance(perf, dict) else {}
    trades = int(metrics.get("trades", 0) or 0)
    win_rate = safe_float(metrics.get("win_rate", 0.5))
    dd14 = safe_float(metrics.get("drawdown_14d", 0.0))
    age = _file_age_minutes(PERF_FILE)

    # classify
    status = "ok"
    if age >= 1440:
        status = "warning"
    if dd14 <= -0.10:
        status = "critical"
    elif dd14 <= -0.05:
        status = "warning"
    if trades >= 10 and win_rate < 0.40:
        status = "warning" if status != "critical" else status

    return {
        "status": status,
        "age_min": age,
        "path": str(PERF_FILE),
        "metrics": metrics,
    }


def _check_aion_brain_health() -> Dict[str, Any]:
    """
    AION brain controls policy bias knobs:
        confidence_bias, risk_bias, aggressiveness, regime_mods
    Your policy_engine reads PATHS["aion_brain"] via _read_aion_brain().
    """
    if not AION_BRAIN_FILE:
        return {"status": "warning", "note": "PATHS['aion_brain'] not set"}

    age = _file_age_minutes(AION_BRAIN_FILE)
    status = _status_from_age(age, warn=1440, crit=10080)  # warn 1d, crit 7d

    ab = {}
    try:
        ab = _read_aion_brain() or {}
    except Exception:
        ab = {}

    meta = ab.get("_meta", {}) if isinstance(ab, dict) else {}
    if not isinstance(meta, dict):
        meta = {}

    cb = safe_float(meta.get("confidence_bias", 1.0))
    rb = safe_float(meta.get("risk_bias", 1.0))
    ag = safe_float(meta.get("aggressiveness", 1.0))

    # sanity: if wildly out of bounds, warn
    if cb < 0.6 or cb > 1.5 or rb < 0.5 or rb > 1.7 or ag < 0.5 or ag > 2.0:
        status = "warning" if status != "critical" else status

    return {
        "status": status,
        "age_min": age,
        "path": str(AION_BRAIN_FILE),
        "meta": {
            "confidence_bias": cb,
            "risk_bias": rb,
            "aggressiveness": ag,
            "updated_at": meta.get("updated_at"),
            "has_regime_mods": isinstance(meta.get("regime_mods"), dict),
        },
    }


# ---------------------------------------------------------------------
# Overrides (kill switch / exposure caps)
# ---------------------------------------------------------------------

def compute_overrides_from_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Back-compat manual override computation.
    """
    dd = float(metrics.get("drawdown_7d", 0.0) or 0.0)
    regime = (metrics.get("regime") or "neutral").lower()
    rconf = float(metrics.get("regime_conf", 0.0) or 0.0)

    if dd <= -0.10:
        overrides = {"kill_switch": True, "conf_min": 0.65, "exposure_cap": 0.3}
    elif dd <= -0.05:
        overrides = {"kill_switch": True, "conf_min": 0.60, "exposure_cap": 0.5}
    elif regime == "panic" and rconf > 0.7:
        overrides = {"kill_switch": False, "conf_min": 0.60, "exposure_cap": 0.6}
    elif regime == "bear" and rconf > 0.6:
        overrides = {"kill_switch": False, "conf_min": 0.56, "exposure_cap": 0.7}
    else:
        overrides = {"kill_switch": False, "conf_min": 0.52, "exposure_cap": 1.2}

    overrides["updated_at"] = datetime.datetime.now(TIMEZONE).isoformat()
    _save_overrides(overrides)
    log(f"[supervisor_agent] ✅ overrides updated → {overrides}")
    return overrides


def update_overrides(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return compute_overrides_from_metrics(metrics)


def _compute_overrides_from_truth(
    exec_perf: Dict[str, Any],
    drift: Dict[str, Any],
    pred_disp: Dict[str, Any],
) -> Dict[str, Any]:
    """
    This is the actual supervisor “autopilot”:
    - uses execution drawdown + win rate
    - uses worst negative drift (system degradation)
    - uses prediction dispersion flatness check (degenerate model)

    Produces:
        kill_switch, conf_min, exposure_cap
    """
    kill = False
    conf_min = 0.52
    exposure = 1.2

    # --- execution performance ---
    metrics = (exec_perf.get("metrics") or {}) if isinstance(exec_perf, dict) else {}
    dd14 = safe_float(metrics.get("drawdown_14d", 0.0))
    win = safe_float(metrics.get("win_rate", 0.5))
    trades = int(metrics.get("trades", 0) or 0)

    if dd14 <= -0.10:
        kill = True
        conf_min = max(conf_min, 0.65)
        exposure = min(exposure, 0.35)
    elif dd14 <= -0.05:
        conf_min = max(conf_min, 0.60)
        exposure = min(exposure, 0.60)

    if trades >= 12 and win < 0.42:
        conf_min = max(conf_min, 0.58)
        exposure = min(exposure, 0.70)

    # --- drift degradation ---
    worst_neg = safe_float(drift.get("worst_negative_drift", 0.0))
    if worst_neg <= -0.02:
        conf_min = max(conf_min, 0.62)
        exposure = min(exposure, 0.60)
    elif worst_neg <= -0.01:
        conf_min = max(conf_min, 0.56)
        exposure = min(exposure, 0.85)

    # --- flat/degenerate model check ---
    # if many horizons have tiny std, force caution
    per_h = pred_disp.get("per_horizon", {}) if isinstance(pred_disp, dict) else {}
    tiny = 0
    total = 0
    for h, st in per_h.items():
        if not isinstance(st, dict):
            continue
        total += 1
        if safe_float(st.get("std", 0.0)) < 0.0015:
            tiny += 1
    if total >= 4 and tiny >= max(2, total // 2):
        conf_min = max(conf_min, 0.60)
        exposure = min(exposure, 0.70)

    conf_min = float(max(0.45, min(0.80, conf_min)))
    exposure = float(max(0.10, min(1.50, exposure)))

    return {
        "kill_switch": bool(kill),
        "conf_min": round(conf_min, 3),
        "exposure_cap": round(exposure, 3),
        "updated_at": datetime.datetime.now(TIMEZONE).isoformat(),
        "source": "truth_loop",
    }


# ---------------------------------------------------------------------
# Main supervisor verdict
# ---------------------------------------------------------------------

def supervisor_verdict() -> Dict[str, Any]:
    """
    High-level system verdict for dashboards / routers:

        {
          "status": "ok" | "warning" | "critical",
          "components": {...},
          "overrides": {...}
        }
    """
    _maybe_log_supervisor("[supervisor_agent] 🔍 Evaluating system health (v1.2 truth-loop)…")

    dataset = _check_dataset_health()
    models = _check_model_health()
    intel = _check_intel_files()
    insights = _check_insights_health()
    drift = _check_drift_health()
    rolling_cov = _check_rolling_coverage()
    pred_disp = _check_prediction_dispersion()
    sector_perf = _check_sector_perf()
    exec_perf = _check_execution_performance()
    aion_brain = _check_aion_brain_health()

    # Determine overall status from components
    statuses = [
        dataset["status"],
        models["status"],
        intel["macro"]["status"],
        intel["news_intel"]["status"],
        intel["social_intel"]["status"],
        insights["status"],
        drift["status"],
        rolling_cov["status"],
        pred_disp["status"],
        sector_perf["status"],
        exec_perf["status"],
        aion_brain["status"],
    ]

    if "critical" in statuses:
        overall = "critical"
    elif "warning" in statuses:
        overall = "warning"
    else:
        overall = "ok"

    # Build overrides based on truth loop (preferred)
    perf_metrics = exec_perf.get("metrics", {}) if isinstance(exec_perf, dict) else {}
    truth_overrides = _compute_overrides_from_truth(
        exec_perf=perf_metrics,
        drift=drift,
        pred_disp=pred_disp,
    )
    _save_overrides(truth_overrides)

    verdict = {
        "status": overall,
        "components": {
            "dataset": dataset,
            "models": models,
            "intel": intel,
            "insights": insights,
            "drift": drift,
            "rolling_coverage": rolling_cov,
            "prediction_dispersion": pred_disp,
            "sector_perf": sector_perf,
            "execution_performance": exec_perf,
            "aion_brain": aion_brain,
        },
        "overrides": truth_overrides,
        "generated_at": datetime.datetime.now(TIMEZONE).isoformat(),
    }

    sig = json.dumps({'status': overall, 'overrides': truth_overrides}, sort_keys=True, default=str)
    _maybe_log_supervisor(f"[supervisor_agent] 🧭 Supervisor verdict: {overall} | overrides={truth_overrides}", sig=sig)
    return verdict


def run_supervisor_agent() -> Dict[str, Any]:
    """
    Backward-compatible wrapper for nightly_job.
    """
    return supervisor_verdict()


if __name__ == "__main__":
    v = supervisor_verdict()
    print(json.dumps(v, indent=2))