"""
dt_backend/jobs/post_market_analysis.py

Post-market analysis job - runs at 16:05 ET daily.

Workflow:
1. Analyze all closed trades from today
2. Analyze missed opportunities  
3. Update performance metrics
4. Check retrain triggers
5. Update DT Brain knobs
6. Generate learning report
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

try:
    from dt_backend.core.config_dt import DT_PATHS
    from dt_backend.core.logger_dt import log
except Exception:
    DT_PATHS = {}
    def log(msg: str) -> None:
        print(msg, flush=True)


def _get_ny_time() -> datetime:
    """Get current time in New York timezone."""
    if ZoneInfo:
        return datetime.now(ZoneInfo("America/New_York"))
    return datetime.now(timezone.utc)


def run_post_market_analysis() -> Dict[str, Any]:
    """Main post-market learning workflow."""
    log("[post_market] 📊 Starting post-market analysis...")
    
    ny_time = _get_ny_time()
    
    report = {
        "date": ny_time.date().isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "steps": {},
    }
    
    try:
        # Step 1: Analyze trade outcomes
        log("[post_market] 1️⃣ Analyzing trade outcomes...")
        try:
            from dt_backend.ml.trade_outcome_analyzer import analyze_all_trades_today
            trade_analysis = analyze_all_trades_today()
            report["steps"]["trade_analysis"] = trade_analysis
            log(f"[post_market] ✅ Trade analysis: {trade_analysis.get('trades_analyzed', 0)} trades")
        except Exception as e:
            log(f"[post_market] ⚠️ Trade analysis error: {e}")
            report["steps"]["trade_analysis"] = {"status": "error", "error": str(e)}
        
        # Step 2: Analyze missed opportunities
        log("[post_market] 2️⃣ Analyzing missed opportunities...")
        try:
            from dt_backend.ml.missed_opportunity_tracker import analyze_missed_today
            missed_analysis = analyze_missed_today()
            report["steps"]["missed_opportunities"] = missed_analysis
            log(f"[post_market] ✅ Missed opportunities: {missed_analysis.get('evaluated', 0)} evaluated")
        except Exception as e:
            log(f"[post_market] ⚠️ Missed opportunity analysis error: {e}")
            report["steps"]["missed_opportunities"] = {"status": "error", "error": str(e)}
        
        # Step 3: Check retrain triggers
        log("[post_market] 3️⃣ Checking retrain triggers...")
        try:
            from dt_backend.services.auto_retrain_dt import check_and_retrain
            retrain_result = check_and_retrain()
            report["steps"]["retrain"] = retrain_result
            
            if retrain_result.get("status") == "deployed":
                log("[post_market] 🚀 Models retrained and deployed")
            elif retrain_result.get("status") == "skipped":
                log("[post_market] ✅ No retrain needed")
            else:
                log(f"[post_market] ℹ️ Retrain status: {retrain_result.get('status')}")
        except Exception as e:
            log(f"[post_market] ⚠️ Retrain check error: {e}")
            report["steps"]["retrain"] = {"status": "error", "error": str(e)}
        
        # Step 4: Update DT Brain
        log("[post_market] 4️⃣ Updating DT Brain knobs...")
        try:
            from dt_backend.core.dt_brain import update_dt_brain
            brain_update = update_dt_brain()
            report["steps"]["brain_update"] = brain_update
            
            adjustments = brain_update.get("adjustments_applied", 0)
            if adjustments > 0:
                log(f"[post_market] 🧠 DT Brain: {adjustments} knobs adjusted")
            else:
                log("[post_market] ✅ DT Brain: no adjustments needed")
        except Exception as e:
            log(f"[post_market] ⚠️ Brain update error: {e}")
            report["steps"]["brain_update"] = {"status": "error", "error": str(e)}
        
        # Step 5: Generate report
        log("[post_market] 5️⃣ Generating report...")
        _save_report(report)
        
        log("[post_market] ✅ Post-market analysis complete")
        report["status"] = "success"
        
    except Exception as e:
        log(f"[post_market] ⚠️ Unexpected error: {e}")
        report["status"] = "error"
        report["error"] = str(e)
    
    return report


def _save_report(report: Dict[str, Any]) -> None:
    """Save post-market analysis report."""
    try:
        learning_path = DT_PATHS.get("learning")
        if learning_path:
            report_dir = Path(learning_path) / "reports"
        else:
            da_brains = DT_PATHS.get("da_brains", Path("da_brains"))
            report_dir = Path(da_brains) / "dt_learning" / "reports"
        
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save daily report
        date_str = report.get("date", datetime.now(timezone.utc).date().isoformat())
        report_file = report_dir / f"post_market_{date_str}.json"
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Also save as "latest"
        latest_file = report_dir / "post_market_latest.json"
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        log(f"[post_market] 📄 Report saved: {report_file}")
        
    except Exception as e:
        log(f"[post_market] ⚠️ Error saving report: {e}")


if __name__ == "__main__":
    # Allow running directly for testing
    result = run_post_market_analysis()
    print(json.dumps(result, indent=2))
