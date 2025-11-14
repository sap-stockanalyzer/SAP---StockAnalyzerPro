import os, glob, importlib, json, sys
print("=== StockAnalyzerPro Wiring Doctor ===")

def ok(msg): print("✅", msg)
def warn(msg): print("⚠️", msg)
def bad(msg): print("❌", msg)

# 1) package
if os.path.exists("backend/__init__.py"):
    ok("backend/ is a Python package")
else:
    bad("Create backend/__init__.py (empty file)")

# 2) critical dirs
for d in [
    "ml_data","ml_data/logs","ml_data/metrics_history","ml_data/prediction_logs",
    "ml_data/prediction_outcomes","ml_data/models","ml_data/drift_reports",
    "stock_cache/daily","news_cache"
]:
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
        warn(f"Created missing directory: {d}")
    else:
        ok(f"Dir exists: {d}")

# 3) json files
must_json = [
    ("stock_cache/master.json", "master ticker map"),
    ("stock_cache/universe.json", "ticker universe (list)")
]
for path, label in must_json:
    if not os.path.exists(path):
        warn(f"Missing {label}: {path}")
    else:
        try:
            with open(path, "r", encoding="utf-8") as f: json.load(f)
            ok(f"JSON OK: {path}")
        except Exception as e:
            bad(f"Invalid JSON {path}: {e}")

# 4) import key modules
mods = [
    "backend.data_pipeline",
    "backend.news_fetcher",
    "backend.ml_data_builder",
    "backend.prediction_logger",
    "backend.outcomes_harvester",
    "backend.online_trainer",
    "backend.drift_monitor",
    "backend.train_lightgbm",
    "backend.ai_model"
]
for m in mods:
    try:
        importlib.import_module(m)
        ok(f"Import OK: {m}")
    except Exception as e:
        warn(f"Import issue in {m}: {e}")

# 5) verify requirements (best-effort)
try:
    import numpy as np, evidently
    ver = np.__version__
    ok(f"NumPy {ver}")
    if tuple(map(int, ver.split(".")[:2])) >= (2,0):
        warn("NumPy>=2 detected: pin numpy<2.0.0 for Evidently 0.4.x")
except Exception as e:
    warn(f"NumPy not importable: {e}")
try:
    import evidently as ev
    ok(f"Evidently present: {ev.__version__}")
except Exception as e:
    warn(f"Evidently import issue: {e}")

print("=== Wiring check complete ===")
