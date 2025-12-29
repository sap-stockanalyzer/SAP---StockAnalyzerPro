"""
AION Analytics — Saved Model Artifacts

This package stores trained intraday models and metadata:

Subfolders:
  • ensemble/
  • lightgbm_intraday/
  • lstm_intraday/
  • transformer_intraday/

Each contains:
  - model files
  - config.json
  - feature_map.json
  - label_map.json

This directory is used for loading model artifacts at runtime.
It does NOT contain training logic — training occurs in dt_backend/ml.
"""

__all__ = [
    "ensemble",
    "lightgbm_intraday",
    "lstm_intraday",
    "transformer_intraday",
]

# ---------------------------------------------------------------------
# Legacy compatibility for intraday model loaders
# ---------------------------------------------------------------------

# Intraday labels remain 3-class: SELL(-1), HOLD(0), BUY(+1)
LABEL_ORDER = ["SELL", "HOLD", "BUY"]
# Accept both string labels and legacy numeric labels (-1/0/1)
LABEL2ID = {"SELL": 0, "HOLD": 1, "BUY": 2, -1: 0, 0: 1, 1: 2}
ID2LABEL = {0: "SELL", 1: "HOLD", 2: "BUY"}
from pathlib import Path

def get_model_dir(name: str = "") -> Path:
    """
    Legacy helper expected by ai_model_intraday.
    Returns the base folder for model artifacts.

    Examples:
        get_model_dir() → dt_backend/models
        get_model_dir("lightgbm_intraday") → dt_backend/models/lightgbm_intraday
    """
    base = Path(__file__).resolve().parent
    if name:
        return base / name
    return base
