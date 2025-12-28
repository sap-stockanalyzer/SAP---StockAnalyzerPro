"""
AION Analytics — ai_model package

This package is a refactor of a previously monolithic `ai_model.py` file.
To avoid circular imports and underscore-star-import pitfalls, modules should
import constants from `backend.core.ai_model.constants` and internal helpers
directly from their owning modules.
"""

from .core_training import train_model, train_all_models, predict_all

__all__ = ["train_model", "train_all_models", "predict_all"]
