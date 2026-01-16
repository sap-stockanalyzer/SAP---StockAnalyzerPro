"""
Caching utilities for performance optimization.
"""

from __future__ import annotations

import json
import time
from functools import wraps, lru_cache
from typing import Any, Callable, Dict, Optional, Tuple


def _make_hashable(obj: Any) -> Any:
    """Convert unhashable types to hashable equivalents."""
    if isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (list, tuple)):
        return tuple(_make_hashable(item) for item in obj)
    elif isinstance(obj, set):
        return frozenset(_make_hashable(item) for item in obj)
    else:
        # For other types, try to use them directly or convert to string
        try:
            hash(obj)
            return obj
        except TypeError:
            return str(obj)


def timed_lru_cache(seconds: int, maxsize: int = 128):
    """
    LRU cache that expires after seconds duration.
    Handles unhashable kwargs by converting them to hashable equivalents.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            time_bucket = int(time.time() // seconds)
            # Convert kwargs to hashable format
            hashable_kwargs = tuple(sorted((k, _make_hashable(v)) for k, v in kwargs.items()))
            cache_key = (time_bucket, args, hashable_kwargs)
            return _cached_call(cache_key)
        
        @lru_cache(maxsize=maxsize)
        def _cached_call(cache_key: Tuple) -> Any:
            _, args, kwargs_tuple = cache_key
            kwargs = dict(kwargs_tuple)
            return func(*args, **kwargs)
        
        wrapper.cache_clear = _cached_call.cache_clear
        return wrapper
    
    return decorator
