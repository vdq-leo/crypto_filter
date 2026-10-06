import os
from src.data import DataManager
from src.metrics import MetricsEngine

_manager = None
_engine = None

def get_manager() -> DataManager:
    global _manager
    if _manager is None:
        data_dir = os.environ.get("DATA_DIR", "data_cache")
        _manager = DataManager(data_dir=data_dir)
    return _manager

def get_engine() -> MetricsEngine:
    global _engine
    if _engine is None:
        _engine = MetricsEngine()
    return _engine

def sanitize_for_json(data):
    """Recursively replaces NaN, Inf, and non-serializable pandas/numpy types with None."""
    import numpy as np
    import pandas as pd
    if isinstance(data, (float, np.floating)):
        return None if (np.isnan(data) or np.isinf(data)) else float(data)
    elif isinstance(data, (int, np.integer)):
        return int(data)
    elif isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [sanitize_for_json(v) for v in data]
    elif isinstance(data, np.ndarray):
        return [sanitize_for_json(v) for v in data.tolist()]
    elif isinstance(data, pd.DataFrame):
        return sanitize_for_json(data.to_dict(orient="records"))
    elif isinstance(data, pd.Series):
        return sanitize_for_json(data.to_dict())
    elif pd.isna(data):
        return None
    return data
