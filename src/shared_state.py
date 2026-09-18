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
