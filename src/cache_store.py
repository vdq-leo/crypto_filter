import os
import time
import threading
from typing import Any, Optional, Tuple
import pandas as pd


class MetricCacheStore:
    """
    Thread-safe TTL cache for computed Market Radar axis values.
    Keyed by (symbol, source, metric, interval, frozen_params) so changing one
    axis reuses cached values for unchanged axes without recomputing or re-fetching.
    """

    TTL_BY_INTERVAL = {
        "snapshot": 60.0,
        "1m": 45.0,
        "3m": 60.0,
        "5m": 60.0,
        "15m": 120.0,
        "30m": 180.0,
        "1h": 300.0,
        "2h": 300.0,
        "4h": 600.0,
        "6h": 600.0,
        "8h": 600.0,
        "12h": 600.0,
        "1d": 900.0,
        "3d": 900.0,
        "1w": 900.0,
    }

    def __init__(self, max_entries: int = 20000):
        self._store = {}
        self._lock = threading.Lock()
        self._max_entries = max_entries

    @staticmethod
    def make_key(symbol: str, source: str, metric: str, interval: str, params: Optional[dict]) -> Tuple:
        frozen_params = tuple(sorted((k, str(v)) for k, v in (params or {}).items()))
        return (symbol, source, metric, interval if source != "snapshot" else "snapshot", frozen_params)

    def get_ttl(self, source: str, interval: str) -> float:
        if source == "snapshot":
            return self.TTL_BY_INTERVAL["snapshot"]
        return self.TTL_BY_INTERVAL.get(interval, 300.0)

    def get(self, key: Tuple, ttl: float) -> Optional[float]:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                ts, val = entry
                if now - ts < ttl:
                    return val
                del self._store[key]
        return None

    def set(self, key: Tuple, value: float) -> None:
        now = time.time()
        with self._lock:
            if len(self._store) >= self._max_entries:
                # Evict oldest 20%
                cutoff = int(self._max_entries * 0.2)
                for k in list(self._store.keys())[:cutoff]:
                    del self._store[k]
            self._store[key] = (now, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._store), "max_entries": self._max_entries}


class SeriesDiskCache:
    """
    Persistent disk + memory cache for non-OHLCV market stats time series
    (Open Interest, Top Trader L/S, Global L/S, Taker Ratio, Funding Rate)
    stored as parquet files under data_cache/stats/.
    """

    TTL_BY_PERIOD = {
        "5m": 180.0,
        "15m": 300.0,
        "30m": 300.0,
        "1h": 600.0,
        "2h": 600.0,
        "4h": 900.0,
        "6h": 900.0,
        "12h": 1800.0,
        "1d": 1800.0,
    }

    def __init__(self, base_dir: str = "data_cache/stats"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self._mem = {}
        self._lock = threading.Lock()

    def _path(self, symbol: str, source: str, period: str) -> str:
        return os.path.join(self.base_dir, f"{symbol}_{source}_{period}.parquet")

    def get(self, symbol: str, source: str, period: str) -> Optional[pd.Series]:
        ttl = self.TTL_BY_PERIOD.get(period, 600.0)
        now = time.time()
        key = (symbol, source, period)

        with self._lock:
            if key in self._mem:
                ts, s = self._mem[key]
                if now - ts < ttl:
                    return s

        path = self._path(symbol, source, period)
        if os.path.exists(path):
            try:
                mtime = os.path.getmtime(path)
                if now - mtime < ttl:
                    df = pd.read_parquet(path)
                    if not df.empty and "ts" in df.columns and "val" in df.columns:
                        s = pd.Series(df["val"].values, index=pd.to_datetime(df["ts"]))
                        with self._lock:
                            self._mem[key] = (mtime, s)
                        return s
            except Exception:
                pass
        return None

    def set(self, symbol: str, source: str, period: str, series: pd.Series) -> None:
        if series is None or series.empty:
            return
        now = time.time()
        key = (symbol, source, period)
        with self._lock:
            self._mem[key] = (now, series)

        path = self._path(symbol, source, period)
        try:
            df = pd.DataFrame({"ts": series.index, "val": series.values})
            df.to_parquet(path, index=False)
        except Exception:
            pass


class UIStateStore:
    """
    Persistent memory + disk store for UI session state (selected assets,
    Market Radar snapshot, Short Scanner scoreboard) so sleep / WebSocket
    reconnects restore the exact previous view without recalculating.
    """

    def __init__(self, base_dir: str = "data_cache/ui_state"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self._mem = {}
        self._lock = threading.Lock()

    def _path(self, key: str) -> str:
        return os.path.join(self.base_dir, f"{key}.json")

    def save(self, key: str, data: dict) -> None:
        import json
        with self._lock:
            self._mem[key] = data
        try:
            with open(self._path(key), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass

    def load(self, key: str) -> Optional[dict]:
        import json
        with self._lock:
            if key in self._mem:
                return self._mem[key]
        path = self._path(key)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                with self._lock:
                    self._mem[key] = data
                return data
            except Exception:
                pass
        return None


_ui_state_store = None


def get_ui_state() -> UIStateStore:
    global _ui_state_store
    if _ui_state_store is None:
        _ui_state_store = UIStateStore()
    return _ui_state_store
