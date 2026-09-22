from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.data import BinanceFuturesFetcher, DataManager
from src.config import AVAILABLE_INTERVALS, MANDATORY_CRYPTO, IGNORED_CRYPTO
from src.shared_state import get_manager, get_engine
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from src.logger import logger

router = APIRouter()

FETCH_IN_PROGRESS = False

# Singletons equivalent for the router
fetcher = BinanceFuturesFetcher()
manager = get_manager()

class FetchRequest(BaseModel):
    symbols: List[str]
    intervals: List[str]
    mode: str = "Range" # "Range" or "Limit"
    days_back: int = 30
    limit: int = 1000

@router.get("/universe")
def get_universe(top_n: int = 50, bottom: bool = False):
    """Get top N symbols combining mandatory and top volume from Binance."""
    try:
        new_syms = fetcher.get_top_volume_symbols(top_n=top_n, bottom=bottom)
        combined = set(MANDATORY_CRYPTO).union(new_syms)
        filtered = {s for s in combined if s not in IGNORED_CRYPTO}
        return {"symbols": sorted(list(filtered))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/klines")
def get_klines(symbol: str, interval: str = "1h", limit: int = 500):
    """Return OHLCV candle data for a symbol from local cache."""
    try:
        df = manager.load_data(symbol, interval, auto_sync=False)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No cached data for {symbol}/{interval}")

        df = df.tail(limit).copy()
        df['open_time'] = pd.to_datetime(df['open_time'])

        candles = []
        for _, row in df.iterrows():
            ts = int(row['open_time'].timestamp())
            candles.append({
                "time": ts,
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume']) if 'volume' in row else 0.0
            })

        return {"symbol": symbol, "interval": interval, "candles": candles}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching klines for {symbol}/{interval}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def background_fetch_task(req: FetchRequest):
    global FETCH_IN_PROGRESS
    FETCH_IN_PROGRESS = True
    try:
        for sym in req.symbols:
            for inter in req.intervals:
                try:
                    first_ts, last_ts = manager.get_cache_range(sym, inter)
                    if req.mode == "Range":
                        end_t = datetime.now()
                        requested_start = end_t - pd.Timedelta(days=req.days_back)

                        end_t_utc = pd.to_datetime(end_t.timestamp(), unit='s')
                        req_start_utc = pd.to_datetime(requested_start.timestamp(), unit='s')

                        dfs_to_fetch = []

                        if not first_ts:
                            df_full = fetcher.fetch_history(sym, inter, start_time=requested_start, end_time=end_t)
                            if not df_full.empty:
                                dfs_to_fetch.append(df_full)
                        else:
                            # Forward Map
                            if last_ts < end_t_utc - pd.Timedelta(minutes=5):
                                start_ts_ms = int(pd.Timestamp(last_ts).tz_localize('UTC').timestamp() * 1000) + 1
                                df_fwd = fetcher.fetch_history(sym, inter, start_time=start_ts_ms, end_time=end_t)
                                if not df_fwd.empty:
                                    dfs_to_fetch.append(df_fwd)

                            # Backward Map
                            if first_ts > req_start_utc + pd.Timedelta(minutes=5):
                                end_ts_ms = int(pd.Timestamp(first_ts).tz_localize('UTC').timestamp() * 1000) - 1
                                df_bwd = fetcher.fetch_history(sym, inter, start_time=requested_start, end_time=end_ts_ms)
                                if not df_bwd.empty:
                                    dfs_to_fetch.append(df_bwd)

                        if dfs_to_fetch:
                            final_df = pd.concat(dfs_to_fetch)
                            manager.append_data(sym, inter, final_df)
                    else:
                        # Limit
                        df = fetcher.fetch_candles(sym, inter, limit=req.limit)
                        if not df.empty:
                            manager.append_data(sym, inter, df)
                except Exception as e:
                    logger.log("API", "ERROR", f"Error fetching {sym} {inter}: {str(e)}")
    finally:
        FETCH_IN_PROGRESS = False

@router.post("/fetch")
def start_fetch(req: FetchRequest, background_tasks: BackgroundTasks):
    """Trigger background fetching of data"""
    global FETCH_IN_PROGRESS
    if FETCH_IN_PROGRESS:
        raise HTTPException(status_code=400, detail="Fetch already in progress")
    background_tasks.add_task(background_fetch_task, req)
    return {"message": "Data fetching started in background"}

@router.get("/fetch-status")
def get_fetch_status():
    """Get status of background fetch and last 5 logs"""
    logs_df = logger.get_logs(limit=5)
    logs_list = logs_df.to_dict(orient="records") if not logs_df.empty else []
    return {
        "in_progress": FETCH_IN_PROGRESS,
        "logs": logs_list
    }

@router.delete("/cache/{interval}")
def delete_cache(interval: str):
    """Delete cached data for interval"""
    try:
        count = manager.delete_data(interval)
        return {"message": f"Successfully deleted {count} files for {interval}", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metadata")
def get_metadata():
    """Get metadata for cached files"""
    try:
        metadata = manager.get_cache_metadata()
        return {"metadata": metadata}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
