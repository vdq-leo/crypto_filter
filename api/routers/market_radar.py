from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from src.data import DataManager
from src.metrics import MetricsEngine
from src.config import BENCHMARK_SYMBOL
from src.shared_state import get_manager, get_engine
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

router = APIRouter()
logger = logging.getLogger(__name__)

manager = get_manager()
engine = get_engine()

class SnapshotRequest(BaseModel):
    symbols: List[str]
    interval: str = "1h"
    filter_window: int = 40

class PathRequest(BaseModel):
    symbols: List[str]
    interval: str = "1h"
    filter_window: int = 40
    step_size: int = 10
    max_points: int = 3
    x_metric: str
    y_metric: str

def get_metric_key(m: str) -> str:
    return 'return' if m == 'metric_return' else m

@router.post("/snapshot")
def get_market_snapshot(req: SnapshotRequest):
    """Calculate market radar snapshot metrics for a list of symbols."""
    try:
        benchmark_df = manager.load_data(BENCHMARK_SYMBOL, req.interval, auto_sync=True)
        benchmark_returns = None
        benchmark_prices = None
        if benchmark_df is not None and not benchmark_df.empty:
            b_close = pd.to_numeric(benchmark_df['close'], errors='coerce').ffill().fillna(0)
            benchmark_prices = b_close
            benchmark_returns = b_close.pct_change().dropna()

        def process_symbol(sym):
            try:
                df = manager.load_data(sym, req.interval, auto_sync=True)
                if df is not None and not df.empty:
                    df = df.tail(req.filter_window * 5)
                    if not df.empty:
                        return engine.compute_all_metrics(
                            {sym: df}, 
                            interval=req.interval, 
                            benchmark_symbol=BENCHMARK_SYMBOL,
                            benchmark_returns=benchmark_returns,
                            benchmark_prices=benchmark_prices,
                            window=req.filter_window
                        )
            except Exception as e:
                logger.error(f"Error computing {sym}: {e}")
            return None

        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_sym = {executor.submit(process_symbol, sym): sym for sym in req.symbols}
            for future in as_completed(future_to_sym):
                sym = future_to_sym[future]
                try:
                    single_res = future.result()
                    if single_res is not None and not single_res.empty:
                        results.append(single_res.iloc[0].to_dict())
                except Exception as e:
                    logger.error(f"Future error for {sym}: {e}")

        # Need to clean up NaNs to None for JSON serialization
        cleaned_results = []
        for r in results:
            clean = {k: (None if pd.isna(v) else v) for k, v in r.items()}
            cleaned_results.append(clean)
            
        return {"metrics": cleaned_results}

    except Exception as e:
        logger.error(f"Snapshot error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/path")
def get_path_analysis(req: PathRequest):
    """Calculate trajectory paths for market radar."""
    try:
        key_x = get_metric_key(req.x_metric)
        key_y = get_metric_key(req.y_metric)
        required_metrics = list(set([key_x, key_y]))

        benchmark_df = manager.load_data(BENCHMARK_SYMBOL, req.interval, auto_sync=True)
        benchmark_prices = None
        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_prices = pd.to_numeric(benchmark_df['close'], errors='coerce').ffill().fillna(0)

        def process_rpg_symbol(sym):
            try:
                df = manager.load_data(sym, req.interval, auto_sync=True)
                if df is not None and not df.empty:
                    df['close'] = pd.to_numeric(df['close'], errors='coerce').ffill().fillna(0)
                    inds = engine.calculate_all_indicators(
                        df, 
                        window=req.filter_window, 
                        interval=req.interval,
                        include_metrics=required_metrics,
                        benchmark_prices=benchmark_prices
                    )
                    
                    if key_x in inds.columns and key_y in inds.columns:
                        sx = inds[key_x]
                        sy = inds[key_y]
                        
                        valid_sx = sx.dropna()
                        valid_sy = sy.dropna()
                        common_idx = valid_sx.index.intersection(valid_sy.index)
                        
                        if len(common_idx) >= 1:
                            indices = []
                            for i in range(req.max_points):
                                idx = -(1 + i * req.step_size)
                                if abs(idx) <= len(common_idx):
                                    indices.append(common_idx[idx])
                            
                            indices = indices[::-1]
                            sx, sy = sx.loc[indices], sy.loc[indices]
                            order = np.linspace(0.2, 1.0, len(sx))
                            
                            points = []
                            for i in range(len(sx)):
                                points.append({
                                    'X_Value': float(sx.values[i]),
                                    'Y_Value': float(sy.values[i]),
                                    'Symbol': sym,
                                    'Order': float(order[i]),
                                    'Marker_Size': float((order[i] + 1) * 8)
                                })
                            return points
            except Exception as e:
                logger.error(f"Error in RPG process for {sym}: {e}")
            return None

        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_sym = {executor.submit(process_rpg_symbol, sym): sym for sym in req.symbols}
            for future in as_completed(future_to_sym):
                res = future.result()
                if res is not None:
                    results.extend(res)

        # Clean NaNs
        cleaned_results = []
        for r in results:
            clean = {k: (None if pd.isna(v) or not np.isfinite(v) else v) for k, v in r.items() if isinstance(v, (int, float, str, bool)) or v is None}
            cleaned_results.append(clean)
            
        return {"data": cleaned_results}

    except Exception as e:
        logger.error(f"RPG error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
