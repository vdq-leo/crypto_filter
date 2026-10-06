from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from src.data import DataManager, BinanceFuturesFetcher
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
fetcher = BinanceFuturesFetcher()

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
        benchmark_df = manager.load_data(BENCHMARK_SYMBOL, req.interval, auto_sync=False)
        if benchmark_df is None or benchmark_df.empty:
            try:
                benchmark_df = manager.load_data(BENCHMARK_SYMBOL, req.interval, auto_sync=True)
            except Exception:
                pass

        benchmark_returns = None
        benchmark_prices = None
        if benchmark_df is not None and not benchmark_df.empty:
            b_close = pd.to_numeric(benchmark_df['close'], errors='coerce').ffill().fillna(0)
            benchmark_prices = b_close
            benchmark_returns = b_close.pct_change().dropna()

        # Pre-fetch funding rates and ADL risks once safely outside thread loop
        funding_rates = {}
        try:
            funding_rates = fetcher.get_funding_rates()
        except Exception as e:
            logger.warning(f"Could not fetch funding rates: {e}")

        adl_risks = {}
        try:
            adl_risks = fetcher.get_all_adl_risks()
        except Exception as e:
            logger.warning(f"Could not fetch ADL risks: {e}")

        def process_symbol(sym):
            try:
                # auto_sync=False avoids firing 200 concurrent API requests to Binance
                df = manager.load_data(sym, req.interval, auto_sync=False)
                if df is not None and not df.empty:
                    df = df.tail(max(req.filter_window * 5, 2500))
                    if not df.empty:
                        adv_df = engine.calculate_all_indicators(
                            df, 
                            window=req.filter_window, 
                            benchmark_returns=benchmark_returns, 
                            benchmark_prices=benchmark_prices, 
                            interval=req.interval, 
                            tail_only=True
                        )
                        row = {
                            'symbol': sym,
                            'count': len(df)
                        }
                        row.update(adv_df.iloc[-1].to_dict())
                        row['funding_rate'] = funding_rates.get(sym, np.nan)
                        row['adl_risk'] = adl_risks.get(sym, np.nan)
                        return row
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
                    if single_res is not None:
                        results.append(single_res)
                except Exception as e:
                    logger.error(f"Future error for {sym}: {e}")

        # Need to clean up NaNs/Infs to None for JSON serialization
        cleaned_results = []
        for r in results:
            clean = {}
            for k, v in r.items():
                if isinstance(v, (float, np.floating)):
                    clean[k] = None if (np.isnan(v) or np.isinf(v)) else float(v)
                elif isinstance(v, (int, np.integer)):
                    clean[k] = int(v)
                elif pd.isna(v):
                    clean[k] = None
                else:
                    clean[k] = v
            cleaned_results.append(clean)
            
        return {"metrics": cleaned_results}
    except Exception as e:
        logger.error(f"Snapshot error: {e}")
        return {"metrics": []}

@router.post("/path")
def get_path_analysis(req: PathRequest):
    """Calculate trajectory paths for market radar."""
    try:
        key_x = get_metric_key(req.x_metric)
        key_y = get_metric_key(req.y_metric)
        required_metrics = list(set([key_x, key_y]))

        benchmark_df = manager.load_data(BENCHMARK_SYMBOL, req.interval, auto_sync=False)
        benchmark_prices = None
        if benchmark_df is not None and not benchmark_df.empty:
            benchmark_prices = pd.to_numeric(benchmark_df['close'], errors='coerce').ffill().fillna(0)

        def process_rpg_symbol(sym):
            try:
                df = manager.load_data(sym, req.interval, auto_sync=False)
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
