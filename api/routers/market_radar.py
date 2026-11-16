from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from src.data import DataManager, BinanceFuturesFetcher
from src.metrics import MetricsEngine
from src.config import BENCHMARK_SYMBOL, ALL_METRICS
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

        # Map interval to Binance stats period
        period_mapping = {
            '1m': '5m', '3m': '5m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '12h': '12h',
            '1d': '1d'
        }
        period = period_mapping.get(req.interval, '1h')

        # Pre-fetch per-symbol API stats in parallel
        api_stats_cache = {}
        def fetch_symbol_stats(sym):
            try:
                oi_hist = fetcher.get_historical_stats(sym, period, fetcher.OPEN_INTEREST_HIST)
                top_pos = fetcher.get_historical_stats(sym, period, fetcher.TOP_LS_POSITION)
                top_acc = fetcher.get_historical_stats(sym, period, fetcher.TOP_LS_ACCOUNT)
                glob_acc = fetcher.get_historical_stats(sym, period, fetcher.GLOBAL_LS_ACCOUNT)
                taker_ratio = fetcher.get_historical_stats(sym, period, fetcher.TAKER_BUY_SELL)
                return sym, {
                    'oi_hist': oi_hist,
                    'top_pos': top_pos,
                    'top_acc': top_acc,
                    'glob_acc': glob_acc,
                    'taker_ratio': taker_ratio
                }
            except Exception as e:
                logger.warning(f"Error pre-fetching stats for {sym}: {e}")
                return sym, {}

        with ThreadPoolExecutor(max_workers=15) as executor:
            future_to_sym = {executor.submit(fetch_symbol_stats, sym): sym for sym in req.symbols}
            for future in as_completed(future_to_sym):
                try:
                    s_sym, s_data = future.result()
                    if s_data:
                        api_stats_cache[s_sym] = s_data
                except Exception:
                    pass

        # Determine max allowed age before triggering auto-sync
        stale_thresholds = {
            '1m': timedelta(minutes=5),
            '3m': timedelta(minutes=10),
            '5m': timedelta(minutes=15),
            '15m': timedelta(minutes=45),
            '30m': timedelta(hours=1),
            '1h': timedelta(hours=2),
            '2h': timedelta(hours=4),
            '4h': timedelta(hours=8),
            '6h': timedelta(hours=12),
            '8h': timedelta(hours=16),
            '12h': timedelta(hours=24),
            '1d': timedelta(days=2),
            '3d': timedelta(days=4),
            '1w': timedelta(days=8)
        }
        max_age = stale_thresholds.get(req.interval, timedelta(hours=2))
        now_dt = datetime.now()

        def process_symbol(sym):
            try:
                # Load cached data first (fast path)
                df = manager.load_data(sym, req.interval, auto_sync=False)
                
                # Check if data is missing or stale
                is_stale = False
                if df is None or df.empty or len(df) < req.filter_window:
                    is_stale = True
                else:
                    try:
                        last_candle = pd.to_datetime(df['open_time'].iloc[-1]).tz_localize(None)
                        if now_dt - last_candle > max_age:
                            is_stale = True
                    except Exception:
                        is_stale = True

                # If missing or stale, fetch newest candles
                if is_stale:
                    df = manager.load_data(sym, req.interval, auto_sync=True)

                if df is not None and not df.empty and len(df) >= req.filter_window:
                    df = df.tail(max(req.filter_window * 5, 5000))
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
                        row['adl_risk'] = float(adl_risks.get(sym, 0.0))

                        # Populate futures stats
                        sym_stats = api_stats_cache.get(sym, {})
                        oi_hist = sym_stats.get('oi_hist', {})
                        top_pos = sym_stats.get('top_pos', {})
                        top_acc = sym_stats.get('top_acc', {})
                        glob_acc = sym_stats.get('glob_acc', {})
                        taker_ratio = sym_stats.get('taker_ratio', {})

                        sum_oi = float(oi_hist.get("sumOpenInterest", 0)) if isinstance(oi_hist, dict) else 0.0
                        circ_supply = float(oi_hist.get("CMCCirculatingSupply", 0)) if isinstance(oi_hist, dict) else 0.0
                        row['oi_circulating'] = (sum_oi / circ_supply) if circ_supply > 0 else np.nan

                        row['top_ls_position_ratio'] = float(top_pos.get("longShortRatio", np.nan)) if isinstance(top_pos, dict) and "longShortRatio" in top_pos else np.nan
                        row['top_ls_account_ratio'] = float(top_acc.get("longShortRatio", np.nan)) if isinstance(top_acc, dict) and "longShortRatio" in top_acc else np.nan
                        row['global_ls_ratio'] = float(glob_acc.get("longShortRatio", np.nan)) if isinstance(glob_acc, dict) and "longShortRatio" in glob_acc else np.nan
                        row['taker_buysell_ratio'] = float(taker_ratio.get("buySellRatio", taker_ratio.get("longShortRatio", np.nan))) if isinstance(taker_ratio, dict) else np.nan

                        # Ensure all keys in ALL_METRICS are explicitly present in the row
                        for m in ALL_METRICS:
                            if m not in row:
                                row[m] = np.nan

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
