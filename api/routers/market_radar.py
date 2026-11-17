from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from src.data import DataManager, BinanceFuturesFetcher
from src.metrics import MetricsEngine
from src.config import BENCHMARK_SYMBOL, ALL_METRICS
from src.shared_state import get_manager, get_engine
from src.cache_store import MetricCacheStore
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

router = APIRouter()
logger = logging.getLogger(__name__)

manager = get_manager()
engine = get_engine()
fetcher = BinanceFuturesFetcher()
metric_cache = MetricCacheStore()

class AxisConfig(BaseModel):
    source: str = "ohlcv"
    metric: str = "price_zscore"
    interval: str = "1h"
    params: Dict[str, Any] = {}

class SnapshotRequest(BaseModel):
    symbols: List[str]
    x: AxisConfig
    y: AxisConfig
    z: Optional[AxisConfig] = None
    force_refresh: bool = False

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
    """Calculate ONLY the chosen per-axis metrics with cache-first lookup."""
    try:
        axes_spec = [
            ("x_val", req.x),
            ("y_val", req.y),
        ]
        if req.z and req.z.metric and req.z.metric != "None":
            axes_spec.append(("z_val", req.z))

        # 0. Check computed MetricCacheStore first for every (symbol, axis)
        cached_rows: Dict[str, Dict[str, float]] = {sym: {} for sym in req.symbols}
        missing_by_sym: Dict[str, List[tuple]] = {sym: [] for sym in req.symbols}
        cache_hits = 0
        cache_misses = 0

        for sym in req.symbols:
            for col_name, ax in axes_spec:
                key = metric_cache.make_key(sym, ax.source, ax.metric, ax.interval, ax.params)
                ttl = metric_cache.get_ttl(ax.source, ax.interval)
                hit = None if req.force_refresh else metric_cache.get(key, ttl)
                if hit is not None:
                    cached_rows[sym][col_name] = hit
                    cache_hits += 1
                else:
                    missing_by_sym[sym].append((col_name, ax, key))
                    cache_misses += 1

        # Determine which sources/metrics/intervals actually have misses
        missing_axes = [item for sym_list in missing_by_sym.values() for item in sym_list]
        missing_syms = [sym for sym, lst in missing_by_sym.items() if lst]

        # 1. Pre-fetch Snapshot / Microstructure data ONLY for missing axes
        snapshot_metrics_chosen = set(ax.metric for _, ax, _ in missing_axes if ax.source == "snapshot")
        funding_rates = fetcher.get_funding_rates() if "funding_rate" in snapshot_metrics_chosen else {}
        adl_risks = fetcher.get_all_adl_risks() if "adl_risk" in snapshot_metrics_chosen else {}
        chg_24h_map = fetcher.get_24h_changes() if "chg_24h" in snapshot_metrics_chosen else {}

        # Pre-warm raw orderbooks ONLY if orderbook_imbalance or impact_spread has misses
        need_book = any(m in snapshot_metrics_chosen for m in ("orderbook_imbalance", "impact_spread"))
        book_syms = [sym for sym in missing_syms if any(ax.source == "snapshot" and ax.metric in ("orderbook_imbalance", "impact_spread") for _, ax, _ in missing_by_sym[sym])]
        book_limit = 100 if len(book_syms) > 80 else 500
        if need_book and book_syms:
            def warm_book(sym):
                try:
                    fetcher.get_orderbooks(sym, limit=book_limit)
                except Exception:
                    pass
            workers = 4 if len(book_syms) > 80 else 8
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(warm_book, book_syms))

        # Pre-fetch oi_circulating ONLY if chosen and missing
        oi_circ_cache = {}
        if "oi_circulating" in snapshot_metrics_chosen:
            oi_syms = [sym for sym in missing_syms if any(ax.source == "snapshot" and ax.metric == "oi_circulating" for _, ax, _ in missing_by_sym[sym])]
            def fetch_oi_circ(sym):
                try:
                    d = fetcher.get_historical_stats(sym, "1h", fetcher.OPEN_INTEREST_HIST)
                    oi = float(d.get("sumOpenInterest", 0)) if isinstance(d, dict) else 0.0
                    sup = float(d.get("CMCCirculatingSupply", 0)) if isinstance(d, dict) else 0.0
                    return sym, (oi / sup if sup > 0 else 0.0)
                except Exception:
                    return sym, 0.0
            with ThreadPoolExecutor(max_workers=8) as executor:
                for sym, val in executor.map(fetch_oi_circ, oi_syms):
                    oi_circ_cache[sym] = val

        # 2. Pre-fetch Non-OHLCV Time-Series grouped by (source, interval) ONLY for missing
        non_ohlcv_tasks = set()
        key_limits = {}
        for sym, lst in missing_by_sym.items():
            for _, ax, _ in lst:
                if ax.source not in ("ohlcv", "snapshot"):
                    k = (ax.source, ax.interval)
                    non_ohlcv_tasks.add((sym, k))
                    w = int((ax.params or {}).get("window", ax.params.get("w_slow", 40) if ax.params else 40))
                    key_limits[k] = max(key_limits.get(k, 0), min(max(w * 4, 200), 500))

        series_cache: Dict[tuple, Dict[str, pd.Series]] = {k: {} for k in key_limits}
        if non_ohlcv_tasks:
            def fetch_ts(task):
                sym, (src, inter) = task
                try:
                    s = fetcher.get_source_series(sym, src, interval=inter, limit=key_limits[(src, inter)])
                    return sym, (src, inter), s
                except Exception:
                    return sym, (src, inter), pd.Series(dtype=float)

            with ThreadPoolExecutor(max_workers=8) as executor:
                for fut in as_completed([executor.submit(fetch_ts, t) for t in non_ohlcv_tasks]):
                    sym, k, s = fut.result()
                    series_cache[k][sym] = s

        # 3. Pre-load BTC benchmark ONLY for (ohlcv, interval) misses that need rel_strength_z / beta_btc
        benchmark_map = {}
        for _, ax, _ in missing_axes:
            if ax.source == "ohlcv" and ax.metric in ("rel_strength_z", "beta_btc") and ax.interval not in benchmark_map:
                bdf = manager.load_data(BENCHMARK_SYMBOL, ax.interval, auto_sync=False)
                if bdf is None or bdf.empty:
                    try:
                        bdf = manager.load_data(BENCHMARK_SYMBOL, ax.interval, auto_sync=True)
                    except Exception:
                        pass
                if bdf is not None and not bdf.empty:
                    b_close = pd.to_numeric(bdf['close'], errors='coerce').ffill().fillna(0)
                    benchmark_map[ax.interval] = (b_close.pct_change().dropna(), b_close)

        stale_thresholds = {
            '1m': timedelta(minutes=5), '3m': timedelta(minutes=10), '5m': timedelta(minutes=15),
            '15m': timedelta(minutes=45), '30m': timedelta(hours=1), '1h': timedelta(hours=2),
            '2h': timedelta(hours=4), '4h': timedelta(hours=8), '6h': timedelta(hours=12),
            '8h': timedelta(hours=16), '12h': timedelta(hours=24), '1d': timedelta(days=2),
            '3d': timedelta(days=4), '1w': timedelta(days=8)
        }
        now_dt = datetime.now()

        def load_ohlcv_for_symbol(sym, inter):
            df = manager.load_data(sym, inter, auto_sync=False)
            max_age = stale_thresholds.get(inter, timedelta(hours=2))
            is_stale = False
            if df is None or df.empty:
                is_stale = True
            else:
                try:
                    last_candle = pd.to_datetime(df['open_time'].iloc[-1]).tz_localize(None)
                    if now_dt - last_candle > max_age:
                        is_stale = True
                except Exception:
                    is_stale = True
            if is_stale:
                df = manager.load_data(sym, inter, auto_sync=True)
            return df

        def process_symbol(sym):
            try:
                row = {'symbol': sym}
                row.update(cached_rows[sym])

                misses = missing_by_sym[sym]
                if misses:
                    sym_ohlcv_intervals = set(ax.interval for _, ax, _ in misses if ax.source == "ohlcv")
                    sym_ohlcv_cache = {inter: load_ohlcv_for_symbol(sym, inter) for inter in sym_ohlcv_intervals}

                    for col_name, ax, cache_key in misses:
                        if ax.source == "snapshot":
                            ax_params = ax.params or {}
                            if ax.metric == "orderbook_imbalance":
                                dev_pct = float(ax_params.get("deviation_pct", 2.0)) / 100.0
                                book_res = fetcher.get_books_status(sym, imbalance_pct=dev_pct, limit=book_limit)
                                val = book_res.get("orderbook_imbalance", 0.0)
                            elif ax.metric == "impact_spread":
                                imp_usd = float(ax_params.get("impact_usd", 1000))
                                book_res = fetcher.get_books_status(sym, impact_usd=imp_usd, limit=book_limit)
                                val = book_res.get("impact_spread", 0.0)
                            elif ax.metric == "adl_risk":
                                val = float(adl_risks.get(sym, 0.0))
                            elif ax.metric == "chg_24h":
                                val = float(chg_24h_map.get(sym, 0.0))
                            elif ax.metric == "funding_rate":
                                val = float(funding_rates.get(sym, 0.0))
                            elif ax.metric == "oi_circulating":
                                val = oi_circ_cache.get(sym, 0.0)
                            else:
                                val = 0.0
                        elif ax.source == "ohlcv":
                            df_sym = sym_ohlcv_cache.get(ax.interval)
                            b_ret, b_prc = benchmark_map.get(ax.interval, (None, None))
                            val = engine.calculate_ohlcv_metric(
                                df_sym, ax.metric, params=ax.params, interval=ax.interval,
                                benchmark_returns=b_ret, benchmark_prices=b_prc
                            )
                        else:
                            s = series_cache.get((ax.source, ax.interval), {}).get(sym, pd.Series(dtype=float))
                            val = engine.calculate_series_metric(
                                s, ax.metric, interval=ax.interval, params=ax.params
                            )

                        clean_val = float(val) if val is not None and pd.notna(val) and not np.isinf(val) else 0.0
                        row[col_name] = clean_val

                        # Cache only valid non-rate-limited results
                        if not fetcher.is_rate_limited():
                            metric_cache.set(cache_key, clean_val)

                if "z_val" not in row:
                    row["z_val"] = 0.0

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
            
        import time
        rate_limited = fetcher.is_rate_limited()
        cooldown_sec = max(0, int(fetcher._rate_limited_until - time.time())) if rate_limited else 0
        return {
            "metrics": cleaned_results,
            "rate_limited": rate_limited,
            "cooldown_sec": cooldown_sec,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses
        }
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
