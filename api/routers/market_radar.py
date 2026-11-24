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


class OvervaluedScanRequest(BaseModel):
    symbols: Optional[List[str]] = None
    interval: str = "1h"
    win_price: int = 24
    win_oi: int = 24
    win_top_ls: int = 24
    win_funding: int = 24
    deviation_pct: float = 2.0
    w_price: float = 20.0
    w_oi: float = 20.0
    w_top_ls: float = 20.0
    w_ob: float = 20.0
    w_funding: float = 20.0


@router.post("/overvalued-scan")
def run_overvalued_scan(req: OvervaluedScanRequest):
    """
    5-Pillar Overvalued Shitcoin Scoring Framework on Asset Filter selection:
      1. Price Inflation (Price Z-Score + VAM, win_price)
      2. OI Expansion (OI USD Z-Score + VAM, win_oi)
      3. Top Trader Short Shift (inverted Top L/S Pos Z-Score + VAM, win_top_ls)
      4. Funding Short Pressure (negative or falling funding, win_funding)
      5. Orderbook Ask-Bias Trigger (-ob_imbalance on top candidates)
    """
    try:
        non_perp = {"QQQUSDT", "SPYUSDT", "XAUUSDT", "XAGUSDT"}

        vol_map = {}
        try:
            for item in fetcher.get_top_volume_symbols_with_meta(top_n=500):
                vol_map[item["symbol"]] = float(item.get("volume", 0.0))
        except Exception:
            pass

        raw_syms = req.symbols if req.symbols else manager.get_universe(top_n=50)
        symbols = [s for s in raw_syms if s not in non_perp]

        if not symbols:
            return {"candidates": [], "rate_limited": fetcher.is_rate_limited()}

        chg_24h_map = fetcher.get_24h_changes()
        live_funding_map = fetcher.get_funding_rates()
        max_win = max(req.win_price, req.win_oi, req.win_top_ls, req.win_funding, 24)
        series_limit = min(max(max_win * 4, 150), 500)

        # Pre-fetch OI, Top L/S Pos & Funding series in parallel (disk+mem cached)
        oi_cache: Dict[str, pd.Series] = {}
        top_cache: Dict[str, pd.Series] = {}
        fund_cache: Dict[str, pd.Series] = {}

        def fetch_stats_pair(sym):
            s_oi = fetcher.get_source_series(sym, "oi_usd", interval=req.interval, limit=series_limit)
            s_top = fetcher.get_source_series(sym, "top_ls_pos", interval=req.interval, limit=series_limit)
            s_fund = fetcher.get_source_series(sym, "funding", interval=req.interval, limit=min(series_limit, 100))
            return sym, s_oi, s_top, s_fund

        with ThreadPoolExecutor(max_workers=6) as executor:
            for sym, s_oi, s_top, s_fund in executor.map(fetch_stats_pair, symbols):
                oi_cache[sym] = s_oi
                top_cache[sym] = s_top
                fund_cache[sym] = s_fund

        stale_thresholds = {
            '1m': timedelta(minutes=5), '3m': timedelta(minutes=10), '5m': timedelta(minutes=15),
            '15m': timedelta(minutes=45), '30m': timedelta(hours=1), '1h': timedelta(hours=2),
            '2h': timedelta(hours=4), '4h': timedelta(hours=8), '6h': timedelta(hours=12),
            '8h': timedelta(hours=16), '12h': timedelta(hours=24), '1d': timedelta(days=2),
        }
        max_age = stale_thresholds.get(req.interval, timedelta(hours=2))
        now_dt = datetime.now()
        p_price = {"window": req.win_price}
        p_oi = {"window": req.win_oi}
        p_top = {"window": req.win_top_ls}
        p_fund = {"window": req.win_funding}

        def clip_score(val: float, scale: float) -> float:
            if not np.isfinite(val) or scale <= 0:
                return 0.0
            return float(np.clip(val / scale, 0.0, 1.0) * 100.0)

        # Stage 2-4: Compute Price Inflation, OI Expansion, Top Trader Short Shift
        pre_candidates = []
        def score_pre(sym):
            try:
                df = manager.load_data(sym, req.interval, auto_sync=False)
                is_stale = False
                if df is None or df.empty or len(df) < req.win_price:
                    is_stale = True
                else:
                    try:
                        last_c = pd.to_datetime(df['open_time'].iloc[-1]).tz_localize(None)
                        if now_dt - last_c > max_age:
                            is_stale = True
                    except Exception:
                        is_stale = True
                if is_stale:
                    df = manager.load_data(sym, req.interval, auto_sync=True)
                if df is None or df.empty:
                    return None

                p_z = engine.calculate_ohlcv_metric(df, "price_zscore", params=p_price, interval=req.interval)
                p_vam = engine.calculate_ohlcv_metric(df, "vam", params=p_price, interval=req.interval)
                s_price = clip_score(0.6 * p_z + 0.4 * p_vam, 3.0)

                s_oi_series = oi_cache.get(sym, pd.Series(dtype=float))
                oi_z = engine.calculate_series_metric(s_oi_series, "price_zscore", interval=req.interval, params=p_oi)
                oi_vam = engine.calculate_series_metric(s_oi_series, "vam", interval=req.interval, params=p_oi)
                s_oi = clip_score(0.6 * oi_z + 0.4 * oi_vam, 3.0)

                s_top_series = top_cache.get(sym, pd.Series(dtype=float))
                top_z = engine.calculate_series_metric(s_top_series, "price_zscore", interval=req.interval, params=p_top)
                top_vam = engine.calculate_series_metric(s_top_series, "vam", interval=req.interval, params=p_top)
                # Negative Z / VAM = Top traders shifting toward SHORT side
                s_top = clip_score(-(0.6 * top_z + 0.4 * top_vam), 2.5)

                # Funding Short Pressure: negative live funding OR decreasing funding over time
                live_fund = float(live_funding_map.get(sym, 0.0))
                s_fund_series = fund_cache.get(sym, pd.Series(dtype=float))
                fund_z = engine.calculate_series_metric(s_fund_series, "price_zscore", interval=req.interval, params=p_fund)
                fund_level_score = clip_score(-live_fund, 0.001)   # -0.10% funding => 100
                fund_trend_score = clip_score(-fund_z, 2.5)        # decreasing funding => 100
                s_funding = 0.5 * fund_level_score + 0.5 * fund_trend_score

                pre_w = max(req.w_price + req.w_oi + req.w_top_ls + req.w_funding, 1e-6)
                pre_score = (
                    req.w_price * s_price
                    + req.w_oi * s_oi
                    + req.w_top_ls * s_top
                    + req.w_funding * s_funding
                ) / pre_w

                return {
                    "symbol": sym,
                    "vol_24h_m": round(vol_map.get(sym, 0.0) / 1e6, 2),
                    "chg_24h": round(float(chg_24h_map.get(sym, 0.0)), 2),
                    "price_z": round(float(p_z), 3),
                    "oi_z": round(float(oi_z), 3),
                    "top_ls_z": round(float(top_z), 3),
                    "funding_pct": round(live_fund * 100.0, 4),
                    "funding_z": round(float(fund_z), 3),
                    "score_price": round(s_price, 1),
                    "score_oi": round(s_oi, 1),
                    "score_top_short": round(s_top, 1),
                    "score_funding": round(s_funding, 1),
                    "pre_score": pre_score,
                }
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            for res in executor.map(score_pre, symbols):
                if res is not None:
                    pre_candidates.append(res)

        pre_candidates.sort(key=lambda x: x["pre_score"], reverse=True)

        # Stage 5: Fetch Orderbook Imbalance ONLY for top 40 pre-scored candidates to avoid 429 bans
        ob_targets = [c["symbol"] for c in pre_candidates[:40]]
        ob_map: Dict[str, float] = {}
        dev_frac = float(req.deviation_pct) / 100.0

        def fetch_ob(sym):
            try:
                res = fetcher.get_books_status(sym, imbalance_pct=dev_frac, limit=100)
                return sym, res.get("orderbook_imbalance")
            except Exception:
                return sym, None

        with ThreadPoolExecutor(max_workers=4) as executor:
            for sym, imb in executor.map(fetch_ob, ob_targets):
                if imb is not None and np.isfinite(imb):
                    ob_map[sym] = float(imb)

        total_w = max(req.w_price + req.w_oi + req.w_top_ls + req.w_ob + req.w_funding, 1e-6)
        final_rows = []
        for c in pre_candidates:
            sym = c["symbol"]
            imb = ob_map.get(sym, 0.0)
            ask_bias = -imb  # negative imbalance = ask-side liquidity dominates
            s_ob = clip_score(ask_bias, 0.40) if sym in ob_map else 0.0

            total_score = (
                req.w_price * c["score_price"]
                + req.w_oi * c["score_oi"]
                + req.w_top_ls * c["score_top_short"]
                + req.w_ob * s_ob
                + req.w_funding * c["score_funding"]
            ) / total_w

            funding_ok = (c["score_funding"] >= 15.0) if req.w_funding > 0 else True
            if c["score_price"] >= 40 and c["score_oi"] >= 30 and c["score_top_short"] >= 30 and ask_bias >= 0.05 and funding_ok:
                signal = "TRIGGERED"
            elif c["score_price"] >= 40 and c["score_oi"] >= 30 and c["score_top_short"] >= 30:
                signal = "SETUP"
            elif total_score >= 45:
                signal = "WATCH"
            else:
                signal = "NEUTRAL"

            c["ob_imbalance"] = round(imb, 4) if sym in ob_map else None
            c["score_ob_ask"] = round(s_ob, 1)
            c["total_score"] = round(total_score, 1)
            c["signal"] = signal
            c.pop("pre_score", None)
            final_rows.append(c)

        final_rows.sort(key=lambda x: x["total_score"], reverse=True)

        import time
        rate_limited = fetcher.is_rate_limited()
        cooldown_sec = max(0, int(fetcher._rate_limited_until - time.time())) if rate_limited else 0
        return {
            "candidates": final_rows,
            "rate_limited": rate_limited,
            "cooldown_sec": cooldown_sec,
        }
    except Exception as e:
        logger.error(f"Overvalued scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
