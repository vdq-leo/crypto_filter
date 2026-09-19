from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

from src.data import DataManager
from src.metrics import MetricsEngine
from src.config import BENCHMARK_SYMBOL, MANDATORY_CRYPTO, IGNORED_CRYPTO
from src.shared_state import get_manager, get_engine
from ml_engine.analysis.multivariate import DecompositionEngine
from ml_engine.data.bars import construct_volume_bars, construct_dollar_bars, calibrate_bar_threshold
from ml_engine.labeling.labeler import Labeler
from scipy.stats import skew, kurtosis
from concurrent.futures import ThreadPoolExecutor
router = APIRouter()
manager = get_manager()
engine = get_engine()

class DiagnosticsRequest(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    metric_window: int = 10
    diag_window: int = 100

def _forecast_garch(returns, steps=10, ann_factor=1):
    from arch import arch_model
    r = np.asarray(returns)
    model = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, rescale=True)
    res = model.fit(disp="off")
    hist_vol = res.conditional_volatility * np.sqrt(ann_factor)
    fc = res.forecast(horizon=steps)
    fc_var = fc.variance.values[-1]
    fc_vol = np.sqrt(fc_var) * np.sqrt(ann_factor)
    return fc_vol, hist_vol

@router.post("/run")
def run_diagnostics(req: DiagnosticsRequest):
    try:
        df = manager.load_data(req.symbol, req.interval, auto_sync=False)
        bench_df = manager.load_data(BENCHMARK_SYMBOL, req.interval, auto_sync=False)
        
        if df is None or df.empty or len(df) < req.diag_window:
            raise HTTPException(status_code=400, detail="Insufficient data for analysis")
            
        prices = pd.to_numeric(df['close'], errors='coerce').ffill().values
        log_rets = np.diff(np.log(prices))
        log_rets = np.nan_to_num(log_rets)
        
        bench_rets = None
        if bench_df is not None and not bench_df.empty:
            b_prices = pd.to_numeric(bench_df['close'], errors='coerce').ffill().values
            bench_rets = np.diff(np.log(b_prices))
            min_len = min(len(log_rets), len(bench_rets))
            log_rets_aligned = log_rets[-min_len:]
            bench_rets_aligned = bench_rets[-min_len:]

        # 1. Base Performance Metrics
        res_sharpe = engine.calculate_sharpe_ratio(log_rets, interval=req.interval)
        res_sortino = engine.calculate_sortino_ratio(log_rets, interval=req.interval)
        res_maxdd = engine.calculate_max_drawdown(prices)
        res_avgdd = engine.calculate_avg_drawdown(prices)
        
        var_threshold = np.percentile(log_rets, 5)
        cvar = log_rets[log_rets <= var_threshold].mean()
        
        ann_factor = engine.get_annual_scaling(req.interval)
        volatility = np.std(log_rets) * np.sqrt(ann_factor)
        
        threshold = 0
        gains = log_rets[log_rets > threshold].sum()
        losses = np.abs(log_rets[log_rets <= threshold].sum())
        omega_ratio = gains / losses if losses != 0 else 0
        
        # Exposure
        beta_, alpha_, r2 = 0, 0, 0
        if bench_rets is not None:
            beta_, alpha_, r2 = engine.calculate_beta_alpha(log_rets_aligned, bench_rets_aligned)

        # Book status
        try:
            book_status = manager.fetcher.get_books_status(req.symbol)
        except:
            book_status = {}

        # 2. Radial Chart Metrics
        try:
            latest_metrics = engine.calculate_all_indicators(
                df.iloc[-req.diag_window * 2:], 
                benchmark_returns=np.log(bench_df['close']).diff()[-req.diag_window * 2:] if bench_df is not None else None,
                interval=req.interval,
                window=req.metric_window 
            )
            col_droped = ["volatility"]
            for col in col_droped:
                if col in latest_metrics.columns:
                    latest_metrics = latest_metrics.drop(columns=[col])
            latest_metrics = latest_metrics.dropna(axis=1, how="all").dropna(axis=0, how="all")
            
            def z_score(x): return (x - x.mean()) / (x.std() + 1e-9)
            for col in latest_metrics.columns:
                latest_metrics[col] = z_score(latest_metrics[col])
                
            metrics_df = latest_metrics.tail(1).T.reset_index()
            metrics_df.columns = ["Metric", "Value"]
            metrics_list = metrics_df.to_dict(orient="records")
        except Exception as e:
            metrics_list = []

        # 3. Market Neutral Cum Ret
        mn_cum_ret = []
        try:
            market_data = {}
            for s in MANDATORY_CRYPTO:
                if s in IGNORED_CRYPTO: continue
                d = manager.load_data(s, req.interval, auto_sync=False)
                if d is not None: market_data[s] = pd.to_numeric(d['close'], errors='coerce').pct_change()
            
            factor_df = pd.DataFrame(market_data).ffill().fillna(0).tail(req.diag_window)
            if factor_df.shape[1] > 2:
                decomp_res = DecompositionEngine.k_factor_decompose(factor_df, k=5)
                sym_series = pd.Series(log_rets[-req.diag_window:], index=factor_df.index)
                pc1 = decomp_res['factor_returns']['PC1'].values.reshape(-1, 1)
                y = sym_series.values
                
                lr = LinearRegression()
                lr.fit(pc1, y)
                residuals = y - lr.predict(pc1)
                mn_cum_ret = np.cumsum(residuals).tolist()
        except Exception as e:
            pass

        # 4. Forecasts
        history_price = prices[-req.diag_window - 10:].tolist()
        fc_mean = []
        fc_ci_lower = []
        fc_ci_upper = []
        try:
            log_history = np.log(history_price)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(log_history, order=(2,1,2))
                model_fit = model.fit()
                fc_res = model_fit.get_forecast(steps=10)
            
            fc_mean = np.exp(fc_res.predicted_mean).tolist()
            ci = np.exp(fc_res.conf_int(alpha=0.05))
            fc_ci_lower = ci.iloc[:, 0].tolist()
            fc_ci_upper = ci.iloc[:, 1].tolist()
        except:
            last_p = history_price[-1] if history_price else 0
            fc_mean = [last_p]*10
            fc_ci_lower = [last_p * 0.99]*10
            fc_ci_upper = [last_p * 1.01]*10

        # GARCH Vol
        hist_vol = []
        fc_vol = []
        try:
            garch_data = log_rets[-req.diag_window:]
            vol_forecast_vals, hv = _forecast_garch(garch_data, steps=10, ann_factor=ann_factor)
            hist_vol = hv.tolist()
            fc_vol = vol_forecast_vals.tolist()
        except:
            hist_vol = [0]*req.diag_window
            fc_vol = [0]*10

        # 5. OHLCV for Candlestick + BB
        ohlcv = []
        try:
            chart_df = df.tail(req.diag_window).copy()
            # Calculate basic BB manually just in case
            window = 20
            num_std_dev = 2
            chart_df['ma'] = chart_df['close'].rolling(window=window).mean()
            chart_df['std'] = chart_df['close'].rolling(window=window).std()
            chart_df['bb_up'] = chart_df['ma'] + (chart_df['std'] * num_std_dev)
            chart_df['bb_dn'] = chart_df['ma'] - (chart_df['std'] * num_std_dev)
            
            # Fill NaNs from rolling
            chart_df = chart_df.bfill()
            
            for _, row in chart_df.iterrows():
                ohlcv.append({
                    "time": str(row['open_time']) if 'open_time' in chart_df.columns else str(row.name),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume']),
                    "bb_up": float(row['bb_up']),
                    "bb_dn": float(row['bb_dn']),
                    "bb_mid": float(row['ma'])
                })
        except Exception as e:
            ohlcv = []

        # 6. Regime Classification
        curr_regime = "Sideways/Neutral"
        try:
            l_algo = Labeler(amplitude_threshold=0.01, max_inactive_period=10)
            lbl_df = l_algo.label(prices[-req.diag_window:])
            curr_lbl_val = lbl_df['label'].iloc[-1]
            if curr_lbl_val == 1: curr_regime = "Uptrend"
            elif curr_lbl_val == -1: curr_regime = "Downtrend"
            
            lbl_status = curr_regime
            lbl_labels = lbl_df['label'].tolist()
            lbl_prices = lbl_df['price'].tolist()
        except:
            lbl_status = "Unknown"
            lbl_labels = []
            lbl_prices = []

        # 7. ADL Risk
        try:
            adl_risk_map = manager.fetcher.get_all_adl_risks()
            current_adl_risk = adl_risk_map.get(req.symbol, 0)
        except:
            current_adl_risk = 0

        # 8. Binance Trading Stats
        period_mapping = {
            '1m': '5m', '3m': '5m', '5m': '5m', '15m': '15m',
            '30m': '30m', '1h': '1h', '2h': '2h', '4h': '4h',
            '6h': '6h', '8h': '6h', '12h': '12h', '1d': '1d',
            '3d': '1d', '1w': '1d', '1M': '1d'
        }
        period = period_mapping.get(req.interval, '1h')
        limit = min(req.diag_window, 500)
        
        try:
            with ThreadPoolExecutor(max_workers=6) as executor:
                f_oi = executor.submit(manager.fetcher.get_historical_stats_series, req.symbol, period, manager.fetcher.OPEN_INTEREST_HIST, limit=limit)
                f_pos = executor.submit(manager.fetcher.get_historical_stats_series, req.symbol, period, manager.fetcher.TOP_LS_POSITION, limit=limit)
                f_acc = executor.submit(manager.fetcher.get_historical_stats_series, req.symbol, period, manager.fetcher.TOP_LS_ACCOUNT, limit=limit)
                f_g_acc = executor.submit(manager.fetcher.get_historical_stats_series, req.symbol, period, manager.fetcher.GLOBAL_LS_ACCOUNT, limit=limit)
                f_taker = executor.submit(manager.fetcher.get_historical_stats_series, req.symbol, period, manager.fetcher.TAKER_BUY_SELL, limit=limit)
                f_fund = executor.submit(manager.fetcher.get_historical_funding_rate, req.symbol, limit=limit)
                
                oi_hist = f_oi.result()
                top_pos = f_pos.result()
                top_acc = f_acc.result()
                glob_acc = f_g_acc.result()
                taker_ratio = f_taker.result()
                fund_hist = f_fund.result()
        except:
            oi_hist = []
            top_pos = []
            top_acc = []
            glob_acc = []
            taker_ratio = []
            fund_hist = []

        def np_safe(v):
            return None if pd.isna(v) or np.isinf(v) else float(v)

        return {
            "performance": {
                "sharpe": np_safe(res_sharpe),
                "sortino": np_safe(res_sortino),
                "maxdd": np_safe(res_maxdd),
                "avgdd": np_safe(res_avgdd),
                "cvar": np_safe(cvar),
                "volatility": np_safe(volatility),
                "omega": np_safe(omega_ratio),
                "beta": np_safe(beta_),
                "alpha": np_safe(alpha_),
                "impact_spread": np_safe(book_status.get("impact_spread", 0)),
                "imbalance": np_safe(book_status.get("orderbook_imbalance", 0)),
                "adl_risk": current_adl_risk
            },
            "charts": {
                "metrics": metrics_list,
                "mn_cum_ret": mn_cum_ret,
                "prices": {
                    "hist": history_price,
                    "forecast": fc_mean,
                    "ci_lower": fc_ci_lower,
                    "ci_upper": fc_ci_upper
                },
                "volatility": {
                    "hist": hist_vol,
                    "forecast": fc_vol
                },
                "regime": {
                    "status": lbl_status,
                    "labels": lbl_labels,
                    "prices": lbl_prices
                },
                "ts_oi": oi_hist,
                "ts_top_pos": top_pos,
                "ts_top_acc": top_acc,
                "ts_glob_acc": glob_acc,
                "ts_taker": taker_ratio,
                "ts_fund": fund_hist,
                "ohlcv": ohlcv
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BarsRequest(BaseModel):
    symbol: str
    interval: str
    vol_th: float
    dollar_th: float

@router.post("/bars")
def generate_bars(req: BarsRequest):
    try:
        df = manager.load_data(req.symbol, req.interval, auto_sync=False)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No data available")
        
        if 'open_time' in df.columns:
            df = df.set_index(pd.to_datetime(df['open_time']))
            
        time_df = df.copy()
        time_df['ret'] = np.log(time_df['close'] / time_df['close'].shift(1))
        
        vol_df = construct_volume_bars(df, req.vol_th)
        if not vol_df.empty:
            vol_df['ret'] = np.log(vol_df['close'] / vol_df['close'].shift(1))
            
        dollar_df = construct_dollar_bars(df, req.dollar_th)
        if not dollar_df.empty:
            dollar_df['ret'] = np.log(dollar_df['close'] / dollar_df['close'].shift(1))
            
        res = {}
        for name, d in [("time", time_df), ("volume", vol_df), ("dollar", dollar_df)]:
            if d is not None and not d.empty:
                d = d.replace([np.inf, -np.inf], None).where(pd.notnull(d), None)
                res[name] = {
                    "data": d.reset_index().to_dict(orient="records"),
                    "stats": {
                        "count": len(d),
                        "skew": skew(d['ret'].dropna()) if len(d['ret'].dropna()) > 1 else 0,
                        "kurtosis": kurtosis(d['ret'].dropna(), fisher=True) if len(d['ret'].dropna()) > 1 else 0
                    }
                }
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CalibrateRequest(BaseModel):
    symbol: str
    interval: str

@router.post("/calibrate-bars")
def calibrate_bars(req: CalibrateRequest):
    try:
        df = manager.load_data(req.symbol, req.interval, auto_sync=False)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No data available")
        
        if 'open_time' in df.columns:
            df = df.set_index(pd.to_datetime(df['open_time']))
            
        opt_vol = calibrate_bar_threshold(df, "Volume Bars")
        opt_dollar = calibrate_bar_threshold(df, "Dollar Bars")
        
        return {
            "vol_th": opt_vol,
            "dollar_th": opt_dollar
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
