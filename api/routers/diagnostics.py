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
                "imbalance": np_safe(book_status.get("orderbook_imbalance", 0))                
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
                "ohlcv": ohlcv
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
