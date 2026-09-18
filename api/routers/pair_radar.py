from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.stattools import coint, adfuller
from scipy import stats
from scipy.stats import gaussian_kde, rankdata

from src.data import DataManager
from src.metrics import MetricsEngine, copula_cond_probs
from src.logger import logger
from src.shared_state import get_manager, get_engine

router = APIRouter()
manager = get_manager()
engine = get_engine()

class PairRequest(BaseModel):
    symbol_a: str
    symbol_b: str
    interval: str = "1h"
    mode: str = "spread" # ratio or spread
    rolling_window: int = 320
    pair_window: int = 500
    
    # Copula Config
    copula_mode: str = "price" # "price" or "log_rets"
    copula_type: str = "t" # gaussian, t, clayton, gumbel
    copula_param: float = 2.0
    r_window: int = 10
    copula_stationarize: bool = False
    copula_ema_window: int = 20

def calculate_rolling_zscore(series, window):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.ewm(span=window).std().replace(0, 1e-9)
    return (series - rolling_mean) / rolling_std

def get_kde_path(vals, ranks):
    if len(vals) < 5: return np.zeros_like(ranks).tolist()
    kde = gaussian_kde(vals, bw_method=0.3)
    dens = kde.evaluate(vals)
    if dens.max() > 0: dens = dens / dens.max()
    return dens.tolist()

@router.post("/generate")
def generate_pair_radar(req: PairRequest):
    try:
        df_a = manager.load_data(req.symbol_a, req.interval, auto_sync=False)
        df_b = manager.load_data(req.symbol_b, req.interval, auto_sync=False)
        
        if df_a is None or df_b is None or df_a.empty or df_b.empty:
            raise HTTPException(status_code=400, detail="Data missing for symbols")
            
        df_a = df_a.set_index("open_time")
        df_b = df_b.set_index("open_time")
        common = df_a.index.intersection(df_b.index)
        
        if len(common) < req.rolling_window:
            raise HTTPException(status_code=400, detail="Insufficient common data")
            
        common = common[-(req.rolling_window + req.pair_window):]
        df_a = df_a.loc[common].copy()
        df_b = df_b.loc[common].copy()

        price_cols = ['open', 'high', 'low', 'close']
        df_a[price_cols] = df_a[price_cols].clip(lower=1e-9)
        df_b[price_cols] = df_b[price_cols].clip(lower=1e-9)

        raw_a = df_a.copy()
        raw_b = df_b.copy()

        df_a[price_cols] = np.log(df_a[price_cols])
        df_b[price_cols] = np.log(df_b[price_cols])

        synthetic = pd.DataFrame(index=common)
        if df_b["close"].std() == 0:
            raise HTTPException(status_code=400, detail="Asset B has zero variance")

        slope, intercept, r_val, p_val, std_err = stats.linregress(df_b["close"], df_a["close"])
        beta = slope
        
        if req.mode == "ratio":
            synthetic["open"] = raw_a["open"] / raw_b["open"]
            synthetic["high"] = raw_a["high"] / raw_b["high"]
            synthetic["low"] = raw_a["low"] / raw_b["low"]
            synthetic["close"] = raw_a["close"] / raw_b["close"]
        else:
            synthetic["open"] = df_a["open"] - (intercept + beta * df_b["open"])
            synthetic["high"] = df_a["high"] - (intercept + beta * df_b["high"])
            synthetic["low"] = df_a["low"] - (intercept + beta * df_b["low"])
            synthetic["close"] = df_a["close"] - (intercept + beta * df_b["close"])

        synthetic["price_a"] = df_a["close"] # log price
        synthetic["price_b"] = df_b["close"] # log price
        synthetic["log_ret_a"] = df_a["close"].diff()
        synthetic["log_ret_b"] = df_b["close"].diff()
        
        # Safe std calc
        std_a, std_b = synthetic["log_ret_a"].std(), synthetic["log_ret_b"].std()
        vol_Ratio = std_a / std_b if std_b != 0 else 0

        y = df_a["close"].values
        x = df_b["close"].values
        residuals = y - (slope * x + intercept)
        if residuals.var() == 0:
            adf_stat, adf_p = 0, 0
        else:
            adf_stat, adf_p, _, _, _, _ = adfuller(residuals)

        r2 = r_val**2
        # Bollinger Bands for synthetic spread
        synthetic["ma"] = synthetic["close"].rolling(window=20).mean()
        synthetic["std"] = synthetic["close"].rolling(window=20).std()
        synthetic["bb_up"] = synthetic["ma"] + (synthetic["std"] * 2)
        synthetic["bb_dn"] = synthetic["ma"] - (synthetic["std"] * 2)
        synthetic = synthetic.bfill()

        z_local = calculate_rolling_zscore(synthetic["close"], req.rolling_window)
        synthetic["zscore"] = z_local
        
        # --- COPULA CALCULATIONS ---
        copula_data = {}
        try:
            w = max(req.pair_window, 10)
            df_plot = synthetic.tail(w).copy()
            r_window = max(req.r_window, 1)

            if req.copula_mode == "price":
                p_a_full = np.exp(synthetic["price_a"])
                p_b_full = np.exp(synthetic["price_b"])

                if req.copula_stationarize:
                    ew = max(req.copula_ema_window, 2)
                    x_full = p_a_full / p_a_full.ewm(span=ew).mean()
                    y_full = p_b_full / p_b_full.ewm(span=ew).mean()
                    title_suffix = f" (Stationary Price, {ew} EMA)"
                else:
                    x_full = p_a_full
                    y_full = p_b_full
                    title_suffix = " (Price Levels)"
                
                x_raw = x_full.tail(w).values
                y_raw = y_full.tail(w).values
            else:
                df_plot = df_plot.dropna(subset=["log_ret_a", "log_ret_b"])
                if len(df_plot) < 2 or r_window < 2:
                    x_raw = df_plot["log_ret_a"].values
                    y_raw = df_plot["log_ret_b"].values
                    title_suffix = " (Raw Log Returns)"
                else:
                    x_raw = df_plot["log_ret_a"].rolling(r_window).sum().dropna().values
                    y_raw = df_plot["log_ret_b"].rolling(r_window).sum().dropna().values
                    title_suffix = " (Grouped Returns)"

            if len(x_raw) > 5:
                cx = np.where(np.isfinite(x_raw), x_raw, 0)
                cy = np.where(np.isfinite(y_raw), y_raw, 0)
                
                u_hist = rankdata(cx) / (len(cx) + 1)
                v_hist = rankdata(cy) / (len(cy) + 1)

                u_curr = u_hist[-1]
                v_curr = v_hist[-1]

                kwargs = {}
                if req.copula_type == "t": kwargs["df"] = req.copula_param
                else: kwargs["theta"] = req.copula_param

                p_uv, p_vu = copula_cond_probs(u_hist, v_hist, u_curr, v_curr, method=req.copula_type, **kwargs)

                dens_a = get_kde_path(cx, u_hist)
                dens_b = get_kde_path(cy, v_hist)

                copula_data = {
                    "u": u_hist.tolist(), "v": v_hist.tolist(),
                    "u_curr": float(u_curr), "v_curr": float(v_curr),
                    "x": cx.tolist(), "y": cy.tolist(),
                    "p_uv": float(p_uv), "p_vu": float(p_vu),
                    "dens_a": dens_a, "dens_b": dens_b,
                    "title_suffix": title_suffix
                }
        except Exception as e:
            logger.error(f"Copula extraction failed: {e}")

        # --- ASSET COMPARISON ---
        comp_data = {}
        try:
            c_df = synthetic.tail(req.pair_window).copy()
            c_ret_a = c_df["log_ret_a"].dropna()
            c_ret_b = c_df["log_ret_b"].dropna()
            
            # Volatility scale ratio
            c_beta = c_ret_a.std() / c_ret_b.std() if c_ret_b.std() > 0 else 1
            cum_ret_a_scaled = (c_ret_a / c_beta).cumsum().tolist()
            cum_ret_b_scaled = c_ret_b.cumsum().tolist()
            
            comp_data = {
                "cum_ret_a": cum_ret_a_scaled,
                "cum_ret_b": cum_ret_b_scaled,
                "ts": c_ret_a.index.astype(str).tolist()
            }
        except Exception as e:
             logger.error(f"Comp extraction failed: {e}")

        # Basic chart data
        chart_data = synthetic.tail(req.pair_window).reset_index()
        chart_data["open_time"] = chart_data["open_time"].astype(str)
        chart_data = chart_data.replace([np.inf, -np.inf], None).where(pd.notnull(chart_data), None)

        def np_safe(v):
            return None if pd.isna(v) or np.isinf(v) else float(v)

        return {
            "metrics": {
                "Coefficient": np_safe(slope),
                "VolRatio": np_safe(vol_Ratio),
                "ADF_P": np_safe(adf_p),
                "R2": np_safe(r2)
            },
            "chart_data": chart_data.to_dict(orient="records"),
            "copula": copula_data,
            "comp": comp_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
