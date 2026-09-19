from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.shared_state import get_manager
from src.portfolio.estimators import HorizonShiftedOLS, CorrelationBlender, build_covariance
from src.portfolio.optimizers import PortfolioOptimizer, VolatilityTargeter

def get_periods_per_year(interval: str) -> float:
    mapping = {
        '1m': 365 * 24 * 60, '3m': 365 * 24 * 20, '5m': 365 * 24 * 12,
        '15m': 365 * 24 * 4, '30m': 365 * 24 * 2, '1h': 365 * 24,
        '2h': 365 * 12, '4h': 365 * 6, '6h': 365 * 4, '8h': 365 * 3,
        '12h': 365 * 2, '1d': 365, '3d': 365 / 3, '1w': 52
    }
    return mapping.get(interval, 365)

router = APIRouter()
manager = get_manager()

class EstimateRequest(BaseModel):
    symbols: List[str]
    estimation_method: str = "ols"
    interval: str = "1d"
    target_horizon: int = 7
    short_window: int = 14
    long_window: int = 60
    correlation_alpha: float = 0.4
    split_ratio: float = 0.6

class OptimizeRequest(BaseModel):
    mu: Dict[str, float]
    sigma: Dict[str, Dict[str, float]]
    method: str = "max_sharpe"
    risk_free_rate: float = 0.0
    target_volatility: float = 0.15
    long_only: bool = True
    min_weight: float = 0.0
    max_weight: float = 1.0

@router.post("/estimate")
def estimate_parameters(req: EstimateRequest):
    try:
        data_dict = {}
        for sym in req.symbols:
            df = manager.load_data(sym, req.interval, auto_sync=True)
            if df is not None and not df.empty:
                # use full available history for robust walk-forward
                prices = pd.to_numeric(df['close'], errors='coerce').ffill().dropna()
                prices.index = df.loc[prices.index, 'open_time']
                prices = prices[~prices.index.duplicated(keep='last')]
                data_dict[sym] = prices

        if not data_dict:
            raise HTTPException(status_code=400, detail="No data available for requested symbols.")

        # Align data on datetime index and forward fill any missing gaps in merged timeframe
        prices_df = pd.DataFrame(data_dict).sort_index().ffill().dropna(how='all')

        results = {}
        vols = {}
        
        ppy = get_periods_per_year(req.interval)
        ret_annualizer = ppy / req.target_horizon if req.estimation_method == 'ols' else ppy
        vol_annualizer = np.sqrt(ppy)
        
        if req.estimation_method == 'historical':
            returns_df = prices_df.pct_change().dropna(how='all')
            # For historical maxdd
            cum_returns = (1 + returns_df).cumprod()
            rolling_max = cum_returns.cummax()
            drawdowns = cum_returns / rolling_max - 1
            maxdd_dict = drawdowns.min().to_dict()
            
            for sym in prices_df.columns:
                ret = returns_df[sym].dropna()
                ann_ret = ret.mean() * ret_annualizer if not ret.empty else 0.0
                ann_vol = ret.std() * vol_annualizer if not ret.empty else 0.0
                mdd = maxdd_dict.get(sym, 0.0)
                
                results[sym] = {
                    'expected_return': ann_ret,
                    'expected_volatility': ann_vol,
                    'expected_maxdd': mdd,
                    'n_obs': len(ret),
                    'validation': {
                        'return': {'r2': 0, 'rmse': 0, 'mae': 0, 'ic': 0},
                        'volatility': {'r2': 0, 'rmse': 0, 'mae': 0, 'ic': 0},
                        'maxdd': {'r2': 0, 'rmse': 0, 'mae': 0, 'ic': 0}
                    }
                }
                vols[sym] = ann_vol
                
            vols_series = pd.Series(vols)
            corr_matrix = returns_df.corr().fillna(0.0)
            
        else:
            ols = HorizonShiftedOLS(req.target_horizon, req.short_window, req.long_window)
            for sym in prices_df.columns:
                prices = prices_df[sym].dropna()
                
                ret_est = ols.fit_predict(prices, metric_type='return', split_ratio=req.split_ratio)
                vol_est = ols.fit_predict(prices, metric_type='volatility', split_ratio=req.split_ratio)
                mdd_est = ols.fit_predict(prices, metric_type='maxdd', split_ratio=req.split_ratio)
                
                # Annualize estimates
                ann_ret = ret_est['estimate'] * ret_annualizer
                ann_vol = vol_est['estimate'] * vol_annualizer
                
                results[sym] = {
                    'expected_return': ann_ret,
                    'expected_volatility': ann_vol,
                    'expected_maxdd': mdd_est['estimate'],
                    'n_obs': len(prices),
                    'validation': {
                        'return': ret_est,
                        'volatility': vol_est,
                        'maxdd': mdd_est
                    }
                }
                vols[sym] = ann_vol
                
            vols_series = pd.Series(vols)
            
            # Method B: Correlation (Shrinkage)
            blender = CorrelationBlender(req.short_window, req.long_window, req.correlation_alpha)
            corr_matrix = blender.estimate(prices_df)
        
        # Build Covariance
        cov_matrix = build_covariance(vols_series, corr_matrix)
        
        return {
            'status': 'success',
            'assets': results,
            'correlation': corr_matrix.to_dict(),
            'covariance': cov_matrix.to_dict()
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize")
def optimize_portfolio(req: OptimizeRequest):
    try:
        mu = pd.Series(req.mu)
        cov = pd.DataFrame(req.sigma)
        
        # Align indices
        cov = cov.reindex(index=mu.index, columns=mu.index).fillna(0.0)
        
        bounds = (req.min_weight, req.max_weight) if req.long_only else (None, None)
        
        optimizer = PortfolioOptimizer(
            expected_returns=mu, 
            covariance_matrix=cov, 
            risk_free_rate=req.risk_free_rate, 
            bounds=bounds
        )
        
        # 1. Optimize for relative weights
        if req.method == 'max_sharpe':
            rel_weights = optimizer.max_sharpe()
        elif req.method == 'min_variance':
            rel_weights = optimizer.min_variance()
        elif req.method == 'risk_parity':
            rel_weights = optimizer.risk_parity()
        elif req.method == 'herc':
            rel_weights = optimizer.hierarchical_optimization(model='HERC')
        elif req.method == 'hrp':
            rel_weights = optimizer.hierarchical_optimization(model='HRP')
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method {req.method}")
            
        # 2. Volatility targeting for final weights
        targeter = VolatilityTargeter(req.target_volatility)
        final_weights, pre_scaled_vol = targeter.scale_weights(rel_weights, cov)
        
        # Risk Contributions (based on relative weights for reporting)
        p_var = np.dot(rel_weights.values.T, np.dot(cov.values, rel_weights.values))
        p_vol = np.sqrt(p_var) if p_var > 0 else 1.0
        mrc = np.dot(cov.values, rel_weights.values) / p_vol
        rc = (rel_weights.values * mrc) / p_vol # percentage risk contribution
        risk_contributions = pd.Series(rc, index=mu.index).to_dict()
        
        # 3. Generate Frontier & Random Cloud
        frontier = optimizer.generate_efficient_frontier(50)
        cloud = optimizer.generate_random_portfolios(2000)
        
        return {
            'status': 'success',
            'relative_weights': rel_weights.to_dict(),
            'final_weights': final_weights.to_dict(),
            'risk_contributions': risk_contributions,
            'metrics': {
                'pre_scaled_volatility': pre_scaled_vol,
                'target_volatility': req.target_volatility,
                'portfolio_return_relative': optimizer._portfolio_return(rel_weights.values),
                'portfolio_return_final': optimizer._portfolio_return(final_weights.values)
            },
            'efficient_frontier': frontier,
            'random_cloud': cloud
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
