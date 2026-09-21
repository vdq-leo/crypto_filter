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
        
        # Base returns and historical max drawdown (used in both paths)
        returns_df = prices_df.pct_change().dropna(how='all')
        
        import riskfolio as rp
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Fill NaNs with 0 (no return) to satisfy Riskfolio's JS computation requirements
            js_means = rp.mean_vector(returns_df.fillna(0.0), method='JS').iloc[0].astype(float)
            
        cum_returns = (1 + returns_df).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = cum_returns / rolling_max - 1
        maxdd_dict = drawdowns.min().to_dict()
        
        if req.estimation_method == 'historical':
            for sym in prices_df.columns:
                ret = returns_df[sym].dropna()
                ann_ret = js_means[sym] * ret_annualizer if not ret.empty else 0.0
                ann_vol = ret.std() * vol_annualizer if not ret.empty else 0.0
                mdd = maxdd_dict.get(sym, 0.0)
                
                results[sym] = {
                    'expected_return': ann_ret,
                    'expected_volatility': ann_vol,
                    'expected_maxdd': mdd,
                    'n_obs': len(ret),
                    'latest_price': float(prices_df[sym].dropna().iloc[-1]) if not prices_df[sym].dropna().empty else 1.0,
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
                # Predict volatility using OLS
                vol_est = ols.fit_predict(prices, metric_type='volatility', split_ratio=req.split_ratio)
                
                # Annualize estimates: Use James-Stein shrunken mean for returns
                ret = returns_df[sym].dropna()
                ann_ret = js_means[sym] * ppy if not ret.empty else 0.0
                ann_vol = vol_est['estimate'] * vol_annualizer
                # Use historical maxdd instead of forecasting it (structurally mismatched)
                ann_mdd = maxdd_dict.get(sym, 0.0)
                
                results[sym] = {
                    'expected_return': ann_ret,
                    'expected_volatility': ann_vol,
                    'expected_maxdd': ann_mdd,
                    'n_obs': len(prices),
                    'latest_price': float(prices.iloc[-1]) if not prices.empty else 1.0,
                    'confidence': {
                        'return': 0.0,
                        'volatility': vol_est.get('confidence', 0.0),
                    },
                    'validation': {
                        'return': {},
                        'volatility': vol_est.get('val_metrics', {}),
                        'maxdd': {}
                    }
                }
                vols[sym] = ann_vol
                
            vols_series = pd.Series(vols)
            
            # Method B: Correlation (Shrinkage toward identity with λ=0.1)
            blender = CorrelationBlender(req.short_window, req.long_window,
                                         req.correlation_alpha, shrinkage=0.1)
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
            bounds=bounds,
            periods_per_year=1.0  # Inputs are already annualized
        )
        
        # 1. Optimize for relative weights
        if req.method == 'max_sharpe':
            res = optimizer.max_sharpe()
        elif req.method == 'min_variance':
            res = optimizer.min_variance()
        elif req.method == 'risk_parity':
            res = optimizer.risk_parity()
        elif req.method == 'herc':
            res = optimizer.hierarchical_optimization(model='HERC')
        elif req.method == 'hrp':
            res = optimizer.hierarchical_optimization(model='HRP')
        elif req.method == 'nco':
            res = optimizer.hierarchical_optimization(model='NCO')
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method {req.method}")
            
        if not res.get('success', False):
            raise HTTPException(status_code=400, detail=res.get('message', 'Optimization failed'))
            
        rel_weights = res['weights']
            
        # 2. Volatility targeting for final weights
        targeter = VolatilityTargeter(req.target_volatility)
        
        # Pass the annualized covariance matrix explicitly to the targeter
        ann_cov_df = pd.DataFrame(optimizer.cov, index=mu.index, columns=mu.index)
        target_res = targeter.scale_weights(rel_weights, ann_cov_df)
        final_weights = target_res['risky_weights']
        
        # Risk Contributions (based on relative weights for reporting)
        p_var = np.dot(rel_weights.values.T, np.dot(optimizer.cov, rel_weights.values))
        p_vol = np.sqrt(max(p_var, 0.0))
        if p_vol > 1e-8:
            mrc = np.dot(optimizer.cov, rel_weights.values) / p_vol
            rc = (rel_weights.values * mrc) / p_vol # percentage risk contribution
        else:
            rc = np.zeros_like(rel_weights.values)
        risk_contributions = pd.Series(rc, index=mu.index).to_dict()
        
        # 3. Generate Frontier & Random Cloud
        frontier = optimizer.generate_efficient_frontier(50)
        cloud = optimizer.generate_random_portfolios(2000)
        
        return {
            'status': 'success',
            'relative_weights': rel_weights.to_dict(),
            'final_weights': final_weights.to_dict(),
            'cash_weight': target_res['cash_weight'],
            'gross_leverage': target_res['gross_leverage'],
            'risk_contributions': risk_contributions,
            'metrics': {
                'pre_scaled_volatility': target_res['expected_volatility'],
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
