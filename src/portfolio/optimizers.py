"""
Portfolio Optimizers v2
=======================

Features:
1. Explicit horizon alignment (periods_per_year).
2. VolatilityTargeter returns {risky_weights, cash_weight, gross_leverage}.
3. Rejection sampling for random portfolios (option A).
4. Explicit result dict format: {weights: Series, success: bool, message: str, ...}.
5. Clean hierarchical integration with custom_cov.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Dict, List, Any, Optional
import riskfolio as rp
import warnings

# Temporarily patch Riskfolio-Lib 7.3.0 HERC linkage bug
def _apply_riskfolio_patch():
    if hasattr(rp.HCPortfolio, '_hierarchical_recursive_bisection'):
        original_hrb = getattr(rp.HCPortfolio, '_hierarchical_recursive_bisection')
        # Check if already patched to avoid recursion
        if not getattr(original_hrb, '_is_patched', False):
            def patched_hrb(self, Z, rm="MV", rf=0, linkage=None, model="HERC", upper_bound=None, lower_bound=None):
                return original_hrb(self, Z, rm=rm, rf=rf, model=model)
            patched_hrb._is_patched = True
            rp.HCPortfolio._hierarchical_recursive_bisection = patched_hrb

_apply_riskfolio_patch()

class PortfolioOptimizer:
    def __init__(self, expected_returns: pd.Series, covariance_matrix: pd.DataFrame, 
                 risk_free_rate: float = 0.0, bounds: Tuple[float, float] = (0.0, 1.0),
                 periods_per_year: float = 365.0):
        """
        Initializes the optimizer.
        :param expected_returns: Expected returns vector (per-period).
        :param covariance_matrix: Covariance matrix (per-period).
        :param risk_free_rate: Risk-free rate (annualized).
        :param bounds: (min_weight, max_weight) relative allocation constraints.
        :param periods_per_year: Scaling factor to convert per-period inputs to annualized terms.
        """
        self.periods = periods_per_year
        # Annualize inputs internally to align with annualized rf and targets
        self.mu = expected_returns.values * self.periods
        self.cov = covariance_matrix.values * self.periods
        self.rf = risk_free_rate
        
        self.assets = expected_returns.index
        self.n_assets = len(self.assets)
        self.bounds = tuple(bounds for _ in range(self.n_assets))
        
        # We always want sum of relative weights to be 1.0
        self.constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        
    def _portfolio_return(self, w: np.ndarray) -> float:
        return np.sum(self.mu * w)
        
    def _portfolio_volatility(self, w: np.ndarray) -> float:
        var = np.dot(w.T, np.dot(self.cov, w))
        return np.sqrt(max(var, 0.0))
        
    def _format_result(self, w: np.ndarray, success: bool, message: str) -> Dict[str, Any]:
        """Format the output dictionary, running validations if successful."""
        res = {
            'weights': pd.Series(w, index=self.assets),
            'success': success,
            'message': message,
        }
        if success:
            res.update(self.validate(w))
        return res

    def validate(self, w: np.ndarray) -> Dict[str, Any]:
        """Post-optimization validation of weights."""
        sum_w = np.sum(w)
        min_w = np.min(w)
        max_w = np.max(w)
        p_var = np.dot(w.T, np.dot(self.cov, w))
        
        # Check bound violations
        lower_bound = self.bounds[0][0]
        upper_bound = self.bounds[0][1]
        bound_violation = max(0, lower_bound - min_w, max_w - upper_bound)
        
        return {
            'sum_weights': float(sum_w),
            'min_weight': float(min_w),
            'max_weight': float(max_w),
            'portfolio_vol': float(np.sqrt(max(p_var, 0.0))),
            'max_bound_violation': float(bound_violation)
        }

    def max_sharpe(self) -> Dict[str, Any]:
        """Maximizes the Sharpe ratio."""
        # Find valid assets (exclude zero-volatility to prevent undefined Sharpe)
        vols = np.sqrt(np.diag(self.cov))
        valid_idx = vols > 1e-8
        
        if not np.any(valid_idx):
            return self._format_result(np.ones(self.n_assets) / self.n_assets, False, "No valid assets (all zero-volatility)")

        def negative_sharpe(w: np.ndarray) -> float:
            p_ret = self._portfolio_return(w)
            p_vol = self._portfolio_volatility(w)
            if p_vol <= 1e-8:
                return 0
            return -(p_ret - self.rf) / p_vol

        init_guess = np.ones(self.n_assets) / self.n_assets
        
        # If some assets are invalid, we could optimize only over valid ones, but simpler to just 
        # force their weight to 0 via bounds temporarily.
        active_bounds = [self.bounds[i] if valid_idx[i] else (0.0, 0.0) for i in range(self.n_assets)]

        res = minimize(
            negative_sharpe, 
            init_guess, 
            method='SLSQP', 
            bounds=active_bounds, 
            constraints=self.constraints,
            options={'disp': False, 'ftol': 1e-9}
        )
        
        if not res.success:
            return self._format_result(init_guess, False, res.message)
            
        return self._format_result(res.x, True, "Optimization successful")

    def min_variance(self) -> Dict[str, Any]:
        """Minimizes portfolio variance."""
        init_guess = np.ones(self.n_assets) / self.n_assets
        
        res = minimize(
            lambda w: np.dot(w.T, np.dot(self.cov, w)), 
            init_guess, 
            method='SLSQP', 
            bounds=self.bounds, 
            constraints=self.constraints,
            options={'disp': False, 'ftol': 1e-9}
        )
        
        if not res.success:
            return self._format_result(init_guess, False, res.message)
            
        return self._format_result(res.x, True, "Optimization successful")

    def risk_parity(self) -> Dict[str, Any]:
        """Equal Risk Contribution (ERC) portfolio."""
        def risk_budget_objective(w: np.ndarray) -> float:
            p_var = np.dot(w.T, np.dot(self.cov, w))
            if p_var <= 1e-8: return 0.0
            p_vol = np.sqrt(p_var)
            
            # Risk contributions
            rc = w * np.dot(self.cov, w) / p_vol
            
            # Dimensionless relative error: sum((RC_i / p_vol) - (1/N))^2
            rel_rc = rc / p_vol
            return np.sum(np.square(rel_rc - (1.0 / self.n_assets)))

        init_guess = np.ones(self.n_assets) / self.n_assets
        
        res = minimize(
            risk_budget_objective, 
            init_guess, 
            method='SLSQP', 
            bounds=self.bounds, 
            constraints=self.constraints,
            options={'disp': False, 'ftol': 1e-9}
        )
        
        if not res.success:
            return self._format_result(init_guess, False, res.message)
            
        return self._format_result(res.x, True, "Optimization successful")

    def hierarchical_optimization(self, model: str = 'HERC') -> Dict[str, Any]:
        """
        Uses Riskfolio-Lib to compute Hierarchical Equal Risk Contribution (HERC) or HRP.
        Validates covariance and utilizes custom_cov parameter securely.
        """
        # Ensure no zero-volatility assets before clustering
        vols = np.sqrt(np.diag(self.cov))
        if np.any(vols <= 1e-8):
            return self._format_result(
                np.ones(self.n_assets) / self.n_assets, 
                False, 
                "Hierarchical methods require strictly positive volatility for all assets."
            )
            
        try:
            # We must pass returns to initialize HCPortfolio due to shape checks in riskfolio
            mock_returns = pd.DataFrame(np.random.randn(500, self.n_assets), columns=self.assets)
            port = rp.HCPortfolio(returns=mock_returns)
            
            if model == 'NCO' and self.bounds:
                lower = pd.Series([b[0] if b[0] is not None else 0.0 for b in self.bounds], index=self.assets)
                upper = pd.Series([b[1] if b[1] is not None else 1.0 for b in self.bounds], index=self.assets)
                port.w_min = lower
                port.w_max = upper
            
            cov_df = pd.DataFrame(self.cov, index=self.assets, columns=self.assets)
            # Use custom_cov for method_cov/codependence
            w = port.optimization(
                model=model, 
                rm='MV', 
                rf=self.rf, 
                method_cov='custom_cov', 
                custom_cov=cov_df
            )
            
            if w is None or w.empty:
                return self._format_result(np.ones(self.n_assets) / self.n_assets, False, "Riskfolio returned empty weights")
                
            return self._format_result(w['weights'].values, True, "Optimization successful")
            
        except Exception as e:
            return self._format_result(np.ones(self.n_assets) / self.n_assets, False, f"Riskfolio exception: {str(e)}")

    def generate_efficient_frontier(self, num_points: int = 50) -> Dict[str, Any]:
        """
        Generates the constrained efficient frontier by minimizing variance subject to target returns.
        """
        # 1. Minimum return (min variance portfolio)
        min_var_res = self.min_variance()
        min_var_w = min_var_res['weights'].values if min_var_res['success'] else np.ones(self.n_assets)/self.n_assets
        min_ret = self._portfolio_return(min_var_w)
        
        # 2. Maximum return (allocate max bound to highest mu assets)
        max_ret_res = minimize(
            lambda w: -self._portfolio_return(w), 
            min_var_w, 
            method='SLSQP', 
            bounds=self.bounds, 
            constraints=self.constraints
        )
        max_ret = self._portfolio_return(max_ret_res.x) if max_ret_res.success else np.max(self.mu)
            
        if max_ret <= min_ret + 1e-6:
            return {'returns': [min_ret], 'volatilities': [self._portfolio_volatility(min_var_w)], 'weights': [min_var_w.tolist()]}

        target_returns = np.linspace(min_ret, max_ret, num_points)
        frontier_vols = []
        frontier_returns = []
        frontier_weights = []
        
        for tr in target_returns:
            c = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                {'type': 'ineq', 'fun': lambda x: self._portfolio_return(x) - tr}
            ]
            
            res = minimize(
                lambda w: np.dot(w.T, np.dot(self.cov, w)), 
                min_var_w, 
                method='SLSQP', 
                bounds=self.bounds, 
                constraints=c,
                options={'disp': False, 'ftol': 1e-7}
            )
            
            if res.success:
                frontier_vols.append(self._portfolio_volatility(res.x))
                frontier_returns.append(self._portfolio_return(res.x))
                frontier_weights.append(res.x.tolist())
                
        return {
            'returns': frontier_returns,
            'volatilities': frontier_vols,
            'weights': frontier_weights
        }

    def generate_random_portfolios(self, num_portfolios: int = 5000, max_attempts: int = 50000) -> Dict[str, List[float]]:
        """
        Generates random feasible portfolios via rejection sampling (Option A).
        Only accepts Dirichlet draws that strictly respect self.bounds.
        """
        returns = []
        vols = []
        
        attempts = 0
        accepted = 0
        min_b = np.array([b[0] for b in self.bounds])
        max_b = np.array([b[1] for b in self.bounds])
        
        while accepted < num_portfolios and attempts < max_attempts:
            w = np.random.dirichlet(np.ones(self.n_assets))
            attempts += 1
            
            # Rejection check against bounds
            if np.all(w >= min_b) and np.all(w <= max_b):
                returns.append(self._portfolio_return(w))
                vols.append(self._portfolio_volatility(w))
                accepted += 1
            
        return {
            'returns': returns,
            'volatilities': vols
        }


class VolatilityTargeter:
    def __init__(self, target_volatility_annual: float):
        """
        :param target_volatility_annual: The absolute target volatility (annualized), e.g., 0.15 for 15%.
        """
        self.target_vol = target_volatility_annual

    def scale_weights(self, relative_weights: pd.Series, covariance_matrix_annual: pd.DataFrame) -> Dict[str, Any]:
        """
        Scales relative weights to achieve target volatility.
        This changes the gross exposure of the portfolio.
        
        :return: Dict with:
          - risky_weights: The final scaled weights allocated to the assets
          - cash_weight: Unallocated capital (if scaled down) or negative (if leveraged)
          - gross_leverage: The total absolute weight of risky assets
          - expected_volatility: The portfolio volatility before scaling
        """
        w = relative_weights.values
        cov = covariance_matrix_annual.reindex(index=relative_weights.index, columns=relative_weights.index).values
        
        p_var = np.dot(w.T, np.dot(cov, w))
        p_vol = np.sqrt(max(p_var, 0.0))
        
        if p_vol <= 1e-8:
            return {
                'risky_weights': relative_weights * 0.0,
                'cash_weight': 1.0,
                'gross_leverage': 0.0,
                'expected_volatility': 0.0
            }
            
        # Scale factor
        k = self.target_vol / p_vol
        
        # Scaled weights
        scaled_w = w * k
        gross_leverage = np.sum(np.abs(scaled_w))
        cash_weight = 1.0 - np.sum(scaled_w) # Simple cash residual
        
        return {
            'risky_weights': pd.Series(scaled_w, index=relative_weights.index),
            'cash_weight': float(cash_weight),
            'gross_leverage': float(gross_leverage),
            'expected_volatility': float(p_vol)
        }
