import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Dict, List

class PortfolioOptimizer:
    def __init__(self, expected_returns: pd.Series, covariance_matrix: pd.DataFrame, risk_free_rate: float = 0.0, bounds: Tuple[float, float] = (0.0, 1.0)):
        """
        Initializes the optimizer.
        :param expected_returns: Expected returns vector (annualized or per-period).
        :param covariance_matrix: Covariance matrix corresponding to the returns.
        :param risk_free_rate: Risk-free rate (same units as expected returns).
        :param bounds: (min_weight, max_weight) for each asset. Defaults to long-only (0, 1).
        """
        self.mu = expected_returns.values
        self.cov = covariance_matrix.values
        self.rf = risk_free_rate
        self.assets = expected_returns.index
        self.n_assets = len(self.assets)
        self.bounds = tuple(bounds for _ in range(self.n_assets))
        
        # We always want sum of weights to be 1.0 for the relative allocation
        self.constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        
    def _portfolio_return(self, w: np.ndarray) -> float:
        return np.sum(self.mu * w)
        
    def _portfolio_volatility(self, w: np.ndarray) -> float:
        return np.sqrt(np.dot(w.T, np.dot(self.cov, w)))

    def max_sharpe(self) -> pd.Series:
        """Maximizes the Sharpe ratio."""
        def negative_sharpe(w: np.ndarray) -> float:
            p_ret = self._portfolio_return(w)
            p_vol = self._portfolio_volatility(w)
            if p_vol == 0:
                return 0
            return -(p_ret - self.rf) / p_vol

        # Initial guess: equal weight
        init_guess = np.array([1.0 / self.n_assets] * self.n_assets)
        
        result = minimize(
            negative_sharpe, 
            init_guess, 
            method='SLSQP', 
            bounds=self.bounds, 
            constraints=self.constraints,
            options={'disp': False, 'ftol': 1e-9}
        )
        
        if not result.success:
            # Fallback to equal weight if optimization fails
            return pd.Series(init_guess, index=self.assets)
            
        return pd.Series(result.x, index=self.assets)

    def min_variance(self) -> pd.Series:
        """Minimizes portfolio variance."""
        def portfolio_variance(w: np.ndarray) -> float:
            return np.dot(w.T, np.dot(self.cov, w))

        init_guess = np.array([1.0 / self.n_assets] * self.n_assets)
        
        result = minimize(
            portfolio_variance, 
            init_guess, 
            method='SLSQP', 
            bounds=self.bounds, 
            constraints=self.constraints,
            options={'disp': False, 'ftol': 1e-9}
        )
        
        if not result.success:
            return pd.Series(init_guess, index=self.assets)
            
        return pd.Series(result.x, index=self.assets)

    def risk_parity(self) -> pd.Series:
        """Equal Risk Contribution (ERC) portfolio."""
        def risk_contribution_error(w: np.ndarray) -> float:
            p_var = np.dot(w.T, np.dot(self.cov, w))
            if p_var == 0:
                return 0
            
            # Marginal Risk Contribution
            mrc = np.dot(self.cov, w) / np.sqrt(p_var)
            
            # Risk Contribution of each asset
            rc = w * mrc
            
            # We want each asset to contribute equally: rc_i = 1/N * p_var / sqrt(p_var) = ...
            # Actually, standard way is that RC of all assets should be equal
            # We can minimize the sum of squared differences between RC of each pair, or var(RC)
            target_rc = p_var / self.n_assets
            return np.sum(np.square(rc - np.sqrt(p_var) * target_rc)) # actually rc sum to p_vol
            
        # Standard formulation for Risk Parity objective
        def risk_budget_objective(w: np.ndarray) -> float:
            p_var = np.dot(w.T, np.dot(self.cov, w))
            if p_var == 0: return 0
            p_vol = np.sqrt(p_var)
            
            # Risk contributions
            rc = w * np.dot(self.cov, w) / p_vol
            
            # Minimize squared difference from target risk budget (1/N)
            # rc_target = p_vol / N
            return np.sum(np.square(rc - (p_vol / self.n_assets)))

        init_guess = np.array([1.0 / self.n_assets] * self.n_assets)
        
        result = minimize(
            risk_budget_objective, 
            init_guess, 
            method='SLSQP', 
            bounds=self.bounds, 
            constraints=self.constraints,
            options={'disp': False, 'ftol': 1e-9}
        )
        
        if not result.success:
            return pd.Series(init_guess, index=self.assets)
            
        return pd.Series(result.x, index=self.assets)

    def generate_efficient_frontier(self, num_points: int = 50) -> Dict[str, List[float]]:
        """
        Generates the true efficient frontier by minimizing variance subject to a target return.
        """
        # 1. Find the minimum and maximum return possible given constraints
        min_var_w = self.min_variance().values
        min_ret = self._portfolio_return(min_var_w)
        
        # Max return is typically putting max weight on the highest return asset
        # We can just optimize for max return directly
        def negative_return(w): return -self._portfolio_return(w)
        max_ret_res = minimize(negative_return, min_var_w, method='SLSQP', bounds=self.bounds, constraints=self.constraints)
        if max_ret_res.success:
            max_ret = self._portfolio_return(max_ret_res.x)
        else:
            max_ret = np.max(self.mu) # fallback
            
        if max_ret <= min_ret:
            # Degenerate case
            return {'returns': [min_ret], 'volatilities': [self._portfolio_volatility(min_var_w)], 'weights': [min_var_w.tolist()]}

        # 2. Grid search over target returns
        target_returns = np.linspace(min_ret, max_ret, num_points)
        frontier_vols = []
        frontier_returns = []
        frontier_weights = []
        
        for tr in target_returns:
            # We want to minimize variance subject to weights sum = 1 AND portfolio return >= target_return
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

    def generate_random_portfolios(self, num_portfolios: int = 5000) -> Dict[str, List[float]]:
        """
        Generates a cloud of random portfolios for background visualization.
        Only strictly respects long-only (0,1) constraints via Dirichlet distribution.
        """
        returns = []
        vols = []
        
        # Using Dirichlet distribution ensures weights sum to 1 and are >= 0
        random_weights = np.random.dirichlet(np.ones(self.n_assets), num_portfolios)
        
        for w in random_weights:
            # If there are custom bounds (e.g. max weight), we'd need to filter or reject,
            # but for visual cloud, pure Dirichlet is usually fine enough to show the space.
            returns.append(self._portfolio_return(w))
            vols.append(self._portfolio_volatility(w))
            
        return {
            'returns': returns,
            'volatilities': vols
        }


class VolatilityTargeter:
    def __init__(self, target_volatility: float):
        """
        :param target_volatility: The absolute target volatility (e.g., 0.15 for 15%).
        """
        self.target_vol = target_volatility

    def scale_weights(self, relative_weights: pd.Series, covariance_matrix: pd.DataFrame) -> Tuple[pd.Series, float]:
        """
        Scales the relative weights so that the portfolio achieves the target volatility.
        :return: (scaled_weights, expected_portfolio_volatility_before_scaling)
        """
        w = relative_weights.values
        cov = covariance_matrix.reindex(index=relative_weights.index, columns=relative_weights.index).values
        
        # Calculate current portfolio volatility
        p_var = np.dot(w.T, np.dot(cov, w))
        p_vol = np.sqrt(p_var)
        
        if p_vol <= 0:
            return relative_weights * 0.0, 0.0
            
        # Scale factor
        k = self.target_vol / p_vol
        
        # Scaled weights
        scaled_w = w * k
        
        return pd.Series(scaled_w, index=relative_weights.index), p_vol
