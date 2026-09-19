import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, Any

class HorizonShiftedOLS:
    def __init__(self, target_horizon: int, short_window: int, long_window: int):
        self.h = target_horizon
        self.w_s = short_window
        self.w_l = long_window

    def _compute_metric(self, prices: pd.Series, metric_type: str, window: int) -> pd.Series:
        """Computes the trailing metric over a specified window."""
        if metric_type == 'return':
            # Log return over the window
            return np.log(prices / prices.shift(window))
        elif metric_type == 'volatility':
            # Rolling std of 1-period log returns over the window
            ret = np.log(prices / prices.shift(1))
            return ret.rolling(window=window, min_periods=max(2, window//2)).std()
        elif metric_type == 'maxdd':
            # Max drawdown over the window
            roll_max = prices.rolling(window=window, min_periods=1).max()
            dd = (prices - roll_max) / roll_max
            return dd.rolling(window=window, min_periods=1).min()
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

    def fit_predict(self, prices: pd.Series, metric_type: str = 'return', split_ratio: float = 0.6) -> Dict[str, Any]:
        """
        Runs walk-forward OLS with an embargo to predict the future metric.
        Returns the final estimate for the current time and validation metrics.
        """
        # 1. Compute features (X1, X2) and target (y)
        x1 = self._compute_metric(prices, metric_type, self.w_s)
        x2 = self._compute_metric(prices, metric_type, self.w_l)
        
        # The target y is the metric computed over the horizon h, but shifted backwards by h.
        # So y_t is the realized metric from t to t+h.
        y_realized = self._compute_metric(prices, metric_type, self.h)
        y = y_realized.shift(-self.h)

        # Combine into a DataFrame to easily drop NaNs
        df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y}).dropna()
        
        if len(df) < 30:
            # Not enough data to train meaningfully
            return self._fallback_result(x1.iloc[-1] if not x1.dropna().empty else 0.0)

        # 2. Walk-forward validation with embargo
        T = len(df)
        train_end_idx = int(T * split_ratio)
        
        # We need to evaluate OOS. We'll do a simple expanding window walk-forward.
        # To avoid data snooping, we apply an embargo of h periods between train and test.
        # We'll just train on [0:train_end_idx] and test on [train_end_idx + h : ] for a single split validation for simplicity in V1,
        # but to get a full OOS series we can step through.
        # For performance, we'll do a single split with embargo for validation metrics.
        
        train_df = df.iloc[:train_end_idx]
        test_df = df.iloc[train_end_idx + self.h:]
        
        if len(test_df) < 10 or len(train_df) < 10:
            return self._fallback_result(x1.iloc[-1])

        # Train on initial training set
        model = LinearRegression()
        model.fit(train_df[['x1', 'x2']], train_df['y'])
        
        # Predict OOS
        y_pred = model.predict(test_df[['x1', 'x2']])
        y_true = test_df['y'].values
        
        # Metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Information Coefficient (IC) - Correlation between prediction and target
        # Handle cases where predictions might be constant (std=0)
        if np.std(y_pred) == 0 or np.std(y_true) == 0:
            ic = 0.0
        else:
            ic = np.corrcoef(y_pred, y_true)[0, 1]

        # 3. Final Estimation using all available historical data to fit the final model
        # Re-train on all data (excluding the last h periods which are NaNs in y)
        final_model = LinearRegression()
        final_model.fit(df[['x1', 'x2']], df['y'])
        
        # Predict the current expected value using the very last known x1 and x2
        current_x1 = x1.iloc[-1]
        current_x2 = x2.iloc[-1]
        
        if pd.isna(current_x1) or pd.isna(current_x2):
            estimate = 0.0
        else:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimate = final_model.predict([[current_x1, current_x2]])[0]
            
        return {
            'estimate': estimate,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'ic': ic,
            'beta1': final_model.coef_[0],
            'beta2': final_model.coef_[1],
            'alpha': final_model.intercept_
        }
        
    def _fallback_result(self, last_val: float) -> Dict[str, Any]:
        return {
            'estimate': last_val,
            'r2': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
            'ic': 0.0,
            'beta1': 1.0,
            'beta2': 0.0,
            'alpha': 0.0
        }


class CorrelationBlender:
    def __init__(self, short_window: int, long_window: int, alpha: float = 0.4):
        self.w_s = short_window
        self.w_l = long_window
        self.alpha = alpha
        
    def estimate(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the regime-blended shrinkage correlation matrix.
        prices_df: DataFrame where columns are symbols and rows are time/prices.
        """
        # Calculate daily log returns
        returns = np.log(prices_df / prices_df.shift(1)).dropna(how='all')
        
        if len(returns) < self.w_l:
            # Not enough data for long window, just use all available
            corr_all = returns.corr()
            return corr_all.fillna(0)
            
        # Short window correlation
        short_returns = returns.tail(self.w_s)
        c_short = short_returns.corr()
        
        # Long window correlation
        long_returns = returns.tail(self.w_l)
        c_long = long_returns.corr()
        
        # Blended
        c_blend = self.alpha * c_short + (1 - self.alpha) * c_long
        return c_blend.fillna(0)


def build_covariance(volatilities: pd.Series, correlation_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstructs the full Covariance matrix (Sigma) from a vector of volatilities (D)
    and a correlation matrix (C).
    Sigma = D * C * D
    """
    # Ensure volatilities index aligns with correlation matrix columns
    symbols = correlation_matrix.columns
    
    # Create diagonal matrix D
    vols = volatilities.reindex(symbols).fillna(0.0).values
    D = np.diag(vols)
    
    # Sigma = D * C * D
    C = correlation_matrix.values
    Sigma = D @ C @ D
    
    return pd.DataFrame(Sigma, index=symbols, columns=symbols)
