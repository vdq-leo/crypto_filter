"""
Metrics Engine for Crypto Filter
Optimized calculation using Numpy.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import numba

@numba.njit
def _rma(x, n_period):
    a = np.full_like(x, np.nan)
    if len(x) < n_period: return a
    start_idx = -1
    for i in range(len(x)):
        if not np.isnan(x[i]):
            if start_idx == -1:
                start_idx = i
            if i - start_idx + 1 == n_period:
                # Calculate mean manually instead of np.nanmean to be safe with Numba
                sum_val = 0.0
                count = 0
                for k in range(start_idx, i+1):
                    if not np.isnan(x[k]):
                        sum_val += x[k]
                        count += 1
                a[i] = sum_val / count if count > 0 else np.nan
                alpha = 1.0 / n_period
                for j in range(i+1, len(x)):
                    a[j] = alpha * x[j] + (1 - alpha) * a[j-1]
                break
    return a

@numba.njit
def _get_pivots(src, left, right, is_high):
    n = len(src)
    pivots = np.full(n, np.nan)
    for i in range(left + right, n):
        idx = i - right
        is_pivot = True
        
        for j in range(idx - left, idx):
            if is_high and src[j] >= src[idx]:
                is_pivot = False; break
            if not is_high and src[j] <= src[idx]:
                is_pivot = False; break
        if not is_pivot: continue
        
        for j in range(idx + 1, i + 1):
            if is_high and src[j] >= src[idx]:
                is_pivot = False; break
            if not is_high and src[j] <= src[idx]:
                is_pivot = False; break
                
        if is_pivot:
            pivots[i] = src[idx]
    return pivots

@numba.njit
def _adaptive_ema_numba(x, alpha):
    a = np.full_like(x, np.nan)
    if len(x) == 0: return a
    start_idx = -1
    for i in range(len(x)):
        if not np.isnan(x[i]):
            start_idx = i
            a[i] = x[i]
            break
    if start_idx == -1: return a
    
    for i in range(start_idx + 1, len(x)):
        if np.isnan(x[i]):
            a[i] = a[i-1]
        else:
            curr_alpha = alpha[i]
            if np.isnan(curr_alpha): curr_alpha = 1.0
            a[i] = curr_alpha * x[i] + (1 - curr_alpha) * a[i-1]
    return a

class MetricsEngine:
    """Calculates metrics from price data using vectorized operations."""
    TICKER_24H = "/fapi/v1/ticker/24hr"
    
    def __init__(self):
        self._results_cache = {} # Key: (symbol, interval, last_timestamp, length)
    
    @staticmethod
    def get_annual_scaling(interval: str) -> float:
        """
        Returns number of periods per year for a given Binance interval.
        Assuming crypto markets 24/7 (365 days).
        """
        scales = {
            '1m': 525600,
            '3m': 175200,
            '5m': 105120,
            '15m': 35040,
            '30m': 17520,
            '1h': 8760,
            '2h': 4380,
            '4h': 2190,
            '6h': 1460,
            '8h': 1095,
            '12h': 730,
            '1d': 365,
            '3d': 121.66,
            '1w': 52.14,
            '1M': 12
        }
        return scales.get(interval, 8760) # Default to 1h if unknown

    @staticmethod
    def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate log returns: ln(P_t / P_{t-1})"""
        with np.errstate(divide='ignore', invalid='ignore'):
            returns = np.log(prices[1:] / prices[:-1])
        return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def calculate_price_zscore(prices: np.ndarray, window: int = 30) -> float:
        """
        Calculates the latest Z-score of price relative to its rolling mean/std.
        Z = (Price - Mean) / Std
        """
        if len(prices) < window: return 0.0
        
        subset = prices[-window:]
        mean = np.mean(subset)
        std = np.std(subset)
        
        if std < 1e-12: return 0.0
        return float((prices[-1] - mean) / std)

    @staticmethod
    def calculate_price_sma_diff(prices: np.ndarray, window: int = 20) -> float:
        """
        Calculates (Price - SMA) / Price.
        """
        if len(prices) < window: return 0.0
        
        sma = np.mean(prices[-window:])
        current_price = prices[-1]
        
        if abs(current_price) < 1e-12: return 0.0
        return float((current_price - sma) / current_price)

    # --- TECHNICAL INDICATOR HELPERS ---
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, n: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=n, min_periods=n).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=n, min_periods=n).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=n, min_periods=n).mean()

    @staticmethod
    def bollinger_bands(series: pd.Series, n: int = 20, k: float = 2.0) -> tuple:
        ma = series.ewm(span=n).mean()
        std = series.ewm(span=n).std()
        upper = ma + (k * std)
        lower = ma - (k * std)
        return upper, lower, ma

    @staticmethod
    def aroon(high: pd.Series, low: pd.Series, n: int = 25) -> tuple:
        aroon_up = high.rolling(n+1).apply(lambda x: float(np.argmax(x)) / n * 100, raw=True)
        aroon_down = low.rolling(n+1).apply(lambda x: float(np.argmin(x)) / n * 100, raw=True)
        return aroon_up, aroon_down

    @staticmethod
    def rsi_adapt(src: pd.Series, w_short: int = 40, w_long: int = 200, vol_factor: float = 1.0) -> pd.Series:
        ret = np.log(src / src.shift(1))
        vS = ret.rolling(w_short).std()
        vL = ret.rolling(w_long).std()
        a = np.clip(vS / (vL * vol_factor), 0.01, 0.99)
        return a * MetricsEngine.rsi(src, w_short) + (1 - a) * MetricsEngine.rsi(src, w_long)

    @staticmethod
    def vama_cma(src: pd.Series, w_short: int = 40, w_long: int = 200, vol_factor: float = 1.0) -> pd.Series:
        ret = np.log(src / src.shift(1))
        vS = ret.rolling(w_short).std()
        vL = ret.rolling(w_long).std()
        a = 2.0 / (np.maximum(1, np.floor(10.0 / np.minimum(vS / (vL * vol_factor), 1.0))) + 1.0)
        vama_vals = _adaptive_ema_numba(src.values, a.values)
        vama = pd.Series(vama_vals, index=src.index)
        proxy_atr = src.diff().abs().rolling(10).mean().replace(0, 1e-9)
        return (src - vama) / proxy_atr

    @staticmethod
    def ewmac(src: pd.Series, w_short: int = 40, w_long: int = 200) -> pd.Series:
        fast = src.ewm(span=w_short, adjust=False).mean()
        slow = src.ewm(span=w_long, adjust=False).mean()
        v = src.rolling(w_short).std().replace(0, 1e-9)
        return (fast - slow) / v

    @staticmethod
    def z_score_indicator(src: pd.Series, length: int = 100) -> pd.Series:
        return (src - src.rolling(length).mean()) / src.rolling(length).std().replace(0, 1e-9)

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        tp = (df['high'] + df['low'] + df['close']) / 3
        return (tp * df['volume']).cumsum() / df['volume'].cumsum()
    
    @staticmethod
    def vama(close: pd.Series, w_short: int = 40, w_slow: int = 120) -> pd.Series:
        rets = np.log(close / close.shift()).fillna(0)
        s_std = rets.ewm(span=w_short).std()
        l_std = rets.ewm(span=w_slow).std()
        alpha = ((l_std * 0.5) / s_std).clip(lower=0.05, upper=0.95).fillna(0.05)
        v_alpha = alpha.values
        v_close = close.values
        res = np.zeros_like(v_close)
        res[0] = v_close[0]
        for i in range(1, len(v_close)):
            res[i] = v_alpha[i] * v_close[i] + (1 - v_alpha[i]) * res[i-1]
        return pd.Series(res, index=close.index)

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, window: int = 40, benchmark_returns: pd.Series = None, benchmark_prices: pd.Series = None, interval: str = '1h', include_metrics: Optional[List[str]] = None, tail_only: bool = False) -> pd.DataFrame:
        """
        Calculates all 11 advanced metrics requested by the user.
        Returns a DataFrame with the same index as the input df.
        """
        w_slow = int(window * 3)
        w_short = int(window * 0.5)
        w_mid = window

        def should_calc(m):
            return include_metrics is None or m in include_metrics
            
        max_warmup = w_slow * 2
        if should_calc('rel_strength_z'):
            max_warmup = max(max_warmup, 120)
            
        if tail_only and len(df) > max_warmup:
            # Keep enough history to warm up EMAs and rolling windows
            df = df.tail(max_warmup).copy()

        res = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Core returns calculation
        ret = close.pct_change()

        # 1. EWVA: (EMA_fast - EMA_slow) / STD(price)
        if should_calc('ewva') or should_calc('vol_rank') or should_calc('vwap_z'):
            std_p = close.ewm(span=window).std()
            
        if should_calc('ewva'):
            ema_f = MetricsEngine.ema(close, w_short)
            ema_s = MetricsEngine.ema(close, w_slow)
            res['ewva'] = (ema_f - ema_s) / std_p
        
        # 2. Aroon Oscillator: (AroonUp - AroonDown) / 100
        if should_calc('aroon_osc'):
            au, ad = MetricsEngine.aroon(high, low, w_mid)
            res['aroon_osc'] = (au - ad) / 100
        
        # 4. RSI (Normalized): RSI / 100
        if should_calc('rsi_norm'):
            res['rsi_norm'] = (MetricsEngine.rsi(close, window) - 50) / 100
                
        # 6. Normalized ATR: ATR / Close
        if should_calc('atr_norm') or should_calc('vol_atr'):
            atr_s = MetricsEngine.atr(high, low, close, w_short)
            atr_norm = atr_s / close.replace(0, 1e-9)
            if should_calc('atr_norm'):
                res['atr_norm'] = atr_norm
        
        # 7. Chaikin Money Flow (CMF)
        if should_calc('cmf'):
            mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1e-9)
            mfv = mfm * volume
            res['cmf'] = mfv.ewm(span=window).sum() / volume.ewm(span=window).sum()
        
        # 8. VWAP Deviation Z-Score: (Close - VWAP) / STD(price)
        if should_calc('vwap_z'):
            vwp = MetricsEngine.vwap(df)
            res['vwap_z'] = (close - vwp) / std_p
        
        # 9. Relative Strength Z-Score (vs benchmark) via Relative Momentum Composite
        if should_calc('rel_strength_z'):
            if benchmark_prices is not None:
                common_idx = df.index.intersection(benchmark_prices.index)
                if len(common_idx) >= 100:
                    asset_p = df.loc[common_idx, 'close']
                    btc_p = benchmark_prices.loc[common_idx]
                    
                    # Params (Reduced to ~1/3 to require less historical data)
                    p_short = 13
                    p_long = 67
                    beta_len = 33
                    smooth = 3
                    volFactor = 1.0
                    
                    # Asset Components
                    a_rsi = MetricsEngine.z_score_indicator(MetricsEngine.rsi_adapt(asset_p, p_short, p_long, volFactor), beta_len)
                    a_cma = MetricsEngine.z_score_indicator(MetricsEngine.vama_cma(asset_p, p_short, p_long, volFactor), beta_len)
                    a_ewmac = MetricsEngine.z_score_indicator(MetricsEngine.ewmac(asset_p, p_short, p_long), beta_len)
                    sym2_idx = (a_rsi + a_cma + a_ewmac) / 3.0
                    
                    # Benchmark Components
                    b_rsi = MetricsEngine.z_score_indicator(MetricsEngine.rsi_adapt(btc_p, p_short, p_long, volFactor), beta_len)
                    b_cma = MetricsEngine.z_score_indicator(MetricsEngine.vama_cma(btc_p, p_short, p_long, volFactor), beta_len)
                    b_ewmac = MetricsEngine.z_score_indicator(MetricsEngine.ewmac(btc_p, p_short, p_long), beta_len)
                    sym1_idx = (b_rsi + b_cma + b_ewmac) / 3.0
                    
                    # Spread (BTC - Asset)
                    spread = sym1_idx - sym2_idx
                    diff = spread.diff()
                    
                    sVol = diff.rolling(p_short).std().clip(lower=1e-6)
                    lVol = diff.rolling(p_long).std().clip(lower=1e-6)
                    
                    alpha = ((1.0 / smooth) / ((lVol * volFactor) / sVol)).clip(lower=0.05, upper=0.95)
                    alpha = alpha.fillna(0.05)
                    
                    spread_s_vals = _adaptive_ema_numba(spread.values, alpha.values)
                    res['rel_strength_z'] = pd.Series(spread_s_vals, index=common_idx).reindex(df.index).fillna(0.0)
                else:
                    res['rel_strength_z'] = np.nan
            else:
                res['rel_strength_z'] = np.nan
            
        # 10. Volatility-Adjusted Momentum (VAM): ROC / (ATR / Vola)
        if should_calc('vam'):
            roc = close.pct_change(w_short)
            res['vam'] = roc / ret.ewm(span=window).std()
        
        # 11. Return Skewness (Rolling)
        if should_calc('skewness'):
            res['skewness'] = ret.rolling(window).skew()
        
        # 12. Lagged Signs
        if should_calc('sign_lag1'): res['sign_lag1'] = np.sign(ret.shift(1))
        if should_calc('sign_lag2'): res['sign_lag2'] = np.sign(ret.shift(2))
        if should_calc('sign_lag3'): res['sign_lag3'] = np.sign(ret.shift(3))

        # 13. Rolling lagged sign
        if should_calc('rolling_sign_lag5'): res['rolling_sign_lag5'] = np.sign(ret.rolling(5).mean())
        if should_calc('rolling_sign_lag10'): res['rolling_sign_lag10'] = np.sign(ret.rolling(10).mean())
        if should_calc('rolling_sign_lag20'): res['rolling_sign_lag20'] = np.sign(ret.rolling(20).mean())

        # 14. Autocorrelation
        if should_calc('autocorr_1'):
            res['autocorr_1'] = ret.rolling(5).corr(ret.shift(1))
        if should_calc('autocorr_5'):
            res['autocorr_5'] = ret.rolling(20).corr(ret.shift(5))
        
        # 15. Imbalance Bar
        if should_calc('imbalance_bar'):
            body = (close - df['open']).abs()
            rng = (high - low).replace(0, 1e-9)
            res['imbalance_bar'] = (body / rng) * np.sign(close - df['open'])
        
        # 16. VAMA (Volatility Adjusted Moving Average)
        if should_calc('vama'):
            res['vama'] = MetricsEngine.vama(close, w_short, w_slow) / close
        
        # --- Standard/Legacy Metrics for Snapshot/Table use ---
        if should_calc('volatility') or should_calc('vol_atr'):
            ann_factor = MetricsEngine.get_annual_scaling(interval)
            vol_series = ret.ewm(span=window).std() * np.sqrt(ann_factor)
            if should_calc('volatility'): res['volatility'] = vol_series
            if should_calc('vol_atr'): res['vol_atr'] = vol_series / atr_norm

        if should_calc('vol_imbalance'):
            res['vol_imbalance'] = ret.ewm(span=w_short).std() / ret.ewm(span=w_slow).std()
        
        if should_calc('volume_imbalance'):
            res['volume_imbalance'] = volume.ewm(span=w_short).std() / volume.ewm(span=w_slow).std()
            
        if should_calc('liquidity_impact'):
            res['liquidity_impact'] = ret.abs().ewm(span=window).mean() / volume.ewm(span=window).mean()
        
        if should_calc('vol_rank'):
            vol = ret.rolling(window).std()
            res['vol_rank'] = vol.rolling(window).rank(pct=True)
            
        if should_calc('calmar_ratio'):
            sum_ret = ret.rolling(window).sum()
            roll_max = close.rolling(window).max()
            drawdown = (close - roll_max) / roll_max
            max_dd = drawdown.rolling(window).min().abs().replace(0, 1e-9)
            res['calmar_ratio'] = sum_ret / max_dd
            
        if should_calc('beta_btc') and benchmark_returns is not None:
            align_df = pd.concat([ret, benchmark_returns], axis=1, join='inner').dropna()
            if not align_df.empty:
                cov = align_df.iloc[:, 0].rolling(window).cov(align_df.iloc[:, 1])
                var = align_df.iloc[:, 1].rolling(window).var().replace(0, 1e-9)
                beta_series = cov / var
                res['beta_btc'] = beta_series.reindex(df.index).ffill()
                
        if should_calc('mr_prob'):
            # --- PineScript Parameters ---
            lookback = 100
            priceZLookback = 50
            adfEmaLen = 10
            adfSlowLen = 50
            crit = -1.5
            adfNormLen = 100
            divNormLen = 100
            slopeNormLen = 50
            volNormLen = 200
            priceZLow = -1.5
            priceZHigh = 1.5
            adfWeight = 0.4
            divWeight = 0.3
            slopeWeight = 0.15
            volWeight = 0.15
            adfSensitivity = 2.0
            divSensitivity = 1.0
            slopeSensitivity = 0.5
            volSensitivity = 1.5
            volShortLen = 30
            volLongLen = 200
            baseProb = 0.50
            modelScale = 1.0
            zClamp = 4.0
            # ---------------------------
            
            n = lookback - 1
            dy = close.diff()
            x = close.shift(1)
            sum_x = x.rolling(n).sum()
            sum_y = dy.rolling(n).sum()
            sum_xx = (x * x).rolling(n).sum()
            sum_xy = (x * dy).rolling(n).sum()
            denom = n * sum_xx - sum_x * sum_x
            beta = (n * sum_xy - sum_x * sum_y) / denom
            alpha = (sum_y - beta * sum_x) / n
            sum_yy = (dy * dy).rolling(n).sum()
            rss = sum_yy - alpha * sum_y - beta * sum_xy
            sigma2 = rss / max(n - 2, 1)
            se = np.sqrt(sigma2 * n / denom)
            adf = beta / se
            
            fast = adf.ewm(span=adfEmaLen, adjust=False).mean()
            slow = fast.rolling(adfSlowLen).mean()
            divergence = fast - slow
            divMean = divergence.rolling(divNormLen).mean()
            divStd = divergence.rolling(divNormLen).std(ddof=0)
            divZ = np.where(divStd > 1e-10, (divergence - divMean) / divStd, 0.0)
            
            divSlope = divergence.diff()
            slopeMean = divSlope.rolling(slopeNormLen).mean()
            slopeStd = divSlope.rolling(slopeNormLen).std(ddof=0)
            slopeZ = np.where(slopeStd > 1e-10, (divSlope - slopeMean) / slopeStd, 0.0)
            
            adfStd = adf.rolling(adfNormLen).std(ddof=0)
            adfZ = np.where(adfStd > 1e-10, (crit - adf) / adfStd, 0.0)
            
            atrShort = close.diff().abs().rolling(volShortLen).mean()
            atrLong = close.diff().abs().rolling(volLongLen).mean()
            volRatio = np.where(atrLong > 0, atrShort / atrLong, 1.0)
            logVolRatio = pd.Series(np.where(volRatio > 0, np.log(volRatio), 0.0), index=close.index)
            volStd = logVolRatio.rolling(volNormLen).std(ddof=0)
            volZ = np.where(volStd > 1e-10, logVolRatio / volStd, 0.0)
            
            adfScore = np.clip(adfZ, -zClamp, zClamp)
            divScore = np.clip(-divZ, -zClamp, zClamp)
            slopeScore = np.clip(-slopeZ, -zClamp, zClamp)
            volScore = np.clip(volZ, -zClamp, zClamp)
            
            adfLogit = adfSensitivity * adfScore
            divLogit = divSensitivity * divScore
            slopeLogit = slopeSensitivity * slopeScore
            volLogit = volSensitivity * volScore
            
            weightSum = adfWeight + divWeight + slopeWeight + volWeight
            weightedLogit = np.where(weightSum > 0, (adfWeight * adfLogit + divWeight * divLogit + slopeWeight * slopeLogit + volWeight * volLogit) / weightSum, 0.0)
            
            modelLogit = np.log(baseProb / (1.0 - baseProb)) + modelScale * weightedLogit
            mrProbability = np.where(modelLogit >= 0, 1.0 / (1.0 + np.exp(-modelLogit)), np.exp(modelLogit) / (1.0 + np.exp(modelLogit)))
            res['mr_prob'] = pd.Series(mrProbability * 100.0, index=close.index)
        
        # 228. FIP (Frog-in-the-Pan)
        if should_calc('fip'):
            def fip_func(x):
                pos = x[x > 0].sum()
                neg = np.abs(x[x < 0].sum())
                
                denom = pos + neg
                if denom == 0:
                    return 0.0
                
                return (pos - neg) / denom

            res['fip'] = ret.rolling(window).apply(fip_func, raw=True)
        
        # Price Z-Score & SMA Diff
        if should_calc('price_zscore'):
            res['price_zscore'] = (close - close.rolling(window=window).mean()) / close.rolling(window=window).std()
        
        # 271. ADF Statistics
        if should_calc('adf_hist') or should_calc('adf_stat'):
            hist, tau, _ = MetricsEngine.calculate_custom_adf_series(close.values, lookback=window)
            if should_calc('adf_hist'): res['adf_hist'] = hist.values
            if should_calc('adf_stat'): res['adf_stat'] = tau.values

        # 272. Rolling Drawdown Metrics
        if should_calc('max_drawdown') or should_calc('avg_drawdown'):
            # MDD(t) = (Price(t) / RollingMax(t)) - 1
            rolling_max = low.rolling(window=window, min_periods=1).max()
            drawdowns = (low / rolling_max) - 1.0
            
            if should_calc('max_drawdown'):
                # Max drawdown in the rolling window
                res['max_drawdown'] = drawdowns.rolling(window=window, min_periods=1).min()
            
            if should_calc('avg_drawdown'):
                # Average of drawdown events (where DD < 0)
                res['avg_drawdown'] = drawdowns.where(drawdowns < 0).rolling(window=window, min_periods=1).mean()

        # 273. Breakout Score
        if should_calc('breakout_score_dist') or should_calc('breakout_score_break'):
            bku_dist, bkd_dist, bku_break, bkd_break = MetricsEngine.calculate_breakout_score(df, len_up=30, len_down=30)
            if should_calc('breakout_score_dist'): res['breakout_score_dist'] = bku_dist - bkd_dist
            if should_calc('breakout_score_break'): res['breakout_score_break'] = bku_break - bkd_break

        return res

    @staticmethod
    def calculate_beta_alpha(asset_returns: np.ndarray, benchmark_returns: np.ndarray) -> tuple:
        """
        Calculate Alpha and Beta relative to benchmark.
        y = alpha + beta * x
        Using numpy linear algebra for speed.
        """
        if len(asset_returns) != len(benchmark_returns):
            # Align lengths
            min_len = min(len(asset_returns), len(benchmark_returns))
            asset_returns = asset_returns[-min_len:]
            benchmark_returns = benchmark_returns[-min_len:]
            
        # Standard OLS: beta = Cov(x,y) / Var(x)
        # alpha = mean(y) - beta * mean(x)
        
        # Stack x for linalg (add constant term for intercept/alpha)
        # A = [x, 1]
        x = benchmark_returns
        y = asset_returns
        
        # Manual calculation is often faster/simpler for 1D
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        covariance = np.mean((x - x_mean) * (y - y_mean))
        variance_x = np.mean((x - x_mean)**2)
        
        if variance_x == 0:
            return 0.0, 0.0, 0.0 # Beta, Alpha, R2
            
        beta = covariance / variance_x
        alpha = y_mean - beta * x_mean
        
        # R-squared
        # SS_res = sum((y - (alpha + beta*x))^2)
        # SS_tot = sum((y - mean(y))^2)
        y_pred = alpha + beta * x
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y_mean)**2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return beta, alpha, r_squared

    @staticmethod
    def check_stationarity(prices: np.ndarray, max_lag: int = None) -> tuple:
        """
        ADF Test.
        Returns: (adf_stat, p_value, is_stationary)
        """
        # Use statsmodels adfuller, but ensure inputs are clean
        # Remove NaNs
        clean_prices = prices[~np.isnan(prices)]
        if len(clean_prices) < 20: # Not enough data
            return 0.0, 1.0, False
            
        try:
            # autolag='AIC' helps determine lags
            result = adfuller(clean_prices, maxlag=max_lag, autolag='AIC')
            stat = result[0]
            p_value = result[1]
            is_stationary = p_value < 0.05
            return stat, p_value, is_stationary
        except Exception:
            return 0.0, 1.0, False

    @staticmethod
    def calculate_volatility(log_returns: np.ndarray, interval: str = '1h') -> float:
        if len(log_returns) < 2: return 0.0
        ann_factor = MetricsEngine.get_annual_scaling(interval)
        return float(np.std(log_returns) * np.sqrt(ann_factor))

    @staticmethod
    def calculate_custom_adf(series: np.ndarray, lookback: int = 20) -> tuple:
        """
        Fast ADF implementation based on PineScript logic.
        Returns: (hist, tau_adf, tau_smooth)
        Effectively linear regression of diffs vs lagged values.
        """
    @staticmethod
    def calculate_custom_adf_series(series: np.ndarray, lookback: int = 20) -> tuple:
        """
        Fast ADF implementation returning full series for rolling usage.
        Returns: (hist_series, tau_sma_series, tau_smooth_series)
        """
        # Ensure prices is clean (replace inf/nan with 0 or last value?)
        # For ADF, dropping NaNs is usually better than filling with 0 if gaps are sparse.
        # But for constant spacing, ffill is better.
        series = pd.Series(series).ffill().fillna(0).values

        if len(series) < lookback + 5:
            nan_s = pd.Series(np.nan, index=range(len(series)))
            return nan_s, nan_s, nan_s

        # Prepare arrays
        vals = series
        dy = np.diff(vals)
        x = vals[:-1]
        
        # Align lengths (dy has length N-1)
        n = len(dy)
        if n < lookback: 
            nan_s = pd.Series(np.nan, index=range(len(series)))
            return nan_s, nan_s, nan_s
                
        dy_s = pd.Series(dy)
        x_s = pd.Series(x)
        
        # Rolling sums with min_periods=1 for robustness
        sum_x = x_s.ewm(span=lookback, min_periods=1).sum()
        sum_y = dy_s.ewm(span=lookback, min_periods=1).sum()
        sum_xx = (x_s**2).ewm(span=lookback, min_periods=1).sum()
        sum_xy = (x_s * dy_s).ewm(span=lookback, min_periods=1).sum()
        sum_yy = (dy_s**2).ewm(span=lookback, min_periods=1).sum()
        
        # Helper for vectorized beta, alpha
        denom = lookback * sum_xx - sum_x**2
        
        # Handle division by zero
        valid = np.abs(denom) > 1e-10
        beta = np.zeros_like(denom)
        alpha = np.zeros_like(denom)
        
        beta[valid] = (lookback * sum_xy[valid] - sum_x[valid] * sum_y[valid]) / denom[valid]
        alpha[valid] = (sum_y[valid] - beta[valid] * sum_x[valid]) / lookback
        
        # SE calculation requires RSS
        rss = (sum_yy + 
               alpha**2 * lookback + 
               beta**2 * sum_xx + 
               2 * alpha * beta * sum_x - 
               2 * alpha * sum_y - 
               2 * beta * sum_xy)
               
        # Ensure positive RSS (floating point errors)
        rss = np.maximum(rss, 0)
        
        sigma2 = rss / max(lookback - 2, 1)
        
        # SE = sqrt(sigma2 * n / denom) = sqrt(sigma2 * lookback / denom)
        se = np.sqrt(sigma2 * lookback / np.maximum(denom, 1e-10))
        
        # t-stat (tau)
        tau = np.zeros_like(beta)
        mask = se > 1e-10
        tau[mask] = beta[mask] / se[mask]
        
        # Convert to series for smoothing
        tau_s = pd.Series(tau).fillna(0)
        
        # Verify: "tauADF := ta.sma(tauADF, 10)"
        tau_sma = tau_s.ewm(span=10, min_periods=1).mean()
        
        # "tauADF_smooth = ta.ema(tauADF, 50)"
        # EMA usually handles NaNs/start better than SMA
        tau_smooth = tau_sma.ewm(span=50, adjust=False, min_periods=1).mean()
        
        hist = tau_sma - tau_smooth
                
        hist = pd.concat([pd.Series([np.nan]), hist], ignore_index=True)
        tau_sma = pd.concat([pd.Series([np.nan]), tau_sma], ignore_index=True)
        tau_smooth = pd.concat([pd.Series([np.nan]), tau_smooth], ignore_index=True)
        
        return hist, tau_sma, tau_smooth

    @staticmethod
    def calculate_custom_adf(series: np.ndarray, lookback: int = 20) -> tuple:
        """
        Wrapper for single value return.
        """
        hist, tau, smooth = MetricsEngine.calculate_custom_adf_series(series, lookback)
        # Return the *latest* value (last valid point)
        return hist.iloc[-1], tau.iloc[-1], smooth.iloc[-1]

    @staticmethod
    def calculate_breakout_score(df: pd.DataFrame, 
                                 len_up: int = 30, len_down: int = 30, 
                                 mult_base: float = 0.5, calcMethod: str = "Atr",
                                 maMin: int = 10, period: int = 50,
                                 volFactor: float = 0.15, k_decay: float = 2.0, eps: float = 0.2, persistBars: int = 3) -> tuple:
        """
        Calculates breakout trendline score (translated from Market_radar Numba logic).
        Returns: score_up_dist, score_down_dist, score_up_break, score_down_break
        """
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        n = len(close)
        
        if n < max(len_up, len_down, period):
            return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)

        def stdev_func(x, n_period):
            return pd.Series(x).rolling(n_period).std(ddof=0).values

        tr = np.zeros_like(close)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr_len_up = _rma(tr, len_up)
        atr_len_down = _rma(tr, len_down)
        atr_14 = _rma(tr, 14)
        
        logret = np.zeros_like(close)
        with np.errstate(divide='ignore', invalid='ignore'):
            logret[1:] = np.log(close[1:] / close[:-1])
        logret[0] = 0.0
        logret = np.nan_to_num(logret)
        
        vol_vama = stdev_func(logret, period)
        targetVol = stdev_func(logret, period * 30) * volFactor
        
        vamaC = np.full(n, np.nan)
        
        span_val = float(maMin)
        for i in range(n):
            if not np.isnan(vol_vama[i]) and vol_vama[i] > 0 and not np.isnan(targetVol[i]):
                scale = min(targetVol[i] / vol_vama[i], 1.0)
                span_val = max(np.floor(maMin / scale), 1.0)
            
            alpha = 1.0 / (span_val + 1.0)
            
            if i == 0 or np.isnan(vamaC[i-1]):
                vamaC[i] = close[i]
            else:
                vamaC[i] = alpha * close[i] + (1 - alpha) * vamaC[i-1]

        vol_sma = pd.Series(vol_vama).rolling(len_up, min_periods=1).mean().values
        vr_mult = np.ones(n)
        mask = vol_sma > 0
        vr_mult[mask] = vol_vama[mask] / vol_sma[mask]

        beta_mult = 0.7
        mult_dyn = mult_base * np.power(vr_mult, -beta_mult)
        mult_dyn = np.clip(mult_dyn, mult_base * 0.1, mult_base * 5.0)
        
        ph = _get_pivots(high, len_up, len_up, True)
        pl = _get_pivots(low, len_down, len_down, False)
        
        slope_up = np.zeros(n)
        slope_down = np.zeros(n)
        
        if calcMethod == "Atr":
            slope_up = (atr_len_up / len_up) * mult_dyn
            slope_down = (atr_len_down / len_down) * mult_dyn
        elif calcMethod == "Stdev":
            slope_up = (stdev_func(close, len_up) / len_up) * mult_dyn
            slope_down = (stdev_func(close, len_down) / len_down) * mult_dyn
        elif calcMethod == "Linreg":
            def sma(x, p):
                return pd.Series(x).rolling(p).mean().values
            def variance(x, p):
                return pd.Series(x).rolling(p).var(ddof=0).values
                
            bar_index = np.arange(n)
            sma_src_n_up = sma(close * bar_index, len_up)
            sma_src_up = sma(close, len_up)
            sma_n_up = sma(bar_index, len_up)
            var_n_up = variance(bar_index, len_up)
            var_safe_up = np.where(var_n_up == 0, 1.0, var_n_up)
            slope_up = np.abs(sma_src_n_up - sma_src_up * sma_n_up) / var_safe_up / 2 * mult_dyn
            
            sma_src_n_down = sma(close * bar_index, len_down)
            sma_src_down = sma(close, len_down)
            sma_n_down = sma(bar_index, len_down)
            var_n_down = variance(bar_index, len_down)
            var_safe_down = np.where(var_n_down == 0, 1.0, var_n_down)
            slope_down = np.abs(sma_src_n_down - sma_src_down * sma_n_down) / var_safe_down / 2 * mult_dyn

        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        slope_ph = np.full(n, np.nan)
        slope_pl = np.full(n, np.nan)
        
        upper_proj = np.full(n, np.nan)
        lower_proj = np.full(n, np.nan)
        
        score_up_dist = np.zeros(n)
        score_down_dist = np.zeros(n)
        score_up_break = np.zeros(n)
        score_down_break = np.zeros(n)
        
        last_valid_up = -1
        last_valid_down = -1
        
        atr_smoothed = pd.Series(atr_14).rolling(10, min_periods=1).mean().values
        
        for i in range(1, n):
            is_ph = not np.isnan(ph[i])
            is_pl = not np.isnan(pl[i])
            
            if is_ph:
                slope_ph[i] = slope_up[i]
                upper[i] = ph[i]
            else:
                slope_ph[i] = slope_ph[i-1]
                upper[i] = upper[i-1]
                
            if is_pl:
                slope_pl[i] = slope_down[i]
                lower[i] = pl[i]
            else:
                slope_pl[i] = slope_pl[i-1]
                lower[i] = lower[i-1]
                
            if not np.isnan(upper[i]):
                upper[i] = upper[i] - (slope_ph[i] if not np.isnan(slope_ph[i]) else 0.0)
                
            if not np.isnan(lower[i]):
                lower[i] = lower[i] + (slope_pl[i] if not np.isnan(slope_pl[i]) else 0.0)
                
            upper_proj[i] = upper[i] - (slope_ph[i] * len_up if not np.isnan(slope_ph[i]) else 0.0)
            lower_proj[i] = lower[i] + (slope_pl[i] * len_down if not np.isnan(slope_pl[i]) else 0.0)
            
            c = close[i]
            a = atr_smoothed[i] if not np.isnan(atr_smoothed[i]) and atr_smoothed[i] != 0 else 1.0
            
            if not np.isnan(upper_proj[i]):
                dist_up = abs(c - upper_proj[i]) / a
                score_up_dist[i] = np.exp(-k_decay * dist_up**2) * (1.0 / (dist_up + eps))
                
            if not np.isnan(lower_proj[i]):
                dist_down = abs(c - lower_proj[i]) / a
                score_down_dist[i] = np.exp(-k_decay * dist_down**2) * (1.0 / (dist_down + eps))
                
            break_up = False
            if i > 0 and not np.isnan(upper_proj[i-1]) and not np.isnan(upper_proj[i]):
                if close[i-1] <= upper_proj[i-1] and close[i] > upper_proj[i]:
                    break_up = True
                    
            break_down = False
            if i > 0 and not np.isnan(lower_proj[i-1]) and not np.isnan(lower_proj[i]):
                if close[i-1] >= lower_proj[i-1] and close[i] < lower_proj[i]:
                    break_down = True
                    
            raw_mom = close[i] - close[i-1]
            valid_break_up = break_up and raw_mom > 0 and close[i] > vamaC[i]
            valid_break_down = break_down and raw_mom < 0 and close[i] < vamaC[i]
            
            if valid_break_up:
                last_valid_up = i
            if valid_break_down:
                last_valid_down = i
                
            if last_valid_up != -1 and (i - last_valid_up) < persistBars:
                score_up_break[i] = 1.0
            else:
                score_up_break[i] = 0.0
                
            if last_valid_down != -1 and (i - last_valid_down) < persistBars:
                score_down_break[i] = 1.0
            else:
                score_down_break[i] = 0.0

        s_up_dist = pd.Series(score_up_dist).ewm(span=3, adjust=False).mean()
        s_down_dist = pd.Series(score_down_dist).ewm(span=3, adjust=False).mean()
        
        vama_c_s = pd.Series(vamaC)
        close_s = pd.Series(close)
        
        s_up_dist = np.where(close_s > vama_c_s, s_up_dist, 0.0)
        s_down_dist = np.where(close_s < vama_c_s, s_down_dist, 0.0)
        
        return (pd.Series(s_up_dist, index=df.index).fillna(0), 
                pd.Series(s_down_dist, index=df.index).fillna(0),
                pd.Series(score_up_break, index=df.index).fillna(0),
                pd.Series(score_down_break, index=df.index).fillna(0))

    @staticmethod
    def calculate_fip(returns: np.ndarray) -> float:
        """
        Frog-in-the-Pan (FIP) momentum from Quantitative Momentum.
        FIP = sign(Past return) * [% negative - % positive]
        """
        if len(returns) < 2: return 0.0
        
        total_ret = np.sum(returns)
        sign_ret = np.sign(total_ret)
        
        n = len(returns)
        neg_pct = np.sum(returns < 0) / n
        pos_pct = np.sum(returns > 0) / n
        
        return sign_ret * (neg_pct - pos_pct)

    @staticmethod
    def calculate_sharpe_ratio(log_returns: np.ndarray, risk_free_rate: float = 0.0, interval: str = '1h') -> float:
        if len(log_returns) < 2: return 0.0
        
        mean_ret = np.mean(log_returns)
        std_ret = np.std(log_returns)
        
        if std_ret < 1e-9: return 0.0
        
        ann_factor = MetricsEngine.get_annual_scaling(interval)
        # Annualized Sharpe = (Mean * Factor) / (Std * Sqrt(Factor)) = (Mean / Std) * Sqrt(Factor)
        return float((mean_ret / std_ret) * np.sqrt(ann_factor))

    @staticmethod
    def calculate_sortino_ratio(log_returns: np.ndarray, target: float = 0.0, interval: str = '1h') -> float:
        """
        Sortino Ratio = (Mean Return - Target) / Downside Deviation
        """
        if len(log_returns) < 2: return 0.0
        
        excess_returns = log_returns - target
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
            
        downside_std = np.std(downside_returns)
        if downside_std < 1e-9: return 0.0
        
        mean_ret = np.mean(excess_returns)
        ann_factor = MetricsEngine.get_annual_scaling(interval)
        
        return float((mean_ret / downside_std) * np.sqrt(ann_factor))

    @staticmethod
    def calculate_max_drawdown(prices: np.ndarray) -> float:
        """
        Max Drawdown = Min((Price / RunningMax) - 1)
        """
        if len(prices) < 2: return 0.0
        
        running_max = np.maximum.accumulate(prices)
        drawdowns = (prices / running_max) - 1
        return float(np.min(drawdowns))

    @staticmethod
    def calculate_avg_drawdown(prices: np.ndarray) -> float:
        """
        Average of all drawdowns (where DD < 0)
        """
        if len(prices) < 2: return 0.0
        
        running_max = np.maximum.accumulate(prices)
        drawdowns = (prices / running_max) - 1
        
        # Filter for actual drawdowns
        negative_dds = drawdowns[drawdowns < 0]
        if len(negative_dds) == 0: return 0.0
        
        return float(np.mean(negative_dds))

    @staticmethod
    def calculate_win_rate(log_returns: np.ndarray) -> float:
        """
        Percentage of positive returns.
        """
        if len(log_returns) == 0: return 0.0
        
        wins = np.sum(log_returns > 0)
        return float(wins / len(log_returns))

    @staticmethod
    def calculate_rolling_metric(df: pd.DataFrame, metric_name: str, window: int = 30, step: int = 1, benchmark_returns: pd.Series = None, benchmark_prices: pd.Series = None, interval: str = '1h') -> pd.Series:
        """
        Calculates a metric over a rolling window.
        Returns a Series of values indexed by the original index.
        """
        if df.empty or len(df) < 2:
             return pd.Series(dtype=float)
             
        # Clean prices: ffill is safest for time series continuity
        df = df.copy()
        df['close'] = pd.to_numeric(df['close'], errors='coerce').ffill().bfill()
        df['high'] = pd.to_numeric(df['high'], errors='coerce').ffill().bfill()
        df['low'] = pd.to_numeric(df['low'], errors='coerce').ffill().bfill()
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        
        # Calculate all standard indicators
        all_inds = MetricsEngine.calculate_all_indicators(df, window=window, benchmark_returns=benchmark_returns, benchmark_prices=benchmark_prices, interval=interval)
        
        if metric_name in all_inds.columns:
            res = all_inds[metric_name]
        else:
            # Fallback for metrics not in the all-in-one helper (if any)
            if metric_name == 'count':
                res = pd.Series(len(df), index=df.index)
            else:
                return pd.Series(dtype=float)
            
        # Apply Step (Stride)
        if step > 1:
            res = res.iloc[::step]
            
        return res

    def compute_all_metrics(self, prices_data: Dict[str, pd.DataFrame], interval: str = '1h', benchmark_symbol: str = 'BTCUSDT', benchmark_returns: Optional[pd.Series] = None, benchmark_prices: Optional[pd.Series] = None, window: int = 40) -> pd.DataFrame:
        """
        Main pipeline to compute metrics for all symbols.
        """
        results = []
        
        # 1. Prepare Benchmark if not provided
        if benchmark_prices is None and benchmark_symbol in prices_data:
            b_df = prices_data[benchmark_symbol]
            benchmark_prices = pd.to_numeric(b_df['close'], errors='coerce').ffill().fillna(0)
        
        if benchmark_prices is not None and benchmark_returns is None:
                benchmark_returns = benchmark_prices.pct_change().dropna()
            
        from src.data import BinanceFuturesFetcher
        import time
        fetcher = BinanceFuturesFetcher()
        funding_rates = fetcher.get_funding_rates()
        adl_risks = fetcher.get_all_adl_risks()

        # Map interval to Binance statistics period
        period_mapping = {
            '1m': '5m',
            '3m': '5m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '8h': '6h',
            '12h': '12h',
            '1d': '1d',
            '3d': '1d',
            '1w': '1d',
            '1M': '1d'
        }
        period = period_mapping.get(interval, '1h')
        
        # Pre-fetch per-symbol API stats in parallel to avoid sequential blocking
        import concurrent.futures
        
        api_stats_cache = {}
        def fetch_symbol_stats(sym):
            try:
                oi_hist = fetcher.get_historical_stats(sym, period, fetcher.OPEN_INTEREST_HIST)
                top_pos = fetcher.get_historical_stats(sym, period, fetcher.TOP_LS_POSITION)
                top_acc = fetcher.get_historical_stats(sym, period, fetcher.TOP_LS_ACCOUNT)
                glob_acc = fetcher.get_historical_stats(sym, period, fetcher.GLOBAL_LS_ACCOUNT)
                taker_ratio = fetcher.get_historical_stats(sym, period, fetcher.TAKER_BUY_SELL)
                return sym, {
                    'oi_hist': oi_hist,
                    'top_pos': top_pos,
                    'top_acc': top_acc,
                    'glob_acc': glob_acc,
                    'taker_ratio': taker_ratio
                }
            except Exception as e:
                print(f"Error pre-fetching stats for {sym}: {e}")
                return sym, {}

        # Fetch missing symbols (not already fetched recently or dynamically)
        symbols_to_fetch = list(prices_data.keys())
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_sym = {executor.submit(fetch_symbol_stats, sym): sym for sym in symbols_to_fetch}
            for future in concurrent.futures.as_completed(future_to_sym):
                sym, data = future.result()
                api_stats_cache[sym] = data
            
        for symbol, df in prices_data.items():
            try:
                if df.empty: continue
                
                # Cache Check
                last_ts = df['open_time'].max() if 'open_time' in df.columns else None
                cache_key = (symbol, interval, last_ts, len(df))
                if cache_key in self._results_cache:
                    cached_entry = self._results_cache[cache_key]
                    if isinstance(cached_entry, dict) and 'compute_time' in cached_entry and 'data' in cached_entry:
                        if time.time() - cached_entry['compute_time'] < 60:
                            results.append(cached_entry['data'])
                            continue

                # Standardize and clean prices
                prices = pd.to_numeric(df['close'], errors='coerce').ffill().fillna(0).values.astype(float)
                if len(prices) < 2: continue
                
                # Advanced Metrics (Standardized)
                adv_df = self.calculate_all_indicators(df, window=window, benchmark_returns=benchmark_returns, benchmark_prices=benchmark_prices, interval=interval, tail_only=True)
                latest_adv = adv_df.iloc[-1].to_dict()
                
                row = {
                    'symbol': symbol,
                    'count': len(prices)
                }
                # Add all standard metrics to row
                row.update(latest_adv)
                row['funding_rate'] = funding_rates.get(symbol, np.nan)
                
                # Retrieve pre-fetched statistics
                sym_stats = api_stats_cache.get(symbol, {})
                oi_hist = sym_stats.get('oi_hist', {})
                top_pos = sym_stats.get('top_pos', {})
                top_acc = sym_stats.get('top_acc', {})
                glob_acc = sym_stats.get('glob_acc', {})
                taker_ratio = sym_stats.get('taker_ratio', {})
                
                # Compute OI / Circulating Supply
                sum_oi = float(oi_hist.get("sumOpenInterest", 0))
                circ_supply = float(oi_hist.get("CMCCirculatingSupply", 0))
                oi_cs = sum_oi / circ_supply if circ_supply > 0 else np.nan
                
                row['oi_circulating'] = oi_cs
                row['top_ls_position_ratio'] = float(top_pos.get("longShortRatio", np.nan))
                row['top_ls_account_ratio'] = float(top_acc.get("longShortRatio", np.nan))
                row['global_ls_ratio'] = float(glob_acc.get("longShortRatio", np.nan))
                row['taker_buysell_ratio'] = float(taker_ratio.get("buySellRatio", taker_ratio.get("longShortRatio", np.nan)))
                row['adl_risk'] = float(adl_risks.get(symbol, 0))
                
                # Store in cache
                self._results_cache[cache_key] = {
                    'data': row,
                    'compute_time': time.time()
                }
                results.append(row)
                
            except Exception as e:
                print(f"Error calculating metrics for {symbol}: {e}")
                
        return pd.DataFrame(results)


import numpy as np
from scipy.stats import norm, t

def copula_cond_probs(u_hist, v_hist, u_curr, v_curr, method="gaussian", **kwargs):
    """
    Compute conditional probabilities:
        P(U <= u_curr | V=v_curr) and P(V <= v_curr | U=u_curr)
    
    Parameters
    ----------
    u_hist, v_hist : array-like
        Historical percentiles (0-1)
    u_curr, v_curr : float
        Current percentiles (0-1)
    method : str
        Copula type: "gaussian", "t", "clayton", "gumbel"
    kwargs : dict
        Extra parameters:
            - For "t": df = degrees of freedom
            - For "clayton"/"gumbel": theta = copula parameter (>0)

    Returns
    -------
    tuple
        (P(U <= u_curr | V=v_curr), P(V <= v_curr | U=u_curr))
    """
    u_hist = np.asarray(u_hist)
    v_hist = np.asarray(v_hist)
    
    if method.lower() == "gaussian":
        z1_hist = norm.ppf(u_hist)
        z2_hist = norm.ppf(v_hist)
        z1_curr = norm.ppf(u_curr)
        z2_curr = norm.ppf(v_curr)
        rho = np.corrcoef(z1_hist, z2_hist)[0,1]

        p_uv = norm.cdf((z1_curr - rho * z2_curr)/np.sqrt(1 - rho**2))
        p_vu = norm.cdf((z2_curr - rho * z1_curr)/np.sqrt(1 - rho**2))
        return p_uv, p_vu

    elif method.lower() == "t":
        df = kwargs.get("df", 5)
        z1_hist = t.ppf(u_hist, df)
        z2_hist = t.ppf(v_hist, df)
        z1_curr = t.ppf(u_curr, df)
        z2_curr = t.ppf(v_curr, df)
        rho = np.corrcoef(z1_hist, z2_hist)[0,1]

        cond_var_uv = ((df + z2_curr**2)/(df+1))*(1 - rho**2)
        cond_mean_uv = rho * z2_curr
        cond_var_vu = ((df + z1_curr**2)/(df+1))*(1 - rho**2)
        cond_mean_vu = rho * z1_curr

        p_uv = t.cdf(z1_curr, df+1, loc=cond_mean_uv, scale=np.sqrt(cond_var_uv))
        p_vu = t.cdf(z2_curr, df+1, loc=cond_mean_vu, scale=np.sqrt(cond_var_vu))
        return p_uv, p_vu

    elif method.lower() == "clayton":
        theta = kwargs.get("theta", 2)
        if theta <= 0:
            raise ValueError("Clayton copula θ must be > 0")

        base = np.maximum(u_curr**(-theta) + v_curr**(-theta) - 1, 1e-10)
        p_uv = (v_curr**(-theta - 1)) * (base**(-(1+1/theta)))
        p_vu = (u_curr**(-theta - 1)) * (base**(-(1+1/theta)))
        return p_uv, p_vu

    elif method.lower() == "gumbel":
        theta = kwargs.get("theta", 2)
        if theta < 1:
            raise ValueError("Gumbel copula θ must be ≥ 1")

        u_tilde = -np.log(u_curr)
        v_tilde = -np.log(v_curr)
        sum_t = np.maximum(u_tilde**theta + v_tilde**theta, 1e-10)
        
        C = np.exp(-(sum_t**(1/theta)))
        
        p_uv = C * (1.0 / v_curr) * (sum_t**(1/theta - 1)) * (v_tilde**(theta - 1))
        p_vu = C * (1.0 / u_curr) * (sum_t**(1/theta - 1)) * (u_tilde**(theta - 1))
        return p_uv, p_vu

    else:
        raise ValueError(f"Unknown copula method: {method}")