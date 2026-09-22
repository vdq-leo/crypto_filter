import numpy as np
import pandas as pd
import numba

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
    @staticmethod
    def rsi(series: pd.Series, n: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=n, min_periods=n).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=n, min_periods=n).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

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

np.random.seed(42)
asset_p = pd.Series(np.exp(np.random.randn(1000).cumsum() * 0.01))
btc_p = pd.Series(np.exp(np.random.randn(1000).cumsum() * 0.01))

p_short = 40
p_long = 200
beta_len = 100
smooth = 10
volFactor = 1.0

a_rsi = MetricsEngine.z_score_indicator(MetricsEngine.rsi_adapt(asset_p, p_short, p_long, volFactor), beta_len)
a_cma = MetricsEngine.z_score_indicator(MetricsEngine.vama_cma(asset_p, p_short, p_long, volFactor), beta_len)
a_ewmac = MetricsEngine.z_score_indicator(MetricsEngine.ewmac(asset_p, p_short, p_long), beta_len)
sym2_idx = (a_rsi + a_cma + a_ewmac) / 3.0

b_rsi = MetricsEngine.z_score_indicator(MetricsEngine.rsi_adapt(btc_p, p_short, p_long, volFactor), beta_len)
b_cma = MetricsEngine.z_score_indicator(MetricsEngine.vama_cma(btc_p, p_short, p_long, volFactor), beta_len)
b_ewmac = MetricsEngine.z_score_indicator(MetricsEngine.ewmac(btc_p, p_short, p_long), beta_len)
sym1_idx = (b_rsi + b_cma + b_ewmac) / 3.0

spread = sym1_idx - sym2_idx
diff = spread.diff()

sVol = diff.rolling(p_short).std().clip(lower=1e-6)
lVol = diff.rolling(p_long).std().clip(lower=1e-6)

alpha = ((1.0 / smooth) / ((lVol * volFactor) / sVol)).clip(lower=0.05, upper=0.95)
alpha = alpha.fillna(0.05)

spread_s_vals = _adaptive_ema_numba(spread.values, alpha.values)
spread_s = pd.Series(spread_s_vals, index=asset_p.index).fillna(0.0)
print(spread_s.tail())
