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

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=n, min_periods=n).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=n, min_periods=n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def rsi_adapt(src, w_short, w_long, volFactor=1.0):
    ret = np.log(src / src.shift(1))
    vS = ret.rolling(w_short).std()
    vL = ret.rolling(w_long).std()
    a = np.clip(vS / (vL * volFactor), 0.01, 0.99)
    return a * rsi(src, w_short) + (1 - a) * rsi(src, w_long)

def vama_cma(src, w_short, w_long, volFactor=1.0):
    ret = np.log(src / src.shift(1))
    vS = ret.rolling(w_short).std()
    vL = ret.rolling(w_long).std()
    a = 2.0 / (np.maximum(1, np.floor(10.0 / np.minimum(vS / (vL * volFactor), 1.0))) + 1.0)
    
    vama_vals = _adaptive_ema_numba(src.values, a.values)
    vama = pd.Series(vama_vals, index=src.index)
    
    proxy_atr = src.diff().abs().rolling(10).mean().replace(0, 1e-9)
    return (src - vama) / proxy_atr

def ewmac(src, w_short, w_long):
    fast = src.ewm(span=w_short, adjust=False).mean()
    slow = src.ewm(span=w_long, adjust=False).mean()
    v = src.rolling(w_short).std().replace(0, 1e-9)
    return (fast - slow) / v

def z_score(src, length):
    return (src - src.rolling(length).mean()) / src.rolling(length).std().replace(0, 1e-9)

np.random.seed(42)
prices = pd.Series(np.exp(np.random.randn(1000).cumsum() * 0.01))
c = vama_cma(prices, 40, 200)
print(c.tail())
