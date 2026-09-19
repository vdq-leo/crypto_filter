import numpy as np
import pandas as pd
from src.metrics import MetricsEngine
from src.data import DataManager

def f_clamp(x, limit):
    return np.clip(x, -limit, limit)

def f_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def calculate_mr_prob(close: pd.Series, lookback=100) -> pd.Series:
    # We need to compute raw adf first.
    # The PineScript lookback is 100 for ADF.
    # We will use calculate_custom_adf_series but we need the raw tau.
    # Let's write the raw tau logic.
    n = lookback - 1
    dy = close.diff()
    x = close.shift(1)
    
    # We can use rolling to compute sum_x, sum_y, sum_xx, sum_xy
    # But PineScript does rolling sum over n bars
    sum_x = x.rolling(n).sum()
    sum_y = dy.rolling(n).sum()
    sum_xx = (x * x).rolling(n).sum()
    sum_xy = (x * dy).rolling(n).sum()
    
    denom = n * sum_xx - sum_x * sum_x
    beta = (n * sum_xy - sum_x * sum_y) / denom
    alpha = (sum_y - beta * sum_x) / n
    
    # rss requires computing e = dy - (alpha + beta * x) for all elements in the window.
    # A faster way: sum(e^2) = sum((dy - (alpha + beta * x))^2)
    # = sum(dy^2) + alpha^2*n + beta^2*sum_x^2 - 2*alpha*sum_y - 2*beta*sum_xy + 2*alpha*beta*sum_x
    # Wait, actually sum_yy - alpha*sum_y - beta*sum_xy
    sum_yy = (dy * dy).rolling(n).sum()
    rss = sum_yy - alpha * sum_y - beta * sum_xy
    
    sigma2 = rss / max(n - 2, 1)
    se = np.sqrt(sigma2 * n / denom)
    
    adf = beta / se
    
    adfEmaLen = 10
    adfSlowLen = 50
    crit = -1.5
    adfNormLen = 100
    divNormLen = 100
    slopeNormLen = 50
    volNormLen = 200
    volShortLen = 30
    volLongLen = 200
    
    fast = adf.ewm(span=adfEmaLen, adjust=False).mean()
    slow = fast.rolling(adfSlowLen).mean()
    
    divergence = fast - slow
    divMean = divergence.rolling(divNormLen).mean()
    divStd = divergence.rolling(divNormLen).std(ddof=0)
    divZ = np.where(divStd > 1e-10, (divergence - divMean) / divStd, 0.0)
    
    divSlope = divergence - divergence.shift(1)
    slopeMean = divSlope.rolling(slopeNormLen).mean()
    slopeStd = divSlope.rolling(slopeNormLen).std(ddof=0)
    slopeZ = np.where(slopeStd > 1e-10, (divSlope - slopeMean) / slopeStd, 0.0)
    
    adfStd = adf.rolling(adfNormLen).std(ddof=0)
    adfZ = np.where(adfStd > 1e-10, (crit - adf) / adfStd, 0.0)
    
    # True Range for volatility. We don't have high/low here easily inside metrics engine unless we pass them.
    # In MetricsEngine, we usually just use close for indicators if we don't pass H/L.
    # Or we can just use std of returns as proxy for ATR if H/L not provided?
    return adf

manager = DataManager()
df = manager.load_data("BTCUSDT", "1h", auto_sync=False)
if df is not None:
    res = calculate_mr_prob(df['close'].head(1000))
    print("Computed MR Prob ADF head/tail:\n", res.dropna().head(), res.dropna().tail())
