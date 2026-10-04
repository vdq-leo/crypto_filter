import numpy as np
import pandas as pd
from src.metrics import MetricsEngine
from src.data import DataManager

def f_clamp(x, limit):
    return np.clip(x, -limit, limit)

def f_sigmoid(x):
    # Using np.where to handle the condition x >= 0 like in PineScript
    # PineScript: x >= 0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x))
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

def calculate_mr_prob(df: pd.DataFrame, lookback=100) -> pd.DataFrame:
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    
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
    
    adfEmaLen = 10
    adfSlowLen = 50
    crit = -1.5
    adfNormLen = 100
    divNormLen = 100
    slopeNormLen = 50
    volNormLen = 200
    priceZLookback = 50
    volShortLen = 30
    volLongLen = 200
    
    fast = adf.ewm(span=adfEmaLen, adjust=False).mean()
    slow = fast.rolling(adfSlowLen).mean()
    
    divergence = fast - slow
    divMean = divergence.rolling(divNormLen).mean()
    divStd = divergence.rolling(divNormLen).std(ddof=1)
    divZ = np.where(divStd > 1e-10, (divergence - divMean) / divStd, 0.0)
    
    divSlope = divergence.diff()
    slopeMean = divSlope.rolling(slopeNormLen).mean()
    slopeStd = divSlope.rolling(slopeNormLen).std(ddof=1)
    slopeZ = np.where(slopeStd > 1e-10, (divSlope - slopeMean) / slopeStd, 0.0)
    
    adfStd = adf.rolling(adfNormLen).std(ddof=1)
    adfZ = np.where(adfStd > 1e-10, (crit - adf) / adfStd, 0.0)
    
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    atrShort = tr.ewm(alpha=1.0 / volShortLen, adjust=False).mean()
    atrLong = tr.ewm(alpha=1.0 / volLongLen, adjust=False).mean()
    
    volRatio = np.where(atrLong > 0, atrShort / atrLong, 1.0)
    logVolRatio = pd.Series(np.where(volRatio > 0, np.log(volRatio), 0.0), index=close.index)
    volStd = logVolRatio.rolling(volNormLen).std(ddof=1)
    volZ = np.where(volStd > 1e-10, logVolRatio / volStd, 0.0)
    
    priceMean = close.rolling(priceZLookback).mean()
    priceStd = close.rolling(priceZLookback).std(ddof=1)
    priceZ = np.where(priceStd > 0, (close - priceMean) / priceStd, 0.0)
    
    zClamp = 4.0
    adfScore = f_clamp(adfZ, zClamp)
    divScore = f_clamp(-divZ, zClamp)
    slopeScore = f_clamp(-slopeZ, zClamp)
    volScore = f_clamp(volZ, zClamp)
    
    adfSensitivity = 2.0
    divSensitivity = 1.0
    slopeSensitivity = 0.5
    volSensitivity = 1.5
    
    adfLogit = adfSensitivity * adfScore
    divLogit = divSensitivity * divScore
    slopeLogit = slopeSensitivity * slopeScore
    volLogit = volSensitivity * volScore
    
    adfProb = f_sigmoid(adfLogit)
    divProb = f_sigmoid(divLogit)
    slopeProb = f_sigmoid(slopeLogit)
    volProb = f_sigmoid(volLogit)
    
    adfWeight = 0.4
    divWeight = 0.3
    slopeWeight = 0.15
    volWeight = 0.15
    
    weightSum = adfWeight + divWeight + slopeWeight + volWeight
    weightedLogit = np.where(weightSum > 0, (adfWeight * adfLogit + divWeight * divLogit + slopeWeight * slopeLogit + volWeight * volLogit) / weightSum, 0.0)
    
    baseProb = 0.50
    modelScale = 1.0
    
    baseLogit = np.log(baseProb / (1.0 - baseProb))
    modelLogit = baseLogit + modelScale * weightedLogit
    mrProbability = f_sigmoid(modelLogit)
    mrProbabilityPct = mrProbability * 100.0
    
    return pd.DataFrame({
        "price_z": priceZ,
        "mr_prob": mrProbabilityPct,
        "adf_prob": adfProb * 100.0,
        "div_prob": divProb * 100.0,
        "slope_prob": slopeProb * 100.0,
        "vol_prob": volProb * 100.0,
        "vol_ratio": volRatio
    }, index=close.index)

manager = DataManager()
df = manager.load_data("BTCUSDT", "1h", auto_sync=False)
if df is not None:
    res = calculate_mr_prob(df)
    print("Computed MR Prob Pct head/tail:\n", res.dropna().tail())
