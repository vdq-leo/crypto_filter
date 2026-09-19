import numpy as np
import pandas as pd
from src.metrics import MetricsEngine

prices = np.array([100, 102, 104, 103, 101, 100, 99, 98, 100, 105] * 20)
h, t, s = MetricsEngine.calculate_custom_adf_series(prices, lookback=100)
print("hist mean:", np.nanmean(h))
print("tau mean:", np.nanmean(t))
print("smooth mean:", np.nanmean(s))
