import pandas as pd
import numpy as np
from src.metrics import MetricsEngine

np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=300, freq="h")
df = pd.DataFrame({
    'open': np.exp(np.random.randn(300).cumsum() * 0.01),
    'high': np.exp(np.random.randn(300).cumsum() * 0.01 + 0.05),
    'low': np.exp(np.random.randn(300).cumsum() * 0.01 - 0.05),
    'close': np.exp(np.random.randn(300).cumsum() * 0.01),
    'volume': np.random.rand(300) * 1000
}, index=dates)

bench_prices = pd.Series(np.exp(np.random.randn(300).cumsum() * 0.01), index=dates)

# Let's mock how backend calls it with tail_only
res = MetricsEngine.calculate_all_indicators(df, window=40, benchmark_prices=bench_prices, include_metrics=['rel_strength_z'], tail_only=True)

print(res['rel_strength_z'].tail())
