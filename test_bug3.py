import pandas as pd
import numpy as np
from src.metrics import MetricsEngine

np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=1000, freq="h")
df = pd.DataFrame({
    'open': np.exp(np.random.randn(1000).cumsum() * 0.01),
    'high': np.exp(np.random.randn(1000).cumsum() * 0.01 + 0.05),
    'low': np.exp(np.random.randn(1000).cumsum() * 0.01 - 0.05),
    'close': np.exp(np.random.randn(1000).cumsum() * 0.01),
    'volume': np.random.rand(1000) * 1000
}, index=dates)

bench_prices = pd.Series(np.exp(np.random.randn(1000).cumsum() * 0.01), index=dates)

# Let's bypass the truncation manually
res = MetricsEngine.calculate_all_indicators(df.tail(350), window=40, benchmark_prices=bench_prices.tail(350), include_metrics=['rel_strength_z'], tail_only=False)

print(res['rel_strength_z'].tail())
