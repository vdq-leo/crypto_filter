import pandas as pd
import numpy as np
import riskfolio as rp

np.random.seed(42)
# Dummy returns (e.g. daily)
returns_df = pd.DataFrame(np.random.randn(100, 3) * 0.02 + 0.001, columns=['A', 'B', 'C'])
# Add extreme outlier
returns_df['C'] += 0.01 

mu_js = rp.mean_vector(returns_df, method='JS')
mu_hist = rp.mean_vector(returns_df, method='hist')

print("Historical Mean:\n", mu_hist)
print("\nJames-Stein:\n", mu_js)
