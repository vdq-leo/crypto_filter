import pandas as pd
import numpy as np
import riskfolio as rp

np.random.seed(42)
returns_df = pd.DataFrame(np.random.randn(100, 3) * 0.02, columns=['A', 'B', 'C'])
returns_df.iloc[:20, 2] = np.nan # Add NaNs

try:
    mu_js = rp.mean_vector(returns_df, method='JS')
    print(mu_js)
except Exception as e:
    print("Error:", type(e), e)
