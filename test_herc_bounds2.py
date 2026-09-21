import pandas as pd
import numpy as np
import riskfolio as rp

np.random.seed(42)
returns_df = pd.DataFrame(np.random.randn(100, 3) * 0.02, columns=['A', 'B', 'C'])

port = rp.HCPortfolio(returns=returns_df)
port.w_max = 0.2 # Set as float!

try:
    w2 = port.optimization(model='HERC', rm='MV', rf=0)
    print("\nWith w_max=0.2:")
    print(w2)
except Exception as e:
    print("\nError:", type(e), e)

