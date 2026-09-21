import pandas as pd
import numpy as np
import riskfolio as rp

np.random.seed(42)
returns_df = pd.DataFrame(np.random.randn(100, 4) * 0.02, columns=['A', 'B', 'C', 'D'])

port = rp.HCPortfolio(returns=returns_df)
port.w_max = 0.4 # Sum is 1.6 > 1.0

try:
    w2 = port.optimization(model='HERC', rm='MV', rf=0)
    print("\nWith w_max=0.4:")
    print(w2)
except Exception as e:
    import traceback
    traceback.print_exc()

