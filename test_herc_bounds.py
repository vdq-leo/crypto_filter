import pandas as pd
import numpy as np
import riskfolio as rp

np.random.seed(42)
returns_df = pd.DataFrame(np.random.randn(100, 3) * 0.02, columns=['A', 'B', 'C'])

# HERC optimizer
port = rp.HCPortfolio(returns=returns_df)
# Set bounds?
port.w_max = np.array([0.2, 0.2, 0.2]) # Wait, if it's an array
try:
    w = port.optimization(model='HERC', rm='MV', rf=0)
    print("Default HERC:")
    print(w)
except Exception as e:
    print(e)
    
try:
    port.upper_bound = 0.2
    w2 = port.optimization(model='HERC', rm='MV', rf=0)
    print("\nWith upper_bound=0.2 on port:")
    print(w2)
except Exception as e:
    print("upper_bound fail:", e)

# Let's check optimization signature again
print("\nOptimization docs:")
print(rp.HCPortfolio.optimization.__doc__)

