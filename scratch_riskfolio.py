import numpy as np
import pandas as pd
import riskfolio as rp

original_hrb = rp.HCPortfolio._hierarchical_recursive_bisection
def patched_hrb(self, Z, rm="MV", rf=0, linkage=None, model="HERC", upper_bound=None, lower_bound=None):
    return original_hrb(self, Z, rm=rm, rf=rf, model=model)
rp.HCPortfolio._hierarchical_recursive_bisection = patched_hrb

n = 10
assets = [f'A{i}' for i in range(n)]

np.random.seed(42)
cov = np.random.randn(n, n)
cov = np.dot(cov, cov.T) # make positive semi-definite
vols = np.sqrt(np.diag(cov))
corr = cov / np.outer(vols, vols)

cov_df = pd.DataFrame(cov, index=assets, columns=assets)
corr_df = pd.DataFrame(corr, index=assets, columns=assets)

# Need actual simulated returns to avoid "Encountered all NA values" in gap stat if HERC does clustering
# HERC needs to do clustering which works best on larger return matrices, or we can just pass cov and codep.
returns = pd.DataFrame(np.random.randn(500, n), columns=assets)
port = rp.HCPortfolio(returns=returns)

port.cov = cov_df
port.codep = corr_df

try:
    # Try HERC
    w_herc = port.optimization(model='HERC', rm='MV', rf=0)
    print("HERC Success:\n", w_herc.head())
    
    # Try HRP
    w_hrp = port.optimization(model='HRP', rm='MV', rf=0)
    print("HRP Success:\n", w_hrp.head())
except Exception as e:
    import traceback
    traceback.print_exc()
