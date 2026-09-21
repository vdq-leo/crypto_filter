import numpy as np
import pandas as pd
import riskfolio as rp

# Dummy returns
ret = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])

port = rp.HCPortfolio(returns=ret)
cov = ret.cov()
# Is it possible to run without monkey patch?
try:
    w = port.optimization(model='HERC', rm='MV', rf=0, method_cov='custom_cov', custom_cov=cov)
    print("Success without patch!")
    print(w)
except Exception as e:
    print("Failed:", e)
