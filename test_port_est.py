import pandas as pd
import numpy as np

# Load local cache data
import glob
spy_f = glob.glob("data_cache/SPYUSDT_1d.parquet")
btc_f = glob.glob("data_cache/BTCUSDT_1d.parquet")
print(spy_f)

if spy_f and btc_f:
    spy = pd.read_parquet(spy_f[0])
    btc = pd.read_parquet(btc_f[0])
    
    spy_p = pd.to_numeric(spy['close'], errors='coerce').ffill().dropna()
    spy_p.index = spy.loc[spy_p.index, 'open_time']
    spy_p = spy_p[~spy_p.index.duplicated(keep='last')]
    
    btc_p = pd.to_numeric(btc['close'], errors='coerce').ffill().dropna()
    btc_p.index = btc.loc[btc_p.index, 'open_time']
    btc_p = btc_p[~btc_p.index.duplicated(keep='last')]
    
    df = pd.DataFrame({'SPY': spy_p, 'BTC': btc_p}).sort_index().ffill().dropna(how='all')
    
    ret = df.pct_change().dropna(how='all')
    print("SPY n_obs:", len(ret['SPY'].dropna()))
    print("SPY vol (sqrt 365):", ret['SPY'].dropna().std() * np.sqrt(365))
