import pandas as pd
import glob
spy = pd.read_parquet(glob.glob("data_cache/SPYUSDT_1d.parquet")[0])
btc = pd.read_parquet(glob.glob("data_cache/BTCUSDT_1d.parquet")[0])

print("SPY head dates:")
print(spy['open_time'].head())

print("BTC head dates:")
print(btc['open_time'].head())
