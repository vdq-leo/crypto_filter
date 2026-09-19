import yfinance as yf
import numpy as np
import pandas as pd

df = yf.Ticker("SPY").history(period="max")
ret = df['Close'].pct_change().dropna()
vol_252 = ret.std() * np.sqrt(252)
print("SPY original annualized vol (sqrt(252)): ", vol_252)

df_ffill = df.resample('D').ffill()
ret_ffill = df_ffill['Close'].pct_change().dropna()
vol_365 = ret_ffill.std() * np.sqrt(365)
print("SPY ffilled annualized vol (sqrt(365)): ", vol_365)
