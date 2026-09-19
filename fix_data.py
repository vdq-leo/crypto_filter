import pandas as pd
import glob
import os

files = glob.glob("/Users/leoinv/.gemini/antigravity-ide/crypto_filter_data/*.parquet")
for f in files:
    try:
        df = pd.read_parquet(f)
        if df.duplicated(subset=['open_time']).any():
            print(f"Fixing duplicates in {f}")
            df = df.drop_duplicates(subset=['open_time'], keep='last')
            df.to_parquet(f, index=False)
    except Exception as e:
        print(f"Error on {f}: {e}")
