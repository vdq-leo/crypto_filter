import requests
import json
from src.config import DEFAULT_FEATURES

req_data = {
    "ticker": "BTCUSDT",
    "interval": "1h",
    "trade_direction": "Long/Short Combine",
    "features": DEFAULT_FEATURES,
    "reg_type": "RF Classifier",
    "test_ratio": 0.2,
    "rf_max_depth": 5,
    "eng_lookback": 20,
    "pred_lookback": 10000,
    "standardize": True,
    "vif_th": 10.0,
    "bar_type": "Time Bars",
    "labeler_type": "BoxRange",
    "labeler_params": {
        "amp_th": 0.05,
        "max_inactive": 10,
        "vol_window": 20,
        "upper_mult": 2.0,
        "lower_mult": 2.0,
        "max_holding": 10
    }
}

try:
    res = requests.post("http://127.0.0.1:3000/api/predictive/analyze", json=req_data)
    print("STATUS:", res.status_code)
    if res.status_code != 200:
        print("BODY:", res.text[:500])
    else:
        print("SUCCESS! Keys returned:", list(res.json().keys()))
except Exception as e:
    print("ERROR:", e)
