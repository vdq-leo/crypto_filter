import requests
import json

req = {
    "ticker": "BTCUSDT",
    "interval": "1h",
    "vol_th": 16382,
    "dollar_th": 936585120
}

try:
    res = requests.post("http://127.0.0.1:3000/api/predictive/engineering", json=req)
    print("STATUS:", res.status_code)
    print("BODY:", res.text[:500])
except Exception as e:
    print("ERR:", e)
