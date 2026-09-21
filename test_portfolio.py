import requests

req = {
    "target_horizon": 20,
    "short_window": 20,
    "long_window": 100,
    "correlation_alpha": 0.4,
    "interval": "1d",
    "estimation_method": "ols",
    "split_ratio": 0.6,
    "symbols": ["BTCUSDT", "ETHUSDT"]
}
res = requests.post("http://127.0.0.1:3000/api/portfolio/estimate", json=req)
print("Status:", res.status_code)
if res.status_code == 200:
    data = res.json()
    print("Assets:", list(data['assets'].keys()))
    btc = data['assets']['BTCUSDT']
    print("BTC Expected Return:", btc['expected_return'])
    print("BTC Validation:", btc['validation']['return'])
    print("BTC Latest Price:", btc['latest_price'])
else:
    print(res.text)
