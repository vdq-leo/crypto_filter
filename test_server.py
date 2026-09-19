from api.routers.predictive import run_engineering, EngineeringRequest
req = EngineeringRequest(
    ticker="BTCUSDT",
    interval="1h",
    vol_th=16382,
    dollar_th=936585120
)
res = run_engineering(req)
