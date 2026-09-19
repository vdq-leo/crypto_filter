import json
from api.routers.predictive import run_engineering, EngineeringRequest
from pydantic import BaseModel

req = EngineeringRequest(
    ticker="BTCUSDT",
    interval="1h",
    vol_th=16382,
    dollar_th=936585120
)

try:
    res = run_engineering(req)
    print("Function executed successfully.")
    print("Testing json dumps...")
    json.dumps(res, allow_nan=False)
    print("JSON dumps successful.")
except Exception as e:
    print("ERROR:", type(e), e)
