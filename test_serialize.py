import json
import math
import pandas as pd
import numpy as np

def _sanitize_for_json(data):
    if isinstance(data, pd.DataFrame):
        d = json.loads(data.to_json(orient='split', date_format='iso'))
        return _sanitize_for_json(d)
    elif isinstance(data, pd.Series):
        d = json.loads(data.to_json(orient='split', date_format='iso'))
        return _sanitize_for_json(d)
    elif isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.number):
            data = np.where(np.isfinite(data), data, None)
        return _sanitize_for_json(data.tolist())
    elif isinstance(data, dict):
        return {k: _sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_for_json(v) for v in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    return data

df = pd.DataFrame({'a': [1.0, np.nan], 'b': ['x', float('nan')]}, dtype=object)
out = _sanitize_for_json(df)
try:
    json.dumps(out, allow_nan=False)
    print("SUCCESS")
except Exception as e:
    print("FAILED:", e)
