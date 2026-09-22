import requests

try:
    r = requests.get("http://127.0.0.1:3000/api/data/universe?top_n=50&bottom=true")
    print("bottom=true:", r.status_code)
    print(r.json())
except Exception as e:
    print(e)
