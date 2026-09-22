import requests
r = requests.get("http://127.0.0.1:3000/api/data/universe")
print(r.status_code)
print(r.json())
