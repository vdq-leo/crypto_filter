from src.data import DataManager
manager = DataManager()
df = manager.load_data("BTCUSDT", "1h", auto_sync=False)
print("TOTAL ROWS IN DB for BTCUSDT:", len(df))
