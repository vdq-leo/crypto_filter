from src.data.manager import DataManager
from src.config import DEFAULT_FEATURES
from ml_engine.analysis.metrics import MetricsEngine

manager = DataManager()
engine = MetricsEngine()

df = manager.load_data("BTCUSDT", "1h", auto_sync=False)
df = df.iloc[-2000:]
all_metrics_df = engine.calculate_all_indicators(df, window=20, interval="1h")

feats_data = {feat: all_metrics_df[feat] for feat in DEFAULT_FEATURES if feat in all_metrics_df.columns}

for k, v in feats_data.items():
    print(k, "NaNs:", v.isna().sum(), "Total:", len(v))
