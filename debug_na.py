from src.data import DataManager
from src.config import DEFAULT_FEATURES
from src.metrics import MetricsEngine
from ml_engine.labeling.labeler import TripleBarrierLabeler
import pandas as pd
import numpy as np

manager = DataManager()
engine = MetricsEngine()

df = manager.load_data("BTCUSDT", "1h", auto_sync=False)
df = df.iloc[-2000:]
all_metrics_df = engine.calculate_all_indicators(df, window=20, interval="1h")

valid_features = [f for f in DEFAULT_FEATURES if f != 'rel_strength_z']
feats_data = {feat: all_metrics_df[feat] for feat in valid_features if feat in all_metrics_df.columns}

temp_df = pd.DataFrame(feats_data)
print("len temp_df:", len(temp_df))
