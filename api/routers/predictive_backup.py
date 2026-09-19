from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.data import DataManager
from src.metrics import MetricsEngine
from src.shared_state import get_manager, get_engine
from ml_engine.modeling.factory import ModelFactory
from ml_engine.modeling.feature_selection import FeatureSelector
from ml_engine.predictive.predictor import IsotonicCalibrator, CalibratedModelWrapper
from ml_engine.labeling.labeler import Labeler, TripleBarrierLabeler, StationarityLabeler, CombinedLabeler, TailSetLabeler
from ml_engine.data.bars import construct_volume_bars, construct_dollar_bars, calibrate_bar_threshold
from sklearn.linear_model import LogisticRegression

router = APIRouter()
manager = get_manager()
engine = get_engine()

class PredictiveRequest(BaseModel):
    ticker: str = "BTCUSDT"
    interval: str = "1h"
    trade_direction: str = "Both Sides" # Long Only, Short Only, Long/Short Combine
    features: List[str] = ["RSI_14", "MACD_hist", "BB_width"]
    reg_type: str = "RF Classifier" # RF Classifier, XGB Classifier
    test_ratio: float = 0.3
    rf_max_depth: int = 5
    eng_lookback: int = 20
    pred_lookback: int = 5000
    bar_type: str = "Time Bars" # Time Bars, Volume Bars, Dollar Bars
    labeler_type: str = "BoxRange" # Trend, BoxRange, Combine, Regime, TailSet
    labeler_params: Dict[str, Any] = {
        "vol_window": 10,
        "upper_mult": 2,
        "lower_mult": 2,
        "max_holding": 10
    }
    standardize: bool = True
    vif_th: float = 10.0

def force_scalar_numeric(df):
    def flatten_cell(x):
        try:
            if isinstance(x, (list, np.ndarray, tuple)):
                if hasattr(x, '__len__') and len(x) > 0:
                    return flatten_cell(x[0])
                return 0.0
            return float(x)
        except:
            return 0.0
    return df.applymap(flatten_cell).astype(float)

def _sanitize_for_json(data):
    if isinstance(data, pd.DataFrame):
        df = data.replace([np.inf, -np.inf], None).where(pd.notnull(data), None)
        return {"columns": list(df.columns), "index": list(df.index.astype(str)), "data": df.values.tolist()}
    elif isinstance(data, pd.Series):
        s = data.replace([np.inf, -np.inf], None).where(pd.notnull(data), None)
        return {"index": list(s.index.astype(str)), "data": s.values.tolist()}
    elif isinstance(data, np.ndarray):
        d = np.where(np.isfinite(data), data, None)
        return d.tolist()
    elif isinstance(data, dict):
        return {k: _sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_for_json(v) for v in data]
    return data

class CalibrateRequest(BaseModel):
    ticker: str
    interval: str

@router.post("/calibrate")
def calibrate_thresholds(req: CalibrateRequest):
    df = manager.load_data(req.ticker, req.interval, auto_sync=False)
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No data available")
    
    if 'open_time' in df.columns:
        df = df.set_index(pd.to_datetime(df['open_time']))
        
    optimal_vol = calibrate_bar_threshold(df, "Volume Bars")
    optimal_dollar = calibrate_bar_threshold(df, "Dollar Bars")
    
    return {
        "volume": optimal_vol,
        "dollar": optimal_dollar
    }

class EngineeringRequest(BaseModel):
    ticker: str
    interval: str
    vol_th: int
    dollar_th: int

@router.post("/engineering")
def run_engineering(req: EngineeringRequest):
    df = manager.load_data(req.ticker, req.interval, auto_sync=False)
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No data available")
    
    if 'open_time' in df.columns:
        df = df.set_index(pd.to_datetime(df['open_time']))
        
    time_df = df.copy()
    time_df['ret'] = np.log(time_df['close'] / time_df['close'].shift(1))
    
    vol_df = construct_volume_bars(df, req.vol_th)
    if not vol_df.empty:
        vol_df['ret'] = np.log(vol_df['close'] / vol_df['close'].shift(1))
        
    dollar_df = construct_dollar_bars(df, req.dollar_th)
    if not dollar_df.empty:
        dollar_df['ret'] = np.log(dollar_df['close'] / dollar_df['close'].shift(1))
        
    return {
        "time": _sanitize_for_json(time_df),
        "volume": _sanitize_for_json(vol_df),
        "dollar": _sanitize_for_json(dollar_df)
    }

class FeatureAnalysisRequest(BaseModel):
    ticker: str
    interval: str
    features: List[str]
    eng_lookback: int
    eng_min_samples: int
    vif_th: float
    pred_lookback: int
    bar_type: str
    vol_th: Optional[int] = 10000
    dollar_th: Optional[int] = 1000000

@router.post("/features")
def run_feature_analysis(req: FeatureAnalysisRequest):
    # First get the right bar type data
    df = manager.load_data(req.ticker, req.interval, auto_sync=False)
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No data available")
    
    if 'open_time' in df.columns:
        df = df.set_index(pd.to_datetime(df['open_time']))

    if req.bar_type == "Volume Bars":
        df = construct_volume_bars(df, req.vol_th)
    elif req.bar_type == "Dollar Bars":
        df = construct_dollar_bars(df, req.dollar_th)

    if df.empty:
        raise HTTPException(status_code=400, detail="Bar construction failed")

    # Limit lookback
    calc_df = df.copy()
    calc_df = calc_df.iloc[-(req.pred_lookback + req.eng_lookback):]
    if 'dollar_vol' in calc_df.columns and 'volume' not in calc_df.columns:
        calc_df['volume'] = calc_df['dollar_vol'] / calc_df['close']
        
    if len(calc_df) < req.eng_lookback + 10:
        raise HTTPException(status_code=400, detail="Insufficient data for lookback")

    feat_df = engine.calculate_all_indicators(calc_df, window=req.eng_lookback, interval=req.interval)
    valid_feats = [f for f in req.features if f in feat_df.columns]
    if not valid_feats:
        raise HTTPException(status_code=400, detail="None of the selected features could be calculated")
        
    feat_df = feat_df[valid_feats].dropna()
    
    if len(feat_df) < req.eng_min_samples:
        raise HTTPException(status_code=400, detail=f"Only {len(feat_df)} samples remaining. Need {req.eng_min_samples}.")

    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    from scipy.stats import skew, kurtosis

    stats_list = []
    vif_dict = {}
    if len(valid_feats) > 1:
        try:
            X_vif = feat_df.loc[:, feat_df.std() > 0]
            if not X_vif.empty and X_vif.shape[1] > 1:
                X = sm.add_constant(X_vif)
                for i, col in enumerate(X_vif.columns):
                    v = variance_inflation_factor(X.values, i + 1)
                    vif_dict[col] = min(v, 99.0)
        except:
            pass

    for col in valid_feats:
        s = feat_df[col].describe()
        vif = vif_dict.get(col, 1.0)
        stats_list.append({
            "Feature": col,
            "Mean": s['mean'],
            "Std": s['std'],
            "Skew": skew(feat_df[col]),
            "Kurtosis": kurtosis(feat_df[col], fisher=True),
            "VIF": vif,
            "Keep": "YES" if vif <= req.vif_th else "NO"
        })
        
    stats_df = pd.DataFrame(stats_list)
    clean_feats = [s['Feature'] for s in stats_list if s['VIF'] <= req.vif_th]
    feat_df_clean = feat_df[clean_feats] if clean_feats else pd.DataFrame()

    return {
        "feat_df": _sanitize_for_json(feat_df),
        "feat_df_clean": _sanitize_for_json(feat_df_clean),
        "stats_df": stats_df.to_dict(orient="records"),
        "bar_type": req.bar_type,
        "vif_threshold": req.vif_th
    }

@router.post("/analyze")
def run_predictive_analysis(req: PredictiveRequest):
    try:
        df = manager.load_data(req.ticker, req.interval, auto_sync=False)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No data available")

        if 'open_time' in df.columns:
            df = df.set_index(pd.to_datetime(df['open_time']))

        # Use limited lookback
        df = df.iloc[-(req.pred_lookback + req.eng_lookback):]

        # Bars Construction
        if req.bar_type == "Volume Bars":
            vol_th = calibrate_bar_threshold(df, "Volume Bars")
            if vol_th is None: vol_th = 50000
            df = construct_volume_bars(df, vol_th)
        elif req.bar_type == "Dollar Bars":
            dollar_th = calibrate_bar_threshold(df, "Dollar Bars")
            if dollar_th is None: dollar_th = 1000000
            df = construct_dollar_bars(df, dollar_th)

        if len(df) < req.eng_lookback + 10:
            raise HTTPException(status_code=400, detail="Not enough bars after construction")

        # Feature calculation
        all_metrics_df = engine.calculate_all_indicators(df, window=req.eng_lookback, interval=req.interval)
        feats_data = {feat: all_metrics_df[feat] for feat in req.features if feat in all_metrics_df.columns}

        # Labeling
        prices = df['close']
        lp = req.labeler_params
        if req.labeler_type == "Trend":
            lobj = Labeler(amplitude_threshold=lp.get('amp_th', 100), max_inactive_period=lp.get('max_inactive', 10))
            y_series = lobj.label(prices)['label']
        elif req.labeler_type == "BoxRange":
            lobj = TripleBarrierLabeler(vol_window=lp.get('vol_window', 10), upper_mult=lp.get('upper_mult', 2), lower_mult=lp.get('lower_mult', 2), max_holding_period=lp.get('max_holding', 10))
            y_series = lobj.label(prices)['label']
        elif req.labeler_type == "Regime":
            lobj = StationarityLabeler(window=lp.get('window', 20), vote_th=lp.get('vote_th', 0.5))
            y_series, _ = lobj.label(prices)
        else:
            lobj = TailSetLabeler(ret_window=lp.get('window', 1), threshold=lp.get('threshold', 1.0))
            y_series = lobj.label(prices)['label']

        y_series.index = prices.index
        y_series = y_series.shift(-1) # Shift 1 for next bar

        if req.trade_direction == "Long Only":
            y_series = y_series.where(y_series > 0, 0)
        elif req.trade_direction == "Short Only":
            y_series = y_series.where(y_series < 0, 0)

        temp_df = pd.DataFrame(feats_data)
        temp_df['Target_Y'] = y_series
        temp_df['raw_return'] = np.log(prices.ffill() / prices.ffill().shift(1)).shift(-1)
        temp_df = temp_df.dropna()

        if temp_df.empty:
            raise HTTPException(status_code=400, detail="Empty dataset after cleaning")

        X_data = temp_df[req.features].copy()
        Y_data = temp_df['Target_Y']
        X_data = force_scalar_numeric(X_data)

        if req.standardize:
            for f in req.features:
                if f in X_data.columns:
                    m, s = X_data[f].mean(), X_data[f].std()
                    if s > 1e-9: X_data[f] = (X_data[f] - m) / s

        active_features = FeatureSelector.apply_vif_filter(X_data, threshold=req.vif_th) if len(req.features) > 1 else req.features
        
        if not active_features:
            raise HTTPException(status_code=400, detail="No features left after VIF filtering")

        X_raw = X_data[active_features].copy()
        
        # Split
        n = len(X_raw)
        test_start = int(n * (1 - req.test_ratio))
        train_n = test_start
        meta_start = int(train_n * 0.6)

        X_train_primary = X_raw.iloc[:meta_start]
        y_train_primary = Y_data.iloc[:meta_start]

        X_train_meta = X_raw.iloc[meta_start:test_start]
        y_train_meta = Y_data.iloc[meta_start:test_start]

        X_test = X_raw.iloc[test_start:]
        y_test = Y_data.iloc[test_start:]

        # Train primary
        model = ModelFactory.create_model(req.reg_type, n_estimators=100, max_depth=req.rf_max_depth)
        model.fit(X_train_primary, y_train_primary)
        
        # Train meta (simplified - logistic regression over primary predictions)
        p_train_meta_input = model.predict(X_train_meta)
        p_train_meta_signals = pd.Series([sig for sig in p_train_meta_input], index=X_train_meta.index)
        y_meta_target = (p_train_meta_signals == y_train_meta).astype(int)
        
        meta_features = pd.concat([X_train_meta, pd.Series(p_train_meta_signals, index=X_train_meta.index, name='primary_pred')], axis=1)
        meta_model = LogisticRegression(class_weight='balanced')
        meta_model.fit(meta_features, y_meta_target)

        # Test Evaluation
        from sklearn.metrics import accuracy_score, classification_report
        p_test_input = model.predict(X_test)
        acc = accuracy_score(y_test, p_test_input)
        report = classification_report(y_test, p_test_input, output_dict=True, zero_division=0)
        
        return {
            "features_used": active_features,
            "train_size": len(X_train_primary),
            "meta_size": len(X_train_meta),
            "test_size": len(X_test),
            "accuracy": acc,
            "classification_report": report
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
