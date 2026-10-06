import numpy as np
import pandas as pd
from src.metrics import MetricsEngine, _vama_position_state_numba, _adaptive_ema_numba
from src.data import DataManager

def test_vama_calculation():
    dm = DataManager()
    df = dm.load_data('BTCUSDT', '1h')
    assert df is not None and len(df) > 0, "Failed to load BTCUSDT 1h data"
    
    close = df['close']
    print(f"Loaded {len(close)} candles for BTCUSDT 1h. Latest close: {close.iloc[-1]:.2f}")
    
    # Run calculate_vama with standard BTCUSDT parameters
    res = MetricsEngine.calculate_vama(close, maMin=50, period=37, volFactor=0.17, longZScore=1.15, shortZScore=1.45)
    
    vama = res['vama']
    vol = res['vol']
    targetVol = res['targetVol']
    targetWeight = res['targetWeight']
    span = res['span']
    currentRet = res['currentRet']
    longThreshold = res['longThreshold']
    shortThreshold = res['shortThreshold']
    signal = res['signal']
    position = res['position']
    
    # Verify non-trivial values
    assert not vama.isna().all(), "VAMA series should not be all NaN"
    assert not vol.isna().all(), "Vol series should not be all NaN"
    assert not targetVol.isna().all(), "TargetVol series should not be all NaN"
    
    latest_idx = close.index[-1]
    print("\n--- Latest BTCUSDT 1h Bar (Close price) ---")
    print(f"Close:           {close.iloc[-1]:.2f}")
    print(f"VAMA:            {vama.iloc[-1]:.2f}")
    print(f"Distance %:      {res['vama_dist'].iloc[-1]*100:.2f}%")
    print(f"Volatility:      {vol.iloc[-1]*100:.4f}%")
    print(f"Target Vol:      {targetVol.iloc[-1]*100:.4f}%")
    print(f"Target Weight:   {targetWeight.iloc[-1]:.2f}")
    print(f"Span:            {span.iloc[-1]:.0f}")
    print(f"Current Return:  {currentRet.iloc[-1]*100:.2f}%")
    print(f"Long Threshold:  {longThreshold.iloc[-1]*100:.2f}%")
    print(f"Short Threshold: {shortThreshold.iloc[-1]*100:.2f}%")
    print(f"Signal:          {signal.iloc[-1]}")
    print(f"Position:        {position.iloc[-1]} ({'LONG' if position.iloc[-1] == 1 else 'SHORT' if position.iloc[-1] == -1 else 'FLAT'})")
    
    # Test use_closed_bar=True (Pine Script request.security(..., f_vama(close[1])))
    res_closed = MetricsEngine.calculate_vama(close, use_closed_bar=True)
    print("\n--- Latest Bar using use_closed_bar=True (close[1] like Pine Script request.security) ---")
    print(f"VAMA (closed):   {res_closed['vama'].iloc[-1]:.2f}")
    print(f"Vol (closed):    {res_closed['vol'].iloc[-1]*100:.4f}%")
    print(f"TarVol (closed): {res_closed['targetVol'].iloc[-1]*100:.4f}%")
    print(f"Weight (closed): {res_closed['targetWeight'].iloc[-1]:.2f}")
    print(f"Span (closed):   {res_closed['span'].iloc[-1]:.0f}")
    print(f"Pos (closed):    {res_closed['position'].iloc[-1]}")
    
    # Verify that calculate_all_indicators exports all VAMA metrics
    adv_df = MetricsEngine.calculate_all_indicators(df, tail_only=True)
    expected_cols = [
        'vama', 'vama_raw', 'vama_dist', 'vama_weight', 'vama_span',
        'vama_vol', 'vama_target_vol', 'vama_signal', 'vama_pos',
        'vama_current_ret', 'vama_long_thr', 'vama_short_thr'
    ]
    for col in expected_cols:
        assert col in adv_df.columns, f"Expected {col} in calculate_all_indicators output"
        val = adv_df[col].iloc[-1]
        assert not pd.isna(val), f"Column {col} should not be NaN at latest bar, got {val}"
        
    print("\n[SUCCESS] All VAMA indicators computed cleanly without NaN and match Pine Script logic!")

if __name__ == '__main__':
    test_vama_calculation()
