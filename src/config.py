"""
Application Configuration
Update this file to change titles, descriptions, symbols, and links.
"""


import os

# --- APP SETTINGS ---
APP_TITLE = "Crypto Market Radar"
APP_ICON = "LEO"
APP_LAYOUT = "wide"
THEME = "quartz"
BG_COLOR = "#1a1a1a"

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:3000/api")

# --- TIMEZONE SETTING ---
# Global offset from UTC in hours (e.g., 7 for UTC+7, -5 for EST)
TIMEZONE_OFFSET = 7

# --- SIDEBAR & WELCOME ---
SIDEBAR_INFO = "Select a module to proceed"
WELCOME_TITLE = "Crypto Market Radar"
WELCOME_TEXT = """
### Welcome to the Advanced Metrics Dashboard

This application allows you to analyze Binance Perpetual Futures with advanced statistical metrics.

#### Modules:
1. **Data Loader**: Fetch and cache historical market data. 
2. **Market Radar**: Visualize and explore metrics.
3. **Regression**: Analyze predictive power of indicators.
"""

# --- DATA SETTINGS ---
# Symbols that are always fetched
MANDATORY_CRYPTO = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'AAVEUSDT', 'LINKUSDT', 'QQQUSDT', 'SPYUSDT', 'XAUUSDT', 'XAGUSDT']
IGNORED_CRYPTO = []

# Benchmark used for Relative Strength and Beta
BENCHMARK_SYMBOL = 'BTCUSDT'

# Default timeframes available in Data Loader
AVAILABLE_INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
DEFAULT_FETCH_INTERVALS = ['1h', '4h', '1d']

# --- EXTERNAL LINKS ---
TRADINGVIEW_URL = "https://www.tradingview.com/chart/"

# --- METRIC LABELS ---
METRIC_LABELS = {
    'volatility': 'Volatility',
    'adf_hist': 'ADF Regime',
    'adf_stat': 'ADF Statistic',
    'fip': 'FIP',
    'count': 'Data Points',
    'ewva': 'EWVA',
    'aroon_osc': 'Aroon Oscillator',
    'rsi_norm': 'RSI (Normalized)',
    'atr_norm': 'Normalized ATR',
    'cmf': 'Chaikin Money Flow',
    'vwap_z': 'VWAP Z-Score',
    'rel_strength_z': 'RS Z-Score',
    'calmar_ratio': 'Calmar Ratio',
    'beta_btc': 'BTC Beta',
    'price_zscore': 'Price Z-Score',
    'vam': 'VAM (Vol-Adj Momentum)',
    'skewness': 'Return Skewness',
    'sign_lag1': 'Sign Lag 1',
    'sign_lag2': 'Sign Lag 2',
    'sign_lag3': 'Sign Lag 3',
    'rolling_sign_lag5': 'Rolling Sign (5)',
    'rolling_sign_lag10': 'Rolling Sign (10)',
    'rolling_sign_lag20': 'Rolling Sign (20)',
    'autocorr_1': 'Autocorr (1)',
    'autocorr_5': 'Autocorr (5)',
    'imbalance_bar': 'Imbalance Bar',
    'vol_imbalance': 'Volatility Imbalance',
    'volume_imbalance': 'Volume Imbalance',
    'vol_atr': 'Vol/ATR Ratio',
    'price_vama': 'Price / VAMA',
    'vama': 'Price / VAMA',
    'liquidity_impact': 'Liquidity Impact',
    'vol_rank': 'Volatility Rank',
    'mr_prob': 'Mean Reversion Probability',
    'price_z': 'Price Z-Score (MR)',
    'adf_prob': 'ADF Probability',
    'div_prob': 'Divergence Probability',
    'slope_prob': 'Slope Probability',
    'vol_prob': 'Volatility Probability',
    'vol_ratio': 'ATR Volatility Ratio',
    'max_drawdown': 'Max Drawdown',
    'avg_drawdown': 'Average Drawdown',
    'breakout_score_dist': 'Breakout Score v1',
    'breakout_score_break': 'Breakout Score v2',
    'orderbook_imbalance': 'Orderbook Imbalance',
    'impact_spread': 'Orderbook Impact Spread',
    'spread': 'Spread',
    'funding_rate': 'Live Funding Rate',
    'oi_circulating': 'OI / Circulating Supply',
    'top_ls_position_ratio': 'Top Trader L/S (Position)',
    'top_ls_account_ratio': 'Top Trader L/S (Account)',
    'global_ls_ratio': 'Global L/S Account Ratio',
    'taker_buysell_ratio': 'Taker Buy/Sell Volume Ratio',
    'adl_risk': 'ADL Risk Rating',
    'chg_24h': '24h Chg (%)',
    'raw_value': 'Raw Value (Latest)',
    'None': 'None'
}

# Supported Data Sources for Market Radar Axes
RADAR_SOURCES = {
    'ohlcv': 'Price (OHLCV)',
    'oi_usd': 'Open Interest (USD)',
    'oi_coins': 'Open Interest (Coins)',
    'top_ls_pos': 'Top Trader L/S (Pos)',
    'top_ls_acc': 'Top Trader L/S (Acc)',
    'global_ls': 'Global L/S Ratio',
    'taker_bs': 'Taker Buy/Sell Ratio',
    'funding': 'Funding Rate',
    'snapshot': 'Snapshot / Orderbook',
}

SNAPSHOT_METRICS = [
    'chg_24h',
    'orderbook_imbalance',
    'impact_spread',
    'adl_risk',
    'funding_rate',
    'oi_circulating',
]

# Per-metric parameter specifications (single source of truth for UI & backend)
METRIC_SPECS = {
    # Snapshot / Direct Microstructure Metrics (no TS source or interval needed)
    'orderbook_imbalance': {
        'label': 'Orderbook Imbalance',
        'category': 'snapshot',
        'params': [
            {'name': 'deviation_pct', 'label': 'Mid Deviation (%)', 'default': 2.0, 'min': 0.1, 'max': 50.0, 'step': 0.5},
        ]
    },
    'impact_spread': {
        'label': 'Orderbook Impact Spread',
        'category': 'snapshot',
        'params': [
            {'name': 'impact_usd', 'label': 'Size ($)', 'default': 1000, 'min': 100, 'max': 1000000, 'step': 500},
        ]
    },
    'adl_risk': {'label': 'ADL Risk Rating', 'category': 'snapshot', 'params': []},
    'chg_24h': {'label': '24h Chg (%)', 'category': 'snapshot', 'params': []},
    'funding_rate': {'label': 'Live Funding Rate', 'category': 'snapshot', 'params': []},
    'oi_circulating': {'label': 'OI / Circulating Supply', 'category': 'snapshot', 'params': []},

    # Time-Series Metrics
    'raw_value': {'label': 'Raw Value (Latest)', 'category': 'ts', 'params': []},
    'price_zscore': {
        'label': 'Z-Score', 'category': 'ts',
        'params': [{'name': 'window', 'label': 'Window', 'default': 40, 'min': 5, 'max': 500, 'step': 5}]
    },
    'vam': {
        'label': 'Momentum (VAM)', 'category': 'ts',
        'params': [{'name': 'window', 'label': 'Window', 'default': 40, 'min': 5, 'max': 500, 'step': 5}]
    },
    'ewva': {
        'label': 'Trend Strength (EWMA)', 'category': 'ts',
        'params': [
            {'name': 'w_short', 'label': 'Fast EMA', 'default': 20, 'min': 2, 'max': 200, 'step': 2},
            {'name': 'w_slow', 'label': 'Slow EMA', 'default': 120, 'min': 10, 'max': 500, 'step': 5},
            {'name': 'window', 'label': 'Std Span', 'default': 40, 'min': 5, 'max': 300, 'step': 5},
        ]
    },
    'price_vama': {
        'label': 'Price / VAMA Ratio', 'category': 'ts',
        'params': [
            {'name': 'maMin', 'label': 'Min MA', 'default': 50, 'min': 10, 'max': 200, 'step': 5},
            {'name': 'period', 'label': 'Vol Period', 'default': 37, 'min': 5, 'max': 200, 'step': 1},
            {'name': 'volFactor', 'label': 'Vol Factor', 'default': 0.17, 'min': 0.01, 'max': 2.0, 'step': 0.01},
        ]
    },
    'rsi_norm': {
        'label': 'RSI (Normalized)', 'category': 'ts',
        'params': [{'name': 'window', 'label': 'RSI Period', 'default': 14, 'min': 2, 'max': 200, 'step': 1}]
    },
    'aroon_osc': {
        'label': 'Aroon Oscillator', 'category': 'ts',
        'params': [{'name': 'window', 'label': 'Aroon Period', 'default': 25, 'min': 5, 'max': 200, 'step': 1}]
    },
    'volatility': {
        'label': 'Volatility (Realized)', 'category': 'ts',
        'params': [{'name': 'window', 'label': 'Window', 'default': 40, 'min': 5, 'max': 500, 'step': 5}]
    },
    'atr_norm': {
        'label': 'Normalized ATR', 'category': 'ts',
        'params': [{'name': 'window', 'label': 'ATR Period', 'default': 14, 'min': 2, 'max': 200, 'step': 1}]
    },
    'adf_stat': {
        'label': 'ADF Stationarity', 'category': 'ts',
        'params': [{'name': 'window', 'label': 'Lookback', 'default': 40, 'min': 10, 'max': 300, 'step': 5}]
    },
    'mr_prob': {
        'label': 'Mean Reversion Prob', 'category': 'ts',
        'params': [{'name': 'window', 'label': 'Window', 'default': 40, 'min': 10, 'max': 300, 'step': 5}]
    },
    'breakout_score_dist': {
        'label': 'Breakout Score', 'category': 'ts',
        'params': [{'name': 'window', 'label': 'Window', 'default': 40, 'min': 10, 'max': 300, 'step': 5}]
    },
    'skewness': {
        'label': 'Return Skewness', 'category': 'ts',
        'params': [{'name': 'window', 'label': 'Window', 'default': 40, 'min': 10, 'max': 500, 'step': 5}]
    },
    'autocorr_1': {
        'label': 'Autocorrelation (Lag 1)', 'category': 'ts',
        'params': [{'name': 'window', 'label': 'Window', 'default': 40, 'min': 10, 'max': 500, 'step': 5}]
    },
    'fip': {
        'label': 'FIP (Frog in the Pan)', 'category': 'ts',
        'params': [{'name': 'window', 'label': 'Window', 'default': 40, 'min': 5, 'max': 500, 'step': 5}]
    },
    'max_drawdown': {
        'label': 'Max Drawdown', 'category': 'ts',
        'params': [{'name': 'window', 'label': 'Window', 'default': 40, 'min': 5, 'max': 500, 'step': 5}]
    },
    'calmar_ratio': {
        'label': 'Calmar Ratio', 'category': 'ts',
        'params': [{'name': 'window', 'label': 'Window', 'default': 40, 'min': 5, 'max': 500, 'step': 5}]
    },
    'rel_strength_z': {
        'label': 'RS Z-Score vs BTC', 'category': 'ts', 'ohlcv_only': True,
        'params': [{'name': 'window', 'label': 'Window', 'default': 40, 'min': 10, 'max': 300, 'step': 5}]
    },
    'beta_btc': {
        'label': 'BTC Beta', 'category': 'ts', 'ohlcv_only': True,
        'params': [{'name': 'window', 'label': 'Window', 'default': 40, 'min': 10, 'max': 300, 'step': 5}]
    },
    'cmf': {
        'label': 'Chaikin Money Flow', 'category': 'ts', 'ohlcv_only': True,
        'params': [{'name': 'window', 'label': 'Window', 'default': 20, 'min': 5, 'max': 200, 'step': 5}]
    },
    'vwap_z': {
        'label': 'VWAP Z-Score', 'category': 'ts', 'ohlcv_only': True,
        'params': [{'name': 'window', 'label': 'Window', 'default': 40, 'min': 5, 'max': 300, 'step': 5}]
    },
    'volume_imbalance': {
        'label': 'Volume Imbalance', 'category': 'ts', 'ohlcv_only': True,
        'params': [{'name': 'window', 'label': 'Window', 'default': 40, 'min': 5, 'max': 300, 'step': 5}]
    },
    'liquidity_impact': {
        'label': 'Liquidity Impact', 'category': 'ts', 'ohlcv_only': True,
        'params': [{'name': 'window', 'label': 'Window', 'default': 40, 'min': 5, 'max': 300, 'step': 5}]
    },
}

# Curated, non-redundant metrics for Market Radar
RADAR_METRICS = [
    'raw_value',
    'chg_24h',
    'price_zscore',
    'vam',
    'ewva',
    'price_vama',
    'rsi_norm',
    'aroon_osc',
    'volatility',
    'atr_norm',
    'adf_stat',
    'mr_prob',
    'breakout_score_dist',
    'skewness',
    'autocorr_1',
    'fip',
    'max_drawdown',
    'calmar_ratio',
    'rel_strength_z',
    'beta_btc',
    'cmf',
    'vwap_z',
    'volume_imbalance',
    'liquidity_impact',
]

# List of all available numeric metrics for axes
ALL_METRICS = [
    'rel_strength_z', 'calmar_ratio', 'beta_btc',
    'price_zscore', 'vam',
    'volatility','vol_imbalance', 'fip', 
    'adf_hist', 'adf_stat', 'ewva', 'aroon_osc', 
    'rsi_norm', 'atr_norm', 'cmf', 'vwap_z', 'skewness',
    'autocorr_1', 'autocorr_5', 'imbalance_bar', 'volume_imbalance', 'vol_atr', 
    'price_vama',
    'liquidity_impact', 'vol_rank', 
    'mr_prob', 'price_z', 'adf_prob', 'div_prob', 'slope_prob', 'vol_prob', 'vol_ratio',
    'max_drawdown', 'avg_drawdown',
    'breakout_score_dist', 'breakout_score_break',
    # 'sign_lag1', 'sign_lag2', 'sign_lag3', 'rolling_sign_lag5', 'rolling_sign_lag10', 'rolling_sign_lag20',
    'funding_rate','oi_circulating', 'top_ls_position_ratio', 'top_ls_account_ratio', 'global_ls_ratio', 'taker_buysell_ratio', 'adl_risk'
]

DEFAULT_FEATURES = [
    'rel_strength_z',
    'fip', 
    'adf_hist', 'ewva', 'aroon_osc', 
    'atr_norm', 'vol_atr', 'price_vama', 'cmf', 'vam', 'skewness',
    'sign_lag1', 'sign_lag2', 'sign_lag3',
    'autocorr_1', 'autocorr_5', 
    'imbalance_bar', 'vol_imbalance', 'volume_imbalance', 'liquidity_impact', 'vol_rank', 'mr_prob',
    'max_drawdown', 'avg_drawdown',
    'breakout_score_dist', 'breakout_score_break',
    'funding_rate',
    'oi_circulating', 'top_ls_position_ratio', 'top_ls_account_ratio', 'global_ls_ratio', 'taker_buysell_ratio', 'adl_risk'
]
