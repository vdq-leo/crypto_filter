"""
Application Configuration
Update this file to change titles, descriptions, symbols, and links.
"""


# --- APP SETTINGS ---
APP_TITLE = "Crypto Market Radar"
APP_ICON = "LEO"
APP_LAYOUT = "wide"
THEME = "quartz"
BG_COLOR = "#1a1a1a"

API_BASE_URL = "http://127.0.0.1:3000/api"

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
    'vama': 'VAMA',
    'liquidity_impact': 'Liquidity Impact',
    'vol_rank': 'Volatility Rank',
    'mr_prob': 'Mean Reversion Probability',
    'max_drawdown': 'Max Drawdown',
    'avg_drawdown': 'Average Drawdown',
    'breakout_score_dist': 'Breakout Score v1',
    'breakout_score_break': 'Breakout Score v2',
    'orderbook_imbalance': 'OBook Imbalance',
    'spread': 'Spread',
    'funding_rate': 'Funding Rate',
    'oi_circulating': 'OI / Circulating Supply',
    'top_ls_position_ratio': 'Top Trader L/S (Position)',
    'top_ls_account_ratio': 'Top Trader L/S (Account)',
    'global_ls_ratio': 'Global L/S Account Ratio',
    'taker_buysell_ratio': 'Taker Buy/Sell Volume Ratio',
    'adl_risk': 'ADL Risk Rating',
    'None': 'None'
}

# List of all available numeric metrics for axes
ALL_METRICS = [
    'rel_strength_z',
    'price_zscore', 'vam',
    'volatility','vol_imbalance', 'fip', 
    'adf_hist', 'adf_stat', 'ewva', 'aroon_osc', 
    'rsi_norm', 'atr_norm', 'cmf', 'vwap_z', 'skewness',
    'autocorr_1', 'autocorr_5', 'imbalance_bar', 'volume_imbalance', 'vol_atr', 'vama', 'liquidity_impact', 'vol_rank', 'mr_prob',
    'max_drawdown', 'avg_drawdown',
    'breakout_score_dist', 'breakout_score_break',
    # 'sign_lag1', 'sign_lag2', 'sign_lag3', 'rolling_sign_lag5', 'rolling_sign_lag10', 'rolling_sign_lag20',
    'funding_rate','oi_circulating', 'top_ls_position_ratio', 'top_ls_account_ratio', 'global_ls_ratio', 'taker_buysell_ratio', 'adl_risk'
]

DEFAULT_FEATURES = [
    'rel_strength_z',
    'fip', 
    'adf_hist', 'ewva', 'aroon_osc', 
    'atr_norm', 'vol_atr', 'vama', 'cmf', 'vam', 'skewness',
    'sign_lag1', 'sign_lag2', 'sign_lag3',
    'autocorr_1', 'autocorr_5', 
    'imbalance_bar', 'vol_imbalance', 'volume_imbalance', 'liquidity_impact', 'vol_rank', 'mr_prob',
    'max_drawdown', 'avg_drawdown',
    'breakout_score_dist', 'breakout_score_break',
    'funding_rate',
    'oi_circulating', 'top_ls_position_ratio', 'top_ls_account_ratio', 'global_ls_ratio', 'taker_buysell_ratio', 'adl_risk'
]
