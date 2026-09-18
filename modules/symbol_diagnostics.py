from shiny import ui, render, reactive
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.colors as pc
from shinywidgets import output_widget, render_widget
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression

from src.data import DataManager
from src.metrics import MetricsEngine
from src.config import AVAILABLE_INTERVALS, BENCHMARK_SYMBOL, METRIC_LABELS, MANDATORY_CRYPTO, IGNORED_CRYPTO
from src.logger import logger
from src.shared_state import get_manager, get_engine
import requests
from ml_engine.labeling.labeler import Labeler
from ml_engine.analysis.multivariate import DecompositionEngine
from ml_engine.data.bars import construct_volume_bars, construct_dollar_bars, calibrate_bar_threshold

# --- GARCH HELPER ---

from arch import arch_model

def forecast_garch(returns, steps=10, ann_factor=1):
    r = np.asarray(returns)
    model = arch_model(
        r,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        rescale=True
    )

    res = model.fit(disp="off")
    hist_vol = res.conditional_volatility * np.sqrt(ann_factor)
    fc = res.forecast(horizon=steps)
    fc_var = fc.variance.values[-1]
    fc_vol = np.sqrt(fc_var) * np.sqrt(ann_factor)
    return fc_vol, hist_vol

# --- STYLING CONSTANTS ---
THEME_BG = "#0b3d91"
THEME_FONT = "Space Mono"
THEME_GRID = "rgba(255, 255, 255, 0.3)"
THEME_ZERO = "rgba(255, 255, 255, 0.5)"
THEME_TEXT = "white"

def apply_theme(fig):
    fig.update_layout(
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        font=dict(family=THEME_FONT, color=THEME_TEXT),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    fig.update_xaxes(gridcolor=THEME_GRID, zerolinecolor=THEME_ZERO)
    fig.update_yaxes(gridcolor=THEME_GRID, zerolinecolor=THEME_ZERO)
    return fig

def symbol_diagnostics_ui():
    return ui.page_fluid(
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_action_button("btn_run_diag", "Run Diagnostics", class_="btn-primary w-100 mt-2"),
                ui.input_selectize("diag_symbol", "Select Symbol", choices=[], selected="BTCUSDT", multiple=False),
                ui.input_select("diag_interval", "Interval", choices=AVAILABLE_INTERVALS, selected="1h"),
                ui.input_numeric("metric_window", "Metrics Window", value=10, min=20, max=500),
                ui.input_numeric("diag_window", "Analysis Window", value=100, min=20, max=500),
            ),
            
            ui.navset_card_underline(
                ui.nav_panel(
                    "Standard Analysis",
                    ui.div(
                        # Hidden marker
                        ui.div(ui.output_text("diag_ready"), class_="d-none"),
                        
                        ui.panel_conditional(
                            "input.btn_run_diag > 0",
                            
                            # 1. Performance Overview
                            ui.card(
                                ui.card_header("Performance Overview"),
                                ui.div(
                                    ui.value_box("CVaR", ui.output_text("val_cvar"), theme="primary", class_="small-box"),
                                    ui.value_box("Volatility", ui.output_text("val_vol"), theme="blue", class_="small-box"),
                                    ui.value_box("Omega Ratio", ui.output_text("val_omega"), theme="red", class_="small-box"),
                                    ui.value_box("Avg Drawdown", ui.output_text("val_avgdd"), theme="orange", class_="small-box"),
                                    ui.value_box("Beta", ui.output_text("val_beta"), theme="info", class_="small-box"),
                                    ui.value_box("Alpha", ui.output_text("val_alpha"), theme="success", class_="small-box"),
                                    ui.value_box("Impact Spread", ui.output_text("val_impact_spread"), theme="warning", class_="small-box"),
                                    ui.value_box("OBook Imbalance", ui.output_text("val_imbalance"), theme="teal", class_="small-box"),
                                    ui.value_box("ADL Risk", ui.output_text("val_adl_risk"), theme="danger", class_="small-box"),
                                    class_="d-flex flex-row flex-nowrap gap-1 overflow-auto justify-content-between",
                                    style="padding-bottom: 5px;"
                                )                               
                            ),
                            
                            # 2. Metrics & Exposure
                            ui.card(
                                ui.card_header("Market-Neutral Analysis"),
                                ui.layout_columns(
                                    ui.div(
                                        output_widget("plot_metrics")
                                    ),
                                    ui.div(
                                        output_widget("plot_mn_cum_ret")
                                    ),
                                    col_widths=[6, 6]
                                )
                            ),
                            
                            # 3. Trading Statistics (Binance)
                            ui.card(
                                ui.card_header("Trading Statistics (Binance Futures)"),
                                ui.layout_columns(
                                    ui.div(output_widget("plot_ts_funding")),
                                    ui.div(output_widget("plot_ts_oi")),
                                    col_widths=[6, 6]
                                ),
                                ui.layout_columns(
                                    ui.div(output_widget("plot_ts_top_pos")),
                                    ui.div(output_widget("plot_ts_top_acc")),
                                    col_widths=[6, 6]
                                ),
                                ui.layout_columns(
                                    ui.div(output_widget("plot_ts_glob_acc")),
                                    ui.div(output_widget("plot_ts_taker")),
                                    col_widths=[6, 6]
                                ),
                            ),
                            
                            # 4. Forecast
                            ui.card(
                                ui.card_header("Forecast (Forward 10 periods)"),
                                ui.layout_columns(
                                    ui.div(
                                        output_widget("plot_forecast_price")
                                    ),
                                    ui.div(
                                        output_widget("plot_forecast_vol")
                                    ),
                                    col_widths=[6, 6]
                                )
                            ),
                            
                            # # 4. Cointegration
                            # ui.card(
                            #     ui.card_header("Cointegration & Correlation"),
                            #     output_widget("plot_coint")
                            # )
                        )
                    )
                ),
                ui.nav_panel(
                    "Financial Bars",
                    ui.div(
                        ui.row(
                            ui.column(
                                4,
                                ui.card(
                                    ui.card_header("Structural Engineering"),
                                    ui.input_select("diag_bar_type", "Active Bar Type", choices=["Time Bars", "Volume Bars", "Dollar Bars"], selected="Dollar Bars"),
                                    ui.layout_columns(
                                        ui.input_numeric("diag_vol_th", "Volume Threshold", value=10000, min=1),
                                        ui.input_numeric("diag_dollar_th", "Dollar Threshold", value=1000000, min=1),
                                    ),
                                    ui.layout_columns(
                                        ui.input_action_button("btn_generate_bars", "Generate All", class_="btn-primary w-100"),
                                        ui.input_action_button("btn_diag_auto_calibrate", "Auto Calibrate", class_="btn-primary w-100"),
                                    ),
                                    ui.hr(),
                                    ui.output_table("diag_engineering_stats_table")
                                )
                            ),
                            ui.column(
                                8,
                                ui.card(
                                    ui.card_header("Return Distribution Comparison"),
                                    output_widget("diag_engineering_dist_plot"),
                                    full_screen=True
                                )
                            )
                        ),
                        ui.row(
                            ui.column(
                                12,
                                ui.card(
                                    ui.card_header("Bar OHLCV Viewer"),
                                    output_widget("plot_financial_bars"),
                                    full_screen=True
                                )
                            )
                        )
                    )
                ),
                id="tab_diag"
            )
        )
    )
    
def symbol_diagnostics_server(input, output, session, global_interval):
    manager = get_manager()
    engine = get_engine()
    
    diag_data = reactive.Value({})
    engineering_results_cache = reactive.Value(None)

    data_info = reactive.Value({"global": {"oldest": "-", "latest": "-"},
                                "symbol": {"oldest": "-", "latest": "-"}})
    
    def get_timestamps(symbol, interval):
        # Disable auto_sync for metadata checks to prevent startup data fetching
        df = manager.load_data(symbol, interval, auto_sync=False)
        if df is not None and not df.empty and 'open_time' in df.columns:
            ts = pd.to_datetime(df['open_time'])
            return {"oldest": str(ts.min()), "latest": str(ts.max())}
        return {"oldest": "-", "latest": "-"}
    
    @reactive.Effect
    def populate_symbols():
        all_syms = manager.get_universe()
        ui.update_selectize("diag_symbol", choices=all_syms, selected="BTCUSDT", server=True)
        
        # Set benchmark/global timestamps once
        global_ts = get_timestamps(BENCHMARK_SYMBOL, input.diag_interval())
        data_info.set({"global": global_ts, "symbol": {"oldest": "-", "latest": "-"}})
    
    @reactive.Effect
    @reactive.event(input.diag_symbol)
    def update_symbol_ts():
        if input.diag_symbol():
            symbol_ts = get_timestamps(input.diag_symbol(), input.diag_interval())
            current_global = data_info.get().get("global", {"oldest": "-", "latest": "-"})
            data_info.set({"global": current_global, "symbol": symbol_ts})
    
    @render.ui
    def data_status():
        d = data_info.get()
        return ui.HTML(f"""
            <div style="font-size: 0.7rem; opacity: 0.7; color: white;">
                <div>Global Data: <div>
                <div>{d['global']['oldest']}</div>
                <div>{d['global']['latest']}</div>
                <br>
                <div>Symbol Data: <div>
                <div>{d['symbol']['oldest']}</div>
                <div>{d['symbol']['latest']}</div>
            </div>
        """)

    @reactive.Effect
    @reactive.event(input.btn_run_diag)
    def _():
        symbol = input.diag_symbol()
        interval = input.diag_interval()
        window = input.diag_window()
        metric_window = input.metric_window()
        
        if not symbol:
            ui.notification_show("Please select a symbol", type="warning")
            return

        with ui.Progress(min=0, max=100) as p:
            p.set(10, message="Loading Data...")
            
            # 1. Load Data
            df = manager.load_data(symbol, interval)
            bench_df = manager.load_data(BENCHMARK_SYMBOL, interval)
            
            if df is None or df.empty or len(df) < window:
                ui.notification_show("Insufficient data for analysis", type="error")
                return
            
             # Update Symbol Date Info
            if 'open_time' in df.columns:
                last_ts_sym = pd.to_datetime(df['open_time']).max()
                curr_info = data_info.get()
                curr_info['symbol'] = str(last_ts_sym)
                data_info.set(curr_info)

            # Clean & Prepare
            prices = pd.to_numeric(df['close'], errors='coerce').ffill().values
            log_rets = np.diff(np.log(prices))
            log_rets = np.nan_to_num(log_rets)
            
            # Load Benchmark
            bench_rets = None
            if bench_df is not None and not bench_df.empty:
                b_prices = pd.to_numeric(bench_df['close'], errors='coerce').ffill().values
                bench_rets = np.diff(np.log(b_prices))
                # Align lengths
                min_len = min(len(log_rets), len(bench_rets))
                log_rets_aligned = log_rets[-min_len:]
                bench_rets_aligned = bench_rets[-min_len:]
            
            p.set(30, message="Calculating Performance...")
            
            # 2. Performance Metrics
            res_sharpe = engine.calculate_sharpe_ratio(log_rets, interval=interval)
            res_sortino = engine.calculate_sortino_ratio(log_rets, interval=interval)
            res_maxdd = engine.calculate_max_drawdown(prices)
            res_avgdd = engine.calculate_avg_drawdown(prices)
            
            # CVaR (Conditional Value at Risk) at 95% confidence level
            var_threshold = np.percentile(log_rets, 5)  # 5th percentile (95% confidence)
            cvar = log_rets[log_rets <= var_threshold].mean()
            
            # Volatility (annualized)
            ann_factor = engine.get_annual_scaling(interval)
            volatility = np.std(log_rets) * np.sqrt(ann_factor)
            
            # Omega Ratio (ratio of gains to losses relative to threshold, using 0 as threshold)
            threshold = 0
            gains = log_rets[log_rets > threshold].sum()
            losses = np.abs(log_rets[log_rets <= threshold].sum())
            omega_ratio = gains / losses if losses != 0 else 0
            
            # 3. Metrics Snapshot
            latest_metrics = engine.calculate_all_indicators(
                df.iloc[-window * 2:], 
                benchmark_returns=np.log(bench_df['close']).diff()[-window * 2:],
                interval=interval,
                window=int(metric_window) 
            )
            
            p.set(50, message="Calculating Exposure...")
            
            # 4. Exposure (Beta)
            beta, alpha, r2 = 0, 0, 0
            if bench_rets is not None:
                beta_, alpha_, r2 = engine.calculate_beta_alpha(log_rets_aligned, bench_rets_aligned)
                
            # Factor Decomp - Load ALL symbols for comprehensive analysis
            market_data = {}
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def _load_sym(s_name):
                d = manager.load_data(s_name, interval)
                if d is not None and not d.empty:
                    return s_name, pd.to_numeric(d['close'], errors='coerce').pct_change()
                return s_name, None
                
            syms_to_load = [s for s in MANDATORY_CRYPTO if s not in IGNORED_CRYPTO]
            with ThreadPoolExecutor(max_workers=10) as executor:
                futs = [executor.submit(_load_sym, s) for s in syms_to_load]
                for fut in as_completed(futs):
                    s_name, ret_series = fut.result()
                    if ret_series is not None:
                        market_data[s_name] = ret_series
            
            factor_df = pd.DataFrame(market_data).ffill().fillna(0).tail(window)
            mn_cum_ret = pd.Series(dtype=float)
            
            if factor_df.shape[1] > 2:
                decomp_res = DecompositionEngine.k_factor_decompose(factor_df, k=5)
                sym_series = pd.Series(log_rets[-window:], index=factor_df.index)
                
                # Market Neutral Calculation: Regress sym_series against PC1
                pc1 = decomp_res['factor_returns']['PC1'].values.reshape(-1, 1)
                y = sym_series.values
                
                lr = LinearRegression()
                lr.fit(pc1, y)
                residuals = y - lr.predict(pc1)
                mn_cum_ret = pd.Series(np.cumsum(residuals), index=factor_df.index)
            
            p.set(70, message="Forecasting...")
            
            # --- Timestamp generation for Forecast ---
            try:
                ts_series = pd.to_datetime(df['open_time'])
                history_ts_full = ts_series.dt.strftime('%Y-%m-%d %H:%M').values
                
                # We need history timestamps for the charts
                # Price chart uses prices[-window-10:]
                price_hist_ts = history_ts_full[-window-10:]
                # Vol chart uses window
                vol_hist_ts = history_ts_full[-window:]
                
                # Generate future timestamps
                last_ts = ts_series.iloc[-1]
                unit = interval[-1]
                val = int(interval[:-1])
                unit_map = {'m': 'min', 'h': 'h', 'd': 'd', 'w': 'W'}
                freq = f"{val}{unit_map.get(unit, 'h')}"
                
                forecast_ts = pd.date_range(
                    start=last_ts + pd.Timedelta(freq), 
                    periods=10, 
                    freq=freq
                ).strftime('%Y-%m-%d %H:%M').values
            except Exception as e:
                logger.log("Symbol Diagnostics", "ERROR", f"Timestamp generation failed: {e}")
                price_hist_ts = np.arange(window + 10)
                vol_hist_ts = np.arange(window)
                forecast_ts = np.arange(window + 10, window + 20)

            # 5. Forecast
            try:
                # ARIMA for Price - Using log-prices to effectively model log-returns
                # (2,1,2) on log-price is equivalent to (2,0,2) on log-returns
                history_price = prices[-window - 10:]
                log_history = np.log(history_price)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(log_history, order=(1,1,1))
                    model_fit = model.fit()
                    fc_res = model_fit.get_forecast(steps=10)
                
                # Transform back from log-space to price-space
                fc_mean = np.exp(fc_res.predicted_mean)
                fc_ci = np.exp(fc_res.conf_int(alpha=0.05))
            except Exception as e:
                print(f"ARIMA failed for {symbol}: {str(e)}")
                # Fallback to last price
                last_p = history_price[-1] if len(history_price) > 0 else 0
                fc_mean = np.full(10, last_p)
                # Simple CI 1% fallback
                fc_ci = np.zeros((10, 2))
                fc_ci[:, 0] = last_p * 0.99
                fc_ci[:, 1] = last_p * 1.01

            # GARCH Volatility Forecast
            try:
                garch_data = log_rets[-window:]

                if len(garch_data) < 20:
                    raise ValueError(f"Insufficient data for GARCH: {len(garch_data)} points")

                vol_forecast_vals, hist_vol = forecast_garch(
                    garch_data,
                    steps=10,
                    ann_factor=MetricsEngine.get_annual_scaling(interval)
                )

            except Exception as e:
                logger.log("Symbol Diagnostics", "ERROR", f"GARCH failed for {symbol}: {str(e)}")
                vol_forecast_vals = np.zeros(10)
                hist_vol = np.zeros(len(garch_data))

            p.set(80, message="Regime Classification...")
            
            # 6. Regime (Labeler - Trend)
            # Use smaller window for labeling loop or just label whole series
            l_algo = Labeler(amplitude_threshold=0.01, max_inactive_period=10) # 1% move, 10 bars inactive
            lbl_df = l_algo.label(prices[-window:])
            curr_lbl_val = lbl_df['label'].iloc[-1]
            if curr_lbl_val == 1: curr_regime = "Uptrend"
            elif curr_lbl_val == -1: curr_regime = "Downtrend"
            else: curr_regime = "Sideways/Neutral"
            
            p.set(90, message="Relationships...")
            
            # Relationships (Deprecated)
            corrs = pd.Series(dtype=float)
            coint_scores = []
            zscore_spreads = []

            col_droped = ["volatility"]
            latest_metrics = latest_metrics.drop(columns=col_droped)
            latest_metrics = latest_metrics.dropna(axis=1, how="all")
            latest_metrics = latest_metrics.dropna(axis=0, how="all")
            
            def z_score(x):
                return (x - x.mean()) / x.std()

            for col in latest_metrics.columns:
                latest_metrics[col] = z_score(latest_metrics[col])

            # 7. Orderbook Status
            p.set(95, message="Fetching Orderbook...")
            try:
                book_status = manager.fetcher.get_books_status(symbol)
            except Exception as e:
                logger.log("Symbol Diagnostics", "ERROR", f"Orderbook fetch failed: {e}")
                book_status = {}

            # 8. Trading Statistics (Binance)
            p.set(98, message="Fetching Trading Stats...")
            period_mapping = {
                '1m': '5m', '3m': '5m', '5m': '5m', '15m': '15m',
                '30m': '30m', '1h': '1h', '2h': '2h', '4h': '4h',
                '6h': '6h', '8h': '6h', '12h': '12h', '1d': '1d',
                '3d': '1d', '1w': '1d', '1M': '1d'
            }
            period = period_mapping.get(interval, '1h')
            limit = min(window, 500)
            
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=6) as executor:
                f_oi = executor.submit(manager.fetcher.get_historical_stats_series, symbol, period, manager.fetcher.OPEN_INTEREST_HIST, limit=limit)
                f_pos = executor.submit(manager.fetcher.get_historical_stats_series, symbol, period, manager.fetcher.TOP_LS_POSITION, limit=limit)
                f_acc = executor.submit(manager.fetcher.get_historical_stats_series, symbol, period, manager.fetcher.TOP_LS_ACCOUNT, limit=limit)
                f_g_acc = executor.submit(manager.fetcher.get_historical_stats_series, symbol, period, manager.fetcher.GLOBAL_LS_ACCOUNT, limit=limit)
                f_taker = executor.submit(manager.fetcher.get_historical_stats_series, symbol, period, manager.fetcher.TAKER_BUY_SELL, limit=limit)
                f_fund = executor.submit(manager.fetcher.get_historical_funding_rate, symbol, limit=limit)
                
                oi_hist = f_oi.result()
                top_pos = f_pos.result()
                top_acc = f_acc.result()
                glob_acc = f_g_acc.result()
                taker_ratio = f_taker.result()
                fund_hist = f_fund.result()
            
            adl_risk_map = manager.fetcher.get_all_adl_risks()
            current_adl_risk = adl_risk_map.get(symbol, 0)

            # Pack Data
            data_pack = {
                "sharpe": res_sharpe,
                "sortino": res_sortino,
                "maxdd": res_maxdd,
                "avgdd": res_avgdd,
                "cvar": cvar,
                "volatility": volatility,
                "omega": omega_ratio,
                "metrics_df": latest_metrics.tail(1).T.reset_index(),
                "beta": beta_,
                "alpha": alpha_,
                "mn_cum_ret": mn_cum_ret if not mn_cum_ret.empty else pd.Series(dtype=float),
                "fc_price": {
                    "hist": history_price, 
                    "fc": fc_mean, 
                    "ci": fc_ci,
                    "hist_ts": price_hist_ts,
                    "fc_ts": forecast_ts
                },
                "fc_vol": {
                    "hist": hist_vol, 
                    "fc": vol_forecast_vals,
                    "hist_ts": vol_hist_ts,
                    "fc_ts": forecast_ts
                },
                "regime": {"status": curr_regime, "labels": lbl_df['label'].values, "prices": lbl_df['price'].values},
                "impact_spread": book_status.get("impact_spread"),
                "imbalance": book_status.get("orderbook_imbalance"),
                "adl_risk": current_adl_risk,
                "ts_oi": oi_hist,
                "ts_top_pos": top_pos,
                "ts_top_acc": top_acc,
                "ts_glob_acc": glob_acc,
                "ts_taker": taker_ratio,
                "ts_fund": fund_hist
            }
            
            diag_data.set(data_pack)
            p.set(100, message="Complete")

    # ----- RENDERERS -----
    
    @render.text
    def diag_ready():
        return "true" if diag_data.get() else "false"

    @render.text
    def val_sharpe(): return f"{diag_data.get().get('sharpe', 0):.2f}" if diag_data.get() else "-"
    @render.text
    def val_sortino(): return f"{diag_data.get().get('sortino', 0):.2f}" if diag_data.get() else "-"
    @render.text
    def val_maxdd(): return f"{diag_data.get().get('maxdd', 0)*100:.2f}%" if diag_data.get() else "-"
    @render.text
    def val_avgdd(): return f"{diag_data.get().get('avgdd', 0)*100:.2f}%" if diag_data.get() else "-"
    @render.text
    def val_winrate(): return f"{diag_data.get().get('winrate', 0)*100:.1f}%" if diag_data.get() else "-"

    @render.text
    def val_beta():
        d = diag_data.get()
        return f"{d.get('beta', 0):.3f}" if d else "-"
    
    @render.text
    def val_alpha():
        d = diag_data.get()
        return f"{d.get('alpha', 0):.4f}" if d else "-"
    
    @render.text
    def val_cvar():
        d = diag_data.get()
        return f"{d.get('cvar', 0)*100:.2f}%" if d else "-"
    
    @render.text
    def val_vol():
        d = diag_data.get()
        return f"{d.get('volatility', 0)*100:.2f}%" if d else "-"
    
    @render.text
    def val_omega():
        d = diag_data.get()
        return f"{d.get('omega', 0):.2f}" if d else "-"

    @render.text
    def val_impact_spread():
        d = diag_data.get()
        v = d.get('impact_spread')
        return f"{v*100:.4f}%" if v is not None else "-"

    @render.text
    def val_imbalance():
        d = diag_data.get()
        v = d.get('imbalance')
        return f"{v*100:.2f}%" if v is not None else "-"

    @render.text
    def val_adl_risk():
        d = diag_data.get()
        return f"{d.get('adl_risk', 0)}" if d else "-"

    def _make_ts_fig(title, x, y, color):
        """Helper: build a single trading-stat line chart."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, name=title, line=dict(color=color)))
        fig.update_layout(title=title, height=280, showlegend=False)
        return apply_theme(fig)

    @render_widget
    def plot_ts_funding():
        d = diag_data.get()
        if not d or not d.get("ts_fund"): return go.Figure()
        df = pd.DataFrame(d["ts_fund"])
        if df.empty or 'fundingTime' not in df.columns: return go.Figure()
        df['time'] = pd.to_datetime(df['fundingTime'], unit='ms')
        return _make_ts_fig("Funding Rate", df['time'], pd.to_numeric(df['fundingRate'], errors='coerce'), 'cyan')

    @render_widget
    def plot_ts_oi():
        d = diag_data.get()
        if not d or not d.get("ts_oi"): return go.Figure()
        df = pd.DataFrame(d["ts_oi"])
        if df.empty or 'timestamp' not in df.columns: return go.Figure()
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        oi = pd.to_numeric(df['sumOpenInterest'], errors='coerce')
        circ = pd.to_numeric(df['CMCCirculatingSupply'], errors='coerce').replace(0, np.nan)
        return _make_ts_fig("OI / Circulating", df['time'], oi / circ, 'orange')

    @render_widget
    def plot_ts_top_pos():
        d = diag_data.get()
        if not d or not d.get("ts_top_pos"): return go.Figure()
        df = pd.DataFrame(d["ts_top_pos"])
        if df.empty or 'timestamp' not in df.columns: return go.Figure()
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        return _make_ts_fig("Top Trader L/S (Positions)", df['time'], pd.to_numeric(df['longShortRatio'], errors='coerce'), 'lime')

    @render_widget
    def plot_ts_top_acc():
        d = diag_data.get()
        if not d or not d.get("ts_top_acc"): return go.Figure()
        df = pd.DataFrame(d["ts_top_acc"])
        if df.empty or 'timestamp' not in df.columns: return go.Figure()
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        return _make_ts_fig("Top Trader L/S (Accounts)", df['time'], pd.to_numeric(df['longShortRatio'], errors='coerce'), 'lime')

    @render_widget
    def plot_ts_glob_acc():
        d = diag_data.get()
        if not d or not d.get("ts_glob_acc"): return go.Figure()
        df = pd.DataFrame(d["ts_glob_acc"])
        if df.empty or 'timestamp' not in df.columns: return go.Figure()
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        return _make_ts_fig("Global L/S Account", df['time'], pd.to_numeric(df['longShortRatio'], errors='coerce'), 'magenta')

    @render_widget
    def plot_ts_taker():
        d = diag_data.get()
        if not d or not d.get("ts_taker"): return go.Figure()
        df = pd.DataFrame(d["ts_taker"])
        if df.empty or 'timestamp' not in df.columns: return go.Figure()
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        col = 'buySellRatio' if 'buySellRatio' in df.columns else 'longShortRatio'
        return _make_ts_fig("Taker Buy/Sell Ratio", df['time'], pd.to_numeric(df[col], errors='coerce'), 'yellow')

    @render_widget
    def plot_metrics():
        d = diag_data.get()
        if not d or d['metrics_df'].empty: return None
        
        df = d['metrics_df']
        if len(df.columns) < 2: return None

        df.columns = ["Metric", "Value"]
        # Filter numeric only
        df = df[pd.to_numeric(df['Value'], errors='coerce').notnull()].head(10)
        df['Value'] = pd.to_numeric(df['Value'])
        df['Metric'] = df['Metric'].map(METRIC_LABELS)
        
        # Radial bar chart (polar)
        fig = go.Figure(go.Barpolar(
            r=df['Value'],
            theta=df['Metric'],
            marker=dict(
                color=df['Value'],
                colorscale='Spectral_r',
                showscale=True,
                colorbar=dict(
                    title="Norm Value"
                )
            ),
            opacity=0.8
        ))

        
        fig.update_layout(
            polar=dict(
                bgcolor=THEME_BG,
                radialaxis=dict(visible=True, gridcolor=THEME_GRID),
                angularaxis=dict(gridcolor=THEME_GRID)
            ),
            paper_bgcolor=THEME_BG,
            plot_bgcolor=THEME_BG,
            font=dict(family=THEME_FONT, color=THEME_TEXT),
            showlegend=False,
            height=450,
            width=700
        )
        return apply_theme(fig)

    @render_widget
    def plot_mn_cum_ret():
        d = diag_data.get()
        if not d or d['mn_cum_ret'].empty: return None
        
        series = d['mn_cum_ret']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(len(series)),
            y=series.values,
            mode='lines',
            line=dict(color='cyan', width=2),
            name='MN Cum Ret'
        ))
        fig.update_layout(
            # title="Market-Neutral Cumulative Return (Ex-PC1)",
            xaxis_title="Periods",
            yaxis_title="Cumulative Idiosyncratic Return",
            height=450, width=700)
        return apply_theme(fig)

    @render_widget
    def plot_forecast_price():
        d = diag_data.get()
        if not d: return go.Figure()
        
        hist = d['fc_price']['hist']
        fc = d['fc_price']['fc']
        ci = d['fc_price']['ci']
        
        x_hist = d['fc_price'].get('hist_ts', np.arange(len(hist)))
        x_fc = d['fc_price'].get('fc_ts', np.arange(len(hist), len(hist) + len(fc)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_hist, y=hist, name="History", line=dict(color="cyan")))
        fig.add_trace(go.Scatter(x=x_fc, y=fc, name="Forecast", line=dict(color="lime", dash="dot", width=1)))
        
        # CI
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_fc, x_fc[::-1]]), # This will work fine with strings
            y=np.concatenate([ci[:,1], ci[:,0][::-1]]),
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='(95%) CI'
        ))
        
        fig.update_layout(
            template="plotly_dark", legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5), height=350, width=700)
        return apply_theme(fig)

    @render_widget
    def plot_forecast_vol():
        d = diag_data.get()
        if not d: return go.Figure()
        
        hist = d['fc_vol']['hist']
        fc = d['fc_vol']['fc']

        diff = hist[-1] - fc[0]
        
        x_hist = d['fc_vol'].get('hist_ts', np.arange(len(hist)))
        x_fc = d['fc_vol'].get('fc_ts', np.arange(len(hist), len(hist) + len(fc)))
        fc = fc + diff

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_hist, y=hist, name="Volatility (GARCH)", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=x_fc, y=fc, name="Forecast", line=dict(color="red", dash="dot", width=1)))
        
        # Add CI if available (±1 std)
        fc_ci = np.full(len(fc), np.std(hist))

        if fc_ci is not None:
            steps = np.arange(1, len(fc) + 1)

            k = 0.1
            scale = k * np.sqrt(steps)

            upper = fc + (fc_ci - fc) * scale
            lower = fc - (fc_ci - fc) * scale

            fig.add_trace(go.Scatter(
                x=np.concatenate([x_fc, x_fc[::-1]]), # This will work fine with strings
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 99, 71, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='(95%) CI'
            ))

        fig.update_layout(
            template="plotly_dark", legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5), height=350, width=700)
        return apply_theme(fig)


    @reactive.calc
    @reactive.event(input.btn_generate_bars)
    def engineering_result():
        symbol = input.diag_symbol()
        interval = input.diag_interval()
        
        if not symbol:
            return None

        with ui.Progress(min=0, max=100) as p:
            p.set(10, message="Loading Data...")
            df = manager.load_data(symbol, interval)
            if df is None or df.empty: return None
            
            if 'open_time' in df.columns:
                df = df.set_index(pd.to_datetime(df['open_time']))
            
            p.set(30, message="Generating Time Bars...")
            time_df = df.copy()
            time_df['ret'] = np.log(time_df['close'] / time_df['close'].shift(1))
            
            p.set(50, message="Generating Volume Bars...")
            vol_df = construct_volume_bars(df, input.diag_vol_th())
            if not vol_df.empty:
                vol_df['ret'] = np.log(vol_df['close'] / vol_df['close'].shift(1))
                
            p.set(70, message="Generating Dollar Bars...")
            dollar_df = construct_dollar_bars(df, input.diag_dollar_th())
            if not dollar_df.empty:
                dollar_df['ret'] = np.log(dollar_df['close'] / dollar_df['close'].shift(1))
            
            p.set(100, message="Complete")
            
            res = {
                "time": time_df,
                "volume": vol_df,
                "dollar": dollar_df,
                "ticker": symbol
            }
            engineering_results_cache.set(res)
            return res

    @render_widget
    def diag_engineering_dist_plot():
        res = engineering_result()
        if res is None: return None

        datasets = []
        labels = []
        
        for k in ["time", "volume", "dollar"]:
            df = res.get(k)
            if df is not None and not df.empty and 'ret' in df.columns:
                rets = df['ret'].dropna()
                if len(rets) > 10:
                    datasets.append(rets.values)
                    labels.append(k.capitalize() + " Bars")
                    
        if not datasets: return None

        colors = pc.diverging.Spectral_r[:len(datasets)]

        fig = ff.create_distplot(
            datasets,
            labels,
            bin_size=.001,
            show_hist=True,
            show_curve=True,
            colors=colors
        )

        fig.update_layout(
            template="plotly_dark",
            height=460,
            width=1000,
            margin=dict(t=40, b=60, l=30, r=30),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            xaxis_title="Log Returns",
            yaxis_title="Density"
        )
        return apply_theme(fig)

    @render.table
    def diag_engineering_stats_table():
        res = engineering_result()
        if res is None: return None
        
        stats_list = []
        for k in ["time", "volume", "dollar"]:
            df = res.get(k)
            if df is not None and not df.empty:
                rets = df['ret'].dropna()
                stats_list.append({
                    "Bar Type": k.capitalize(),
                    "Count": len(df),
                    "Skew": skew(rets),
                    "Kurtosis": kurtosis(rets, fisher=True)
                })
        
        df_stats = pd.DataFrame(stats_list)
        return (
            df_stats.style
            .hide(axis="index")
            .format({"Skew": "{:.4f}", "Kurtosis": "{:.4f}"})
            .set_properties(**{"font-family": "'Space Mono', monospace", "text-align": "center"})
            .set_table_styles([
                {"selector": "th", "props": [("color", "black"), ("background-color", "#FCC780"), ("font-weight", "bold")]},
                {"selector": "td", "props": [("border", "1px solid #1a4da3")]}
            ])
        )

    @reactive.Effect
    @reactive.event(input.btn_diag_auto_calibrate)
    async def _auto_calibrate_diag():
        symbol = input.diag_symbol()
        interval = input.diag_interval()
        
        if not symbol:
            ui.notification_show("Please select a symbol", type="warning")
            return

        with ui.Progress(min=0, max=100) as p:
            p.set(20, message="Loading Data...")
            df = manager.load_data(symbol, interval)
            if df is None or df.empty:
                ui.notification_show("No data available", type="error")
                return
            
            if 'open_time' in df.columns:
                df = df.set_index(pd.to_datetime(df['open_time']))

            p.set(40, message="Calibrating Volume...")
            opt_vol = calibrate_bar_threshold(df, "Volume Bars")
            if opt_vol:
                ui.update_numeric("diag_vol_th", value=int(opt_vol))
                
            p.set(70, message="Calibrating Dollar...")
            opt_dollar = calibrate_bar_threshold(df, "Dollar Bars")
            if opt_dollar:
                ui.update_numeric("diag_dollar_th", value=int(opt_dollar))
            
            p.set(100, message="Complete")
            ui.notification_show(f"✓ Calibrated! Vol: {opt_vol:,}, Dollar: {opt_dollar:,}", type="message")

    @render_widget
    def plot_financial_bars():
        res = engineering_results_cache.get()
        if res is None: return None
        
        b_type = input.diag_bar_type()
        target = "time" if b_type == "Time Bars" else ("volume" if b_type == "Volume Bars" else "dollar")
        df = res.get(target)
        
        if df is None or df.empty: return None

        df.index = pd.to_datetime(df.index)
        df.index = df.index.strftime("%Y-%m-%d %H:%M")
        
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC",
            increasing_line_color="lightgray",
            decreasing_line_color="#ff4b4b"
        )])
        
        fig.update_layout(
            showlegend=False,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.12,
                xanchor="center",
                x=0.5
            ),
            template="plotly_dark",
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            height=600,
            width=1500,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        fig.update_xaxes(rangeslider_visible=False, type='category')
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.15)", zeroline=False)
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.15)", zeroline=True,
                        zerolinecolor="rgba(255,255,255,0.2)")

        fig = apply_theme(fig)
        fig.update_layout(
            margin=dict(l=50, r=100, t=50, b=50)
        )
        return fig
