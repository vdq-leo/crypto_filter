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

from src.config import AVAILABLE_INTERVALS, BENCHMARK_SYMBOL, METRIC_LABELS, MANDATORY_CRYPTO, IGNORED_CRYPTO, API_BASE_URL
from src.logger import logger
import requests

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
    
    diag_data = reactive.Value({})
    engineering_results_cache = reactive.Value(None)

    data_info = reactive.Value({"global": {"oldest": "-", "latest": "-"},
                                "symbol": {"oldest": "-", "latest": "-"}})
    
    def get_timestamps(symbol, interval):
        try:
            res = requests.get(f"{API_BASE_URL}/data/klines", params={"symbol": symbol, "interval": interval, "limit": 10000})
            if res.status_code == 200:
                data = res.json().get("candles", [])
                if data:
                    oldest = pd.to_datetime(data[0]["time"], unit='s')
                    latest = pd.to_datetime(data[-1]["time"], unit='s')
                    return {"oldest": str(oldest), "latest": str(latest)}
        except:
            pass
        return {"oldest": "-", "latest": "-"}
    
    @reactive.Effect
    def populate_symbols():
        try:
            res = requests.get(f"{API_BASE_URL}/data/universe")
            all_syms = res.json()["symbols"] if res.status_code == 200 else []
        except:
            all_syms = []
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
            
            # 1. Fetch Diagnostics Data
            try:
                payload = {
                    "symbol": symbol,
                    "interval": interval,
                    "metric_window": int(metric_window),
                    "diag_window": int(window)
                }
                res = requests.post(f"{API_BASE_URL}/diagnostics/run", json=payload)
                if res.status_code == 200:
                    api_data = res.json()
                    
                    # Transform back to data_pack format
                    perf = api_data.get("performance", {})
                    charts = api_data.get("charts", {})
                    
                    data_pack = {
                        "sharpe": perf.get("sharpe", 0),
                        "sortino": perf.get("sortino", 0),
                        "maxdd": perf.get("maxdd", 0),
                        "avgdd": perf.get("avgdd", 0),
                        "cvar": perf.get("cvar", 0),
                        "volatility": perf.get("volatility", 0),
                        "omega": perf.get("omega", 0),
                        "beta": perf.get("beta", 0),
                        "alpha": perf.get("alpha", 0),
                        "impact_spread": perf.get("impact_spread", 0),
                        "imbalance": perf.get("imbalance", 0),
                        "adl_risk": perf.get("adl_risk", 0),
                        "metrics_df": pd.DataFrame(charts.get("metrics", [])),
                        "mn_cum_ret": pd.Series(charts.get("mn_cum_ret", [])),
                        "fc_price": {
                            "hist": charts.get("prices", {}).get("hist", []),
                            "fc": charts.get("prices", {}).get("forecast", []),
                            "ci": np.column_stack((charts.get("prices", {}).get("ci_lower", []), charts.get("prices", {}).get("ci_upper", []))),
                        },
                        "fc_vol": {
                            "hist": charts.get("volatility", {}).get("hist", []),
                            "fc": charts.get("volatility", {}).get("forecast", [])
                        },
                        "regime": charts.get("regime", {}),
                        "ts_oi": charts.get("ts_oi", []),
                        "ts_top_pos": charts.get("ts_top_pos", []),
                        "ts_top_acc": charts.get("ts_top_acc", []),
                        "ts_glob_acc": charts.get("ts_glob_acc", []),
                        "ts_taker": charts.get("ts_taker", []),
                        "ts_fund": charts.get("ts_fund", [])
                    }
                    diag_data.set(data_pack)
                    p.set(100, message="Complete")
                else:
                    ui.notification_show(f"Calculation error: {res.text}", type="error")
            except Exception as e:
                logger.log("Symbol Diagnostics", "ERROR", f"Diagnostics Run failed: {e}")
                ui.notification_show(f"API Connection error: {str(e)}", type="error")

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
            p.set(50, message="Generating Bars from API...")
            
            payload = {
                "symbol": symbol,
                "interval": interval,
                "vol_th": input.diag_vol_th(),
                "dollar_th": input.diag_dollar_th()
            }
            try:
                res_api = requests.post(f"{API_BASE_URL}/diagnostics/bars", json=payload)
                if res_api.status_code == 200:
                    data = res_api.json()
                    
                    def to_df(bar_data):
                        if not bar_data or "data" not in bar_data: return pd.DataFrame()
                        df = pd.DataFrame(bar_data["data"])
                        if not df.empty and 'index' in df.columns:
                            df = df.set_index('index')
                        return df
                    
                    time_df = to_df(data.get("time"))
                    vol_df = to_df(data.get("volume"))
                    dollar_df = to_df(data.get("dollar"))
                    
                    p.set(100, message="Complete")
                    
                    res = {
                        "time": time_df,
                        "volume": vol_df,
                        "dollar": dollar_df,
                        "ticker": symbol,
                        "stats": {
                            "time": data.get("time", {}).get("stats", {}),
                            "volume": data.get("volume", {}).get("stats", {}),
                            "dollar": data.get("dollar", {}).get("stats", {})
                        }
                    }
                    engineering_results_cache.set(res)
                    return res
                else:
                    ui.notification_show(f"Calculation error: {res_api.text}", type="error")
                    return None
            except Exception as e:
                logger.log("Symbol Diagnostics", "ERROR", f"Bars generation failed: {e}")
                ui.notification_show(f"API Connection error: {str(e)}", type="error")
                return None

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
        stats_dict = res.get("stats", {})
        for k in ["time", "volume", "dollar"]:
            s = stats_dict.get(k, {})
            if s and s.get("count", 0) > 0:
                stats_list.append({
                    "Bar Type": k.capitalize(),
                    "Count": s.get("count", 0),
                    "Skew": s.get("skew", 0),
                    "Kurtosis": s.get("kurtosis", 0)
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
            p.set(50, message="Calibrating...")
            
            payload = {
                "symbol": symbol,
                "interval": interval
            }
            try:
                res = requests.post(f"{API_BASE_URL}/diagnostics/calibrate-bars", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    opt_vol = data.get("vol_th")
                    opt_dollar = data.get("dollar_th")
                    
                    if opt_vol:
                        ui.update_numeric("diag_vol_th", value=int(opt_vol))
                    if opt_dollar:
                        ui.update_numeric("diag_dollar_th", value=int(opt_dollar))
                    
                    p.set(100, message="Complete")
                    ui.notification_show(f"✓ Calibrated! Vol: {opt_vol:,}, Dollar: {opt_dollar:,}", type="message")
                else:
                    ui.notification_show(f"Calibration error: {res.text}", type="error")
            except Exception as e:
                logger.log("Symbol Diagnostics", "ERROR", f"Auto calibrate failed: {e}")
                ui.notification_show(f"API Connection error: {str(e)}", type="error")

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
