from shiny import ui, render, reactive
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from src.data import DataManager
from statsmodels.tsa.stattools import coint, adfuller
from scipy import stats
from scipy.stats import gaussian_kde, rankdata
from src.metrics import MetricsEngine, copula_cond_probs
from src.config import AVAILABLE_INTERVALS, API_BASE_URL
from src.logger import logger
from src.shared_state import get_manager, get_engine
import requests

def _sanitize(data):
    """Replace inf/nan with 0 to prevent Plotly JSON serialization errors."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.replace([np.inf, -np.inf], np.nan).fillna(0)
    elif isinstance(data, np.ndarray):
        d = np.where(np.isfinite(data), data, 0)
        return d
    return data

def calculate_rolling_zscore(series, window):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.ewm(window).std().replace(0, 1e-9)
    return (series - rolling_mean) / rolling_std

def pair_radar_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Pair Analysis"),
            ui.input_action_button("btn_gen_pair", "Generate Radar", class_="btn-primary w-100 mt-3"),
            ui.input_select("pair_interval", "Timeframe", choices=AVAILABLE_INTERVALS, selected="1h"),
            ui.input_selectize("symbol_a", "Asset A", choices=[]),
            ui.input_selectize("symbol_b", "Asset B", choices=[]),
            ui.input_select("pair_mode", "Generation Mode", 
                           choices={"ratio": "Ratio", "spread": "Spread"}, selected="spread"),
            ui.hr(),
            ui.input_numeric("rolling_window", "Rolling Window", value=320, min=5),
            ui.input_numeric("pair_window", "Window", value=2000, min=30),
        ),
        ui.card(
            ui.card_header("Pair Analysis Diagnostics"),
            ui.output_ui("pair_metrics_viz"),
            min_height="50px",
            max_height="150px"
        ),
        ui.navset_card_pill(
            ui.nav_panel("Dashboard",
                ui.card(
                    ui.card_header("Statistical Analytics Dashboard"),
                    output_widget("pair_main_chart"),
                    full_screen=True
                ),
            ),
            ui.nav_panel("Copula Analysis",
                ui.row(
                    ui.column(
                        3,
                        ui.card(
                            ui.card_header("Setup"),
                            ui.input_selectize(
                                "copula_mode",
                                "Copula Data Mode",
                                choices={
                                    "price": "Normal Price",
                                    "log_rets": "Log Returns"
                                },
                                selected="price",
                                multiple=False
                            ),
                            ui.panel_conditional(
                                "input.copula_mode === 'log_rets'",
                                ui.input_numeric(
                                    "r_window",
                                    "Group Return Window",
                                    value=10,
                                    min=1
                                )
                            ),
                            ui.panel_conditional(
                                "input.copula_mode === 'price'",
                                ui.input_switch(
                                    "copula_stationarize",
                                    "Stationarize",
                                    value=False
                                ),
                                ui.panel_conditional(
                                    "input.copula_stationarize",
                                    ui.input_numeric(
                                        "copula_ema_window",
                                        "Smoothing Window",
                                        value=20,
                                        min=2
                                    )
                                )
                            ),
                            ui.hr(),
                            ui.input_select("copula_type", "Copula Type", 
                                          choices={"gaussian": "Gaussian", "t": "Student-t", 
                                                   "clayton": "Clayton", "gumbel": "Gumbel"}, 
                                          selected="t"),
                            ui.input_numeric("copula_param", "Copula Parameter (df/theta)", value=2.0, min=0.1, step=0.1),
                            full_screen=False
                        ),
                        ui.card(
                            ui.card_header("Conditional Probabilities"),
                            ui.output_ui("copula_prob_info")
                        )
                    ),
                    ui.column(
                        9,
                        ui.card(
                            ui.card_header("Copula Dependency"),
                            output_widget("pair_copula_chart"),
                            full_screen=True
                        )
                    )
                )
            ),
            ui.nav_panel("Asset Comparison",
                ui.card(
                    ui.card_header(
                        "Price Comparison",
                        ui.input_radio_buttons(
                            "comp_price_mode",
                            None,
                            choices={"vol": "Vol Scale", "stat": "Stationary"},
                            selected="vol",
                            inline=True
                        ),
                        class_="d-flex justify-content-between align-items-center"
                    ),
                    output_widget("pair_comp_chart"),
                    full_screen=True
                )
            ),
            id="pair_tabs"
        )
    )

def pair_radar_server(input, output, session, global_interval):
    manager = get_manager()
    pair_data = reactive.Value(pd.DataFrame())
    metrics_res = reactive.Value({})
    
    # Manual debounce for copula inputs
    copula_trigger = reactive.Value(0)
    
    @reactive.effect
    @reactive.event(input.copula_ema_window, input.r_window, input.copula_stationarize, input.copula_mode, input.copula_type, input.copula_param)
    async def _debounce_copula():
        import asyncio
        await asyncio.sleep(0.5)
        copula_trigger.set(copula_trigger.get() + 1)

    @reactive.effect
    def populate_selectors():
        with ui.Progress(min=0, max=1) as p:
            p.set(0, message="Synchronizing Universe...")
            all_syms = manager.get_universe()
            p.set(1, message="Updating Pair Selectors...")
            ui.update_selectize("symbol_a", choices=all_syms, selected="BTCUSDT")
            ui.update_selectize("symbol_b", choices=all_syms, selected="ETHUSDT")
            ui.update_select("pair_interval", choices=AVAILABLE_INTERVALS, selected=global_interval.get())

    @reactive.effect
    @reactive.event(input.r_window)
    def _update_symbol_choices():
        # Preserving selection while refreshing universe choices
        if not input.r_window():
            ui.update_numeric("r_window", value=1)
        else:
            ui.update_numeric("r_window", value=input.r_window())
        
    @reactive.Effect
    @reactive.event(input.btn_gen_pair)
    def _generate_radar():
        sym_a = input.symbol_a()
        sym_b = input.symbol_b()
        interval = input.pair_interval()
        mode = input.pair_mode()
        window = input.rolling_window()

        if not sym_a or not sym_b:
            return

        with ui.Progress(min=0, max=100) as p:
            p.set(20, message="Fetching pair data from API...")
            
            payload = {
                "symbol_a": sym_a,
                "symbol_b": sym_b,
                "interval": interval,
                "mode": mode,
                "rolling_window": window,
                "pair_window": input.pair_window(),
                "copula_mode": input.copula_mode(),
                "copula_type": input.copula_type(),
                "copula_param": input.copula_param(),
                "r_window": int(input.r_window() or 1),
                "copula_stationarize": input.copula_stationarize(),
                "copula_ema_window": input.copula_ema_window()
            }
            
            try:
                res = requests.post(f"{API_BASE_URL}/pair_radar/generate", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    
                    chart_data = data.get("chart_data", [])
                    if chart_data:
                        synthetic = pd.DataFrame(chart_data)
                        if "open_time" in synthetic.columns:
                            synthetic["open_time"] = pd.to_datetime(synthetic["open_time"])
                            synthetic = synthetic.set_index("open_time")
                        
                        pair_data.set(synthetic)
                    
                    metrics_res.set(data.get("metrics", {}))
                    
                    p.set(100, message="Complete")
                else:
                    ui.notification_show(f"Calculation error: {res.text}", type="error")
            except Exception as e:
                logger.log("Pair Radar", "ERROR", f"API Error: {e}")
                ui.notification_show(f"API Connection error: {str(e)}", type="error")

    @render_widget
    def pair_main_chart():
        df = pair_data.get()
        if df.empty:
            return None

        df.index = pd.to_datetime(df.index)
        df.index = df.index.strftime("%Y-%m-%d %H:%M")
        window = input.rolling_window()

        df = df.tail(input.pair_window() + window)


        df = df.tail(input.pair_window())

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.6, 0.2, 0.2]
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Pair Price",
            increasing_line_color="lightgray",
            decreasing_line_color="#ff4b4b"
        ), row=1, col=1)

        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['ema'],
            name="Basis",
            line=dict(color="rgba(255,0,0,0.8)", width=2),
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['bb_upper'],
            name="BB Upper",
            line=dict(color="rgba(0,255,255,0.5)", width=1),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['bb_lower'],
            name="BB Lower",
            line=dict(color="rgba(0,255,255,0.5)", width=1),
            fill='tonexty',
            fillcolor="rgba(0,255,255,0.1)"
        ), row=1, col=1)

        # Z-score
        z = _sanitize(df['zscore'])
        fig.add_trace(go.Scatter(
            x=df.index,
            y=z,
            name="Z-score",
            line=dict(color="#00ffff", width=1.5),
            fill='tozeroy',
            fillcolor="rgba(0,255,255,0.1)"
        ), row=2, col=1)

        fig.add_hline(y=2.0, line_dash="dash", line_color="rgba(255,255,255,0.4)", row=2, col=1)
        fig.add_hline(y=-2.0, line_dash="dash", line_color="rgba(255,255,255,0.4)", row=2, col=1)

        # Correlation plots (Row 3)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=_sanitize(df['corr_pearson']),
            name="Pearson (Ret)",
            line=dict(color="#FFD700", width=1.5)
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=_sanitize(df['corr_pearson_price']),
            name="Pearson (Price)",
            line=dict(color="#FFA500", width=1.5)
        ), row=3, col=1)

        fig.update_layout(
            showlegend=True,
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
            height=800,
            width=1470,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        fig.update_xaxes(rangeslider_visible=False)

        for i in [1, 2, 3]:
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.15)", zeroline=False, row=i, col=1)
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.15)", zeroline=True,
                            zerolinecolor="rgba(255,255,255,0.2)", row=i, col=1)

        return fig

    @render.ui
    def pair_metrics_viz():
        met = metrics_res.get()
        if not met:
            return None

        # Define all metrics
        metrics = [
            "Coefficient",
            "VolRatio",
            "ADF_P",
            "R2",
            "HalfLife",
            "SpreadVol",
            "BetaStability",
            "P_Correlation",
            "R_Correlation"
        ]

        boxes = []
        for m in metrics:
            v = met.get(m)
            if v is not None:
                boxes.append(
                    ui.value_box(
                        title=m,
                        value=f"{v:.2f}",
                        icon="bar-chart",
                        color="cyan",
                        width=1,
                        class_="small-box"
                    )
                )
        # Put all boxes in a single row using flexbox for maximum compactness
        return ui.div(
            *boxes,
            class_="d-flex flex-row flex-nowrap gap-1 overflow-auto justify-content-between",
            style="padding-bottom: 5px;"
        )

    @reactive.calc
    def copula_stats():
        copula_trigger.get() # Dependency for manual debounce
        df = pair_data.get()
        if df.empty: return None

        mode = input.copula_mode()
        w = max(input.pair_window(), 10)
        df_plot = df.tail(w).copy()
        
        r_window = max(int(input.r_window() or 1), 1)

        if mode == "price":
            # Note: price_a in synthetic is log(price)
            # Use full df for EMA calculation to avoid edge effects
            p_a_full = np.exp(df["price_a"])
            p_b_full = np.exp(df["price_b"])

            if input.copula_stationarize():
                try:
                    ew = max(int(input.copula_ema_window() or 20), 2)
                except (TypeError, ValueError):
                    ew = 20
                x_full = p_a_full / p_a_full.ewm(span=ew).mean()
                y_full = p_b_full / p_b_full.ewm(span=ew).mean()
                title_suffix = f" (Stationary Price, {ew} EMA)"
            else:
                x_full = p_a_full
                y_full = p_b_full
                title_suffix = " (Price Levels)"
            
            x_raw = x_full.tail(w).values
            y_raw = y_full.tail(w).values
        else:
            df_plot = df.tail(w).copy()
            df_plot = df_plot.dropna(subset=["log_ret_a", "log_ret_b"])
            if len(df_plot) < 2 or r_window < 2:
                x_raw = df_plot["log_ret_a"].values
                y_raw = df_plot["log_ret_b"].values
                title_suffix = " (Raw Log Returns)"
            else:
                x_raw = df_plot["log_ret_a"].rolling(r_window).sum().dropna().values
                y_raw = df_plot["log_ret_b"].rolling(r_window).sum().dropna().values
                title_suffix = " (Log Returns)"

        if len(x_raw) < 1: 
            return None

        x = _sanitize(x_raw)
        y = _sanitize(y_raw)

        # Ranks (U, V)
        u_hist = rankdata(x) / (len(x) + 1)
        v_hist = rankdata(y) / (len(y) + 1)

        u_curr = u_hist[-1]
        v_curr = v_hist[-1]

        # Conditional Probability
        c_type = input.copula_type()
        c_param = input.copula_param()
        
        p_uv, p_vu = 0.5, 0.5
        try:
            kwargs = {}
            if c_type == "t": kwargs["df"] = c_param
            else: kwargs["theta"] = c_param

            # Calculate theoretical conditional probs
            p_uv, p_vu = copula_cond_probs(u_hist, v_hist, u_curr, v_curr, method=c_type, **kwargs)
        except Exception as e:
            logger.log("Pair Radar", "ERROR", f"Copula Prob Error: {e}")
            p_uv, p_vu = 0.5, 0.5

        # --- KDE for Marginals ---
        # Instead of flat histograms of ranks, we show the density of the original data 
        # mapped to the rank axis.
        def get_kde_path(vals, ranks):
            if len(vals) < 5: return np.zeros_like(ranks)
            kde = gaussian_kde(vals, bw_method=0.3)
            # Evaluate KDE at the original points
            dens = kde.evaluate(vals)
            # Normalize for visualization
            if dens.max() > 0: dens = dens / dens.max()
            return dens

        dens_a = get_kde_path(x, u_hist)
        dens_b = get_kde_path(y, v_hist)

        return {
            "u": u_hist, "v": v_hist, 
            "u_curr": u_curr, "v_curr": v_curr,
            "x": x, "y": y,
            "p_uv": p_uv,
            "p_vu": p_vu,
            "dens_a": dens_a,
            "dens_b": dens_b,
            "title_suffix": title_suffix
        }

    @render.ui
    def copula_prob_info():
        s = copula_stats()
        if not s:
            uv_str = "---"
        else:
            p_uv = s["p_uv"]
            p_vu = s["p_vu"]
            uv_str = f"{p_uv:.4f}"
            vu_str = f"{p_vu:.4f}"

        return ui.div(
            ui.value_box(
                "P(U \u2264 u | V=v)", 
                uv_str,
                theme="info",
                style="flex: 1",
                class_="small-box"
            ),
            ui.value_box(
                "P(V \u2264 v | U=u)", 
                vu_str,
                theme="info",
                style="flex: 1",
                class_="small-box"
            ),
            class_="d-flex flex-row gap-2"
        )

    @render_widget
    def pair_copula_chart():
        s = copula_stats()
        if not s: return None

        u, v = s["u"], s["v"]
        u_curr, v_curr = s["u_curr"], s["v_curr"]
        x, y = s["x"], s["y"]
        p_uv, p_vu = s["p_uv"], s["p_vu"]
        dens_a, dens_b = s["dens_a"], s["dens_b"]
        title_suffix = s["title_suffix"]

        # Manual subplots for scatter + marginals
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.85, 0.15],
            row_heights=[0.15, 0.85],
            shared_xaxes=True,
            shared_yaxes=True,
            horizontal_spacing=0.01,
            vertical_spacing=0.01
        )

        # Main Scatter (Row 2, Col 1)
        fig.add_trace(go.Scatter(
            x=u, y=v,
            mode='markers',
            marker=dict(size=7, color='cyan', opacity=0.4, line=dict(width=0.4, color='white')),
            text=[f"A: {v1:,.4f}<br>B: {v2:,.4f}<br>Rank A: {r1:.3f}<br>Rank B: {r2:.3f}" for v1, v2, r1, r2 in zip(x, y, u, v)],
            hovertemplate="%{text}<extra></extra>",
            name="Joint Distribution"
        ), row=2, col=1)

        # Path
        if len(u) >= 10:
            fig.add_trace(go.Scatter(
                x=u[-10:], y=v[-10:],
                mode='lines+markers',
                line=dict(color='yellow', width=1.2, dash='dot'),
                marker=dict(size=5, color='yellow'),
                name="Last 10",
                hoverinfo='skip'
            ), row=2, col=1)

        # Current
        fig.add_trace(go.Scatter(
            x=[u_curr], y=[v_curr],
            mode='markers',
            marker=dict(size=14, color='red', symbol='cross', line=dict(width=2, color='white')),
            text=[f"CURRENT<br>A: {x[-1]:,.4f}<br>B: {y[-1]:,.4f}<br>Rank A: {u_curr:.4f}<br>Rank B: {v_curr:.4f}<br>P(U|V): {p_uv:.4f}<br>P(V|U): {p_vu:.4f}"],
            hovertemplate="%{text}<extra></extra>",
            name="Current"
        ), row=2, col=1)

        # Marginals as KDE Areas
        # X Marginal (Row 1, Col 1)
        sort_idx_a = np.argsort(u)
        fig.add_trace(go.Scatter(
            x=u[sort_idx_a], y=dens_a[sort_idx_a],
            fill='tozeroy', 
            line=dict(color='cyan', width=1),
            fillcolor='rgba(0, 255, 255, 0.2)',
            name="Rank A Density",
            hoverinfo='skip'
        ), row=1, col=1)
        
        # Y Marginal (Row 2, Col 2)
        sort_idx_b = np.argsort(v)
        fig.add_trace(go.Scatter(
            y=v[sort_idx_b], x=dens_b[sort_idx_b], # Swap x/y for vertical marginal
            fill='tozerox', 
            line=dict(color='cyan', width=1),
            fillcolor='rgba(0, 255, 255, 0.2)',
            name="Rank B Density",
            hoverinfo='skip'
        ), row=2, col=2)

        # Refined crosshairs
        fig.add_vline(x=u_curr, line_dash="dash", line_color="rgba(255,255,255,0.4)", line_width=1, row=2, col=1)
        fig.add_hline(y=v_curr, line_dash="dash", line_color="rgba(255,255,255,0.4)", line_width=1, row=2, col=1)

        fig.update_layout(
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            height=850,
            width=1090,
            showlegend=False,
            template="plotly_dark",
            font=dict(family="Space Mono", color="white")
        )

        # Axis Styling - High visibility zerolines
        axis_style = dict(
            gridcolor="rgba(255, 255, 255, 0.1)",
            zeroline=True,
            zerolinecolor="rgba(255, 255, 255, 0.4)",
            zerolinewidth=1.5,
            range=[-0.02, 1.02],
            showline=True,
            linecolor="rgba(255,255,255,0.2)"
        )
        fig.update_xaxes(axis_style, row=2, col=1, title_text=f"(U) Rank {input.symbol_a()} {title_suffix}")
        fig.update_yaxes(axis_style, row=2, col=1, title_text=f"(V) Rank {input.symbol_b()} {title_suffix}")
        
        # Marginal axes hidden but styled with white low-opacity grid
        m_axis_style = dict(
            showticklabels=False, 
            zeroline=True, 
            zerolinecolor="rgba(255, 255, 255, 0.4)",
            gridcolor="rgba(255, 255, 255, 0.2)", # Increased opacity
            showgrid=True
        )
        fig.update_xaxes(m_axis_style, row=1, col=1)
        fig.update_yaxes(m_axis_style, row=1, col=1)
        fig.update_xaxes(m_axis_style, row=2, col=2)
        fig.update_yaxes(m_axis_style, row=2, col=2)

        return fig

    @render_widget
    def pair_comp_chart():
        df = pair_data.get()
        if df.empty or "cum_ret_a" not in df.columns or "cum_ret_b" not in df.columns:
            return None

        window = input.rolling_window()

        df_plot = df.tail(input.pair_window() + window).copy()
        df_plot.index = pd.to_datetime(df_plot.index)
        df_plot.index = df_plot.index.strftime("%Y-%m-%d %H:%M:%S")

        df_plot = df.tail(input.pair_window()).copy()

        # Use pre-calculated log returns for scaling comparison
        ret_a = df_plot["log_ret_a"].dropna()
        ret_b = df_plot["log_ret_b"].dropna()

        # Calculate beta (vol ratio) for better visual scaling
        beta = ret_a.std() / ret_b.std()

        # Scale cumulative return of A to match B visually
        cum_ret_a_scaled = (ret_a / beta).cumsum()
        cum_ret_b_scaled = ret_b.cumsum()

        # Create figure with 2 rows: Performance, Corr
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": True}], [{}]]
        )

        # --- Stationary Price Calculation ---
        try:
            ew_comp = max(int(input.copula_ema_window() or 20), 2)
        except (TypeError, ValueError):
            ew_comp = 20
            
        p_a_full = np.exp(df["price_a"])
        p_b_full = np.exp(df["price_b"])
        stat_a_full = p_a_full / p_a_full.ewm(span=ew_comp).mean()
        stat_b_full = p_b_full / p_b_full.ewm(span=ew_comp).mean()
        
        stat_a = stat_a_full.tail(input.pair_window()).values
        stat_b = stat_b_full.tail(input.pair_window()).values

        # --- Row 1: Performance Comparison ---
        mode = input.comp_price_mode()
        
        if mode == "vol":
            # Vol scaled
            fig.add_trace(go.Scatter(
                x=df_plot.index,
                y=cum_ret_a_scaled,
                name=f"{input.symbol_a()} Vol Scaled",
                line=dict(color="#FFD60A", width=1.5),
                opacity=1
            ), row=1, col=1, secondary_y=False)

            fig.add_trace(go.Scatter(
                x=df_plot.index,
                y=cum_ret_b_scaled,
                name=f"{input.symbol_b()} Vol Scaled",
                line=dict(color="#FF006E", width=1.5),
                opacity=1
            ), row=1, col=1, secondary_y=False)
            
            fig.update_yaxes(title_text="Cumulative Log Return", row=1, col=1, secondary_y=False)

        elif mode == "stat":
            # Stationary Prices
            fig.add_trace(go.Scatter(
                x=df_plot.index,
                y=stat_a,
                name=f"{input.symbol_a()} Stationary",
                line=dict(color="#00FFFF", width=1.5),
                opacity=0.9
            ), row=1, col=1, secondary_y=False)

            fig.add_trace(go.Scatter(
                x=df_plot.index,
                y=stat_b,
                name=f"{input.symbol_b()} Stationary",
                line=dict(color="#FAFF00", width=1.5),
                opacity=0.9
            ), row=1, col=1, secondary_y=False)
            
            fig.update_yaxes(title_text="Ratio to EMA", row=1, col=1, secondary_y=False)

        # --- Row 2: Correlation ---
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=_sanitize(df_plot['corr_pearson']),
            name="Pearson (Ret)",
            line=dict(color="#FFD700", width=1)
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=_sanitize(df_plot['corr_pearson_price']),
            name="Pearson (Price)",
            line=dict(color="#FFA500", width=1)
        ), row=2, col=1)

        # Layout
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            font=dict(family="Space Mono", color="white"),
            height=800,
            width=1470,
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.12,
                xanchor="center",
                x=0.5
            )
        )

        fig.update_yaxes(row=1, col=1, secondary_y=False)
        fig.update_yaxes(row=1, col=1, secondary_y=True, showgrid=False)
        fig.update_yaxes(row=2, col=1)

        for i in [1, 2]:
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.15)", zeroline=False, row=i, col=1)
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.15)", zeroline=True, 
                            zerolinecolor="rgba(255,255,255,0.2)", row=i, col=1)
            
        return fig