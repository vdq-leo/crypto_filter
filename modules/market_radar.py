from shiny import ui, render, reactive
import faicons as fa
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.config import METRIC_LABELS, BENCHMARK_SYMBOL, TRADINGVIEW_URL, ALL_METRICS, RADAR_SOURCES, RADAR_METRICS, SNAPSHOT_METRICS, METRIC_SPECS, AVAILABLE_INTERVALS, MANDATORY_CRYPTO, IGNORED_CRYPTO, API_BASE_URL
from src.logger import logger
from scipy import stats
import requests

def market_radar_ui():
    return ui.navset_card_underline(
        ui.nav_panel("Market Radar",
            ui.layout_sidebar(
                ui.sidebar(

                    ui.h5("Market Filters"),

                    ui.input_action_button(
                        "btn_calc_snapshot",
                        "Calculate Metrics",
                        class_="btn-primary w-100 mb-2"
                    ),
                    ui.output_ui("radar_universe_badge"),
                    ui.input_selectize(
                        "focus_symbol",
                        "Focus Symbol",
                        options={"placeholder": "Select a symbol"},
                        choices=[],
                        multiple=False
                    ),

                    ui.hr(class_="mt-2 mb-2"),

                    # ----- AXES (Source + Interval side-by-side, Metric, Dynamic Params) -----
                    ui.div(
                        ui.div("X AXIS", class_="text-uppercase fw-bold small text-info mb-1", style="letter-spacing: 0.8px;"),
                        ui.layout_columns(
                            ui.input_select("x_source", "Source", choices=RADAR_SOURCES, selected="ohlcv", width="100%"),
                            ui.panel_conditional(
                                "input.x_source != 'snapshot'",
                                ui.input_select("x_interval", "Int", choices=AVAILABLE_INTERVALS, selected="1h", width="100%")
                            ),
                            col_widths=[8, 4]
                        ),
                        ui.input_select("x_axis", "Metric", choices={m: METRIC_LABELS.get(m, m) for m in RADAR_METRICS}, selected="price_zscore", width="100%"),
                        ui.output_ui("x_params_ui"),
                        class_="p-2 mb-2 rounded border border-secondary bg-dark bg-opacity-25",
                        style="overflow: hidden;"
                    ),

                    ui.div(
                        ui.div("Y AXIS", class_="text-uppercase fw-bold small text-info mb-1", style="letter-spacing: 0.8px;"),
                        ui.layout_columns(
                            ui.input_select("y_source", "Source", choices=RADAR_SOURCES, selected="ohlcv", width="100%"),
                            ui.panel_conditional(
                                "input.y_source != 'snapshot'",
                                ui.input_select("y_interval", "Int", choices=AVAILABLE_INTERVALS, selected="1h", width="100%")
                            ),
                            col_widths=[8, 4]
                        ),
                        ui.input_select("y_axis", "Metric", choices={m: METRIC_LABELS.get(m, m) for m in RADAR_METRICS}, selected="vam", width="100%"),
                        ui.output_ui("y_params_ui"),
                        class_="p-2 mb-2 rounded border border-secondary bg-dark bg-opacity-25",
                        style="overflow: hidden;"
                    ),

                    ui.div(
                        ui.div("Z AXIS (SIZE & COLOR)", class_="text-uppercase fw-bold small text-info mb-1", style="letter-spacing: 0.8px;"),
                        ui.layout_columns(
                            ui.input_select("z_source", "Source", choices=RADAR_SOURCES, selected="ohlcv", width="100%"),
                            ui.panel_conditional(
                                "input.z_source != 'snapshot' && input.z_axis != 'None'",
                                ui.input_select("z_interval", "Int", choices=AVAILABLE_INTERVALS, selected="1h", width="100%")
                            ),
                            col_widths=[8, 4]
                        ),
                        ui.input_select("z_axis", "Metric", choices={"None": "None", **{m: METRIC_LABELS.get(m, m) for m in RADAR_METRICS}}, selected="volume_imbalance", width="100%"),
                        ui.output_ui("z_params_ui"),
                        class_="p-2 mb-2 rounded border border-secondary bg-dark bg-opacity-25",
                        style="overflow: hidden;"
                    ),

                    # ----- NORMALIZE (single row) -----
                    ui.layout_columns(
                        ui.input_select("x_norm", "Norm X", choices=["None", "Rank", "Z-Score"], selected="None"),
                        ui.input_select("y_norm", "Norm Y", choices=["None", "Rank", "Z-Score"], selected="None"),
                        col_widths=[6, 6]
                    ),

                    ui.hr(class_="mt-2 mb-2"),
                    ui.input_switch(
                        "drop_zeros",
                        "Exclude Zero Metrics",
                        value=False
                    ),
                    ui.input_switch(
                        "show_regression",
                        "Show Regression Line",
                        value=True
                    ),   

                    ui.hr(class_="mt-2 mb-2"),              
                    ui.p("Assets are managed globally in the ASSET FILTER tab.", class_="text-muted small"),
                    width=300
                ),
                ui.div(
                    # Hidden markers to ensure reactive outputs are transmitted
                    ui.div(ui.output_text("snapshot_ready"), class_="d-none"),
                    
                    ui.panel_conditional(
                        "input.btn_calc_snapshot == 0",
                        ui.div(ui.h4("Click 'Calculate Metrics' to see data"), class_="text-center mt-5")
                    ),
                    ui.panel_conditional(
                        "input.btn_calc_snapshot > 0",
                        ui.div(
                            ui.card(
                                ui.card_header("Market Radar Snapshot"),
                                output_widget("snapshot_chart"),
                                full_screen=True
                            ),
                            ui.card(
                                ui.card_header("Data Table"),
                                ui.output_data_frame("snapshot_table")
                            )
                        )
                    )
                )
            ),
            value="tab_radar"
        ),
        ui.nav_panel("Path Analysis",
            ui.layout_sidebar(
                ui.sidebar(

                    ui.h5("Path Settings"),

                    ui.input_action_button(
                        "btn_gen_rpg",
                        "Generate Path",
                        class_="btn-primary w-100 mb-2"
                    ),
                    ui.output_ui("rpg_universe_badge"),

                    ui.input_selectize(
                        "rpg_focus_symbol",
                        "Focus Symbol",
                        choices=[],
                        multiple=False
                    ),

                    ui.input_select(
                        "rpg_interval",
                        "Interval",
                        choices=AVAILABLE_INTERVALS,
                        selected="1h"
                    ),

                    ui.hr(class_="mt-2 mb-2"),

                    # ----- AXES (each own row) -----
                    ui.input_select(
                        "rpg_x",
                        "X Axis",
                        choices={m: METRIC_LABELS.get(m, m) for m in RADAR_METRICS},
                        selected="rel_strength_z"
                    ),

                    ui.input_select(
                        "rpg_y",
                        "Y Axis",
                        choices={m: METRIC_LABELS.get(m, m) for m in RADAR_METRICS},
                        selected="breakout_score_dist"
                    ),

                    # ----- LOG SCALE (single row) -----
                    ui.layout_columns(
                        ui.input_select("rpg_x_norm", "Norm X", choices=["None", "Rank", "Z-Score"], selected="None"),
                        ui.input_select("rpg_y_norm", "Norm Y", choices=["None", "Rank", "Z-Score"], selected="None"),
                        col_widths=[6, 6]
                    ),

                    ui.hr(class_="mt-2 mb-2"),

                    # ----- NUMERIC SETTINGS -----
                    ui.layout_columns(
                        ui.input_numeric(
                            "step_size",
                            "Stride",
                            value=10,
                            min=1,
                            max=50
                        ),
                        ui.input_numeric(
                            "max_points",
                            "Length",
                            value=3,
                            min=1,
                            max=200
                        ),
                        col_widths=[5, 5]
                    ),

                    ui.hr(class_="mt-2 mb-2"),

                    ui.hr(class_="mt-2 mb-2"),
                    ui.p("Compare symbols are managed globally in the ASSET FILTER tab.", class_="text-muted small")
                ),

                ui.div(
                    # Hidden marker
                    ui.div(ui.output_text("rpg_ready"), class_="d-none"),
                    
                    ui.panel_conditional(
                        "input.btn_gen_rpg == 0",
                        ui.div(ui.h4("Click 'Generate Path' to see movement"), class_="text-center mt-5")
                    ),
                    ui.panel_conditional(
                        "input.btn_gen_rpg > 0",
                        ui.div(
                            ui.card(
                                ui.card_header("Path Analysis"),
                                output_widget("rpg_chart"),
                                full_screen=True
                            ),
                            ui.card(
                                ui.card_header("Path Data"),
                                ui.output_data_frame("rpg_table")
                            )
                        )
                    )
                )
            ),
            value="tab_rpg"
        ),
        id="radar_nav"
    )

def market_radar_server(input, output, session, global_interval, global_universe):
    snapshot_data = reactive.Value(pd.DataFrame())
    active_axes_meta = reactive.Value({"x_label": "X", "y_label": "Y", "z_label": "None", "z_active": False})
    rpg_data = reactive.Value(pd.DataFrame())
    selected_symbol_data = reactive.Value(None)
    
    # Track selected symbols for snapshot and RPG
    selected_symbols_radar = reactive.Value(set())
    selected_symbols_rpg = reactive.Value(set())
    
    # Track focus symbol for RPG specifically
    selected_rpg_focus_data = reactive.Value(None)

    logger.log("Market Radar", "INFO", "Server initialized")

    @reactive.effect
    @reactive.event(global_universe)
    def _update_radar_symbols_from_global():
        syms = global_universe.get()
        if syms:
            clean_syms = [s for s in syms if s not in IGNORED_CRYPTO]
            selected_symbols_radar.set(set(clean_syms))
            selected_symbols_rpg.set(set(clean_syms))
            curr_sel = sorted(list(clean_syms))
            
            with reactive.isolate():
                curr_focus = input.focus_symbol()
                curr_rpg_focus = input.rpg_focus_symbol()
            new_focus = curr_focus if (curr_focus and curr_focus in curr_sel) else (curr_sel[0] if curr_sel else "")
            new_rpg_focus = curr_rpg_focus if (curr_rpg_focus and curr_rpg_focus in curr_sel) else (curr_sel[0] if curr_sel else "")
            ui.update_selectize("focus_symbol", choices=curr_sel, selected=new_focus)
            ui.update_selectize("rpg_focus_symbol", choices=curr_sel, selected=new_rpg_focus)

    @reactive.effect
    @reactive.event(input.focus_symbol, ignore_init=True)
    def _sync_focus_symbol():
        symbol = input.focus_symbol()
        if symbol:
            sym_upper = symbol.upper()
            selected_symbol_data.set(sym_upper)
            if input.btn_calc_snapshot() > 0:
                ui.update_selectize("quick_symbol", selected=[sym_upper])

    @reactive.effect
    def _sync_rpg_focus_symbol():
        symbol = input.rpg_focus_symbol()
        if symbol:
            selected_rpg_focus_data.set(symbol.upper())

    def _get_metric_choices_for_source(source_key, include_none=False):
        if source_key == "snapshot":
            choices = {m: METRIC_LABELS.get(m, m) for m in SNAPSHOT_METRICS}
        elif source_key == "ohlcv":
            choices = {m: METRIC_LABELS.get(m, m) for m in RADAR_METRICS}
        else:
            choices = {
                m: METRIC_LABELS.get(m, m)
                for m in RADAR_METRICS
                if not METRIC_SPECS.get(m, {}).get("ohlcv_only", False)
            }
        if include_none:
            return {"None": "None", **choices}
        return choices

    @reactive.effect
    @reactive.event(input.x_source, ignore_init=True)
    def _():
        choices = _get_metric_choices_for_source(input.x_source())
        with reactive.isolate():
            curr = input.x_axis()
        sel = curr if curr in choices else next(iter(choices))
        ui.update_select("x_axis", choices=choices, selected=sel)

    @reactive.effect
    @reactive.event(input.y_source, ignore_init=True)
    def _():
        choices = _get_metric_choices_for_source(input.y_source())
        with reactive.isolate():
            curr = input.y_axis()
        sel = curr if curr in choices else next(iter(choices))
        ui.update_select("y_axis", choices=choices, selected=sel)

    @reactive.effect
    @reactive.event(input.z_source, ignore_init=True)
    def _():
        choices = _get_metric_choices_for_source(input.z_source(), include_none=True)
        with reactive.isolate():
            curr = input.z_axis()
        sel = curr if curr in choices else next(iter(choices))
        ui.update_select("z_axis", choices=choices, selected=sel)

    def _render_axis_params(prefix, metric_key):
        if not metric_key or metric_key == "None":
            return ui.div()
        spec = METRIC_SPECS.get(metric_key, {})
        params = spec.get("params", [])
        if not params:
            return ui.div()
        inputs = []
        for p in params:
            inp_id = f"{prefix}_param_{p['name']}"
            with reactive.isolate():
                try:
                    curr_val = getattr(input, inp_id)()
                except Exception:
                    curr_val = None
            val = curr_val if curr_val is not None else p["default"]
            inputs.append(
                ui.input_numeric(
                    inp_id, p["label"], value=val,
                    min=p.get("min"), max=p.get("max"), step=p.get("step"),
                    width="100%"
                )
            )
        if len(inputs) == 1:
            return inputs[0]
        rows = []
        for i in range(0, len(inputs), 2):
            pair = inputs[i:i+2]
            if len(pair) == 2:
                rows.append(ui.layout_columns(*pair, col_widths=[6, 6]))
            else:
                rows.append(pair[0])
        return ui.div(*rows)

    @render.ui
    def x_params_ui():
        return _render_axis_params("x", input.x_axis())

    @render.ui
    def y_params_ui():
        return _render_axis_params("y", input.y_axis())

    @render.ui
    def z_params_ui():
        return _render_axis_params("z", input.z_axis())

    def _collect_axis_config(prefix, source_val, metric_val, interval_val):
        params = {}
        spec = METRIC_SPECS.get(metric_val, {})
        for p in spec.get("params", []):
            inp_id = f"{prefix}_param_{p['name']}"
            try:
                val = getattr(input, inp_id)()
                params[p["name"]] = val if val is not None else p["default"]
            except Exception:
                params[p["name"]] = p["default"]
        return {
            "source": source_val,
            "metric": metric_val,
            "interval": interval_val if source_val != "snapshot" else "1h",
            "params": params,
        }

    async def _run_snapshot_fetch(show_progress=True):
        try:
            with reactive.isolate():
                syms = list(selected_symbols_radar.get() or global_universe.get() or MANDATORY_CRYPTO)
            symbols = sorted([s for s in syms if s not in IGNORED_CRYPTO])
            if not symbols:
                ui.notification_show("Please select symbols for analysis in Asset Filter.", type="warning")
                return

            x_cfg = _collect_axis_config("x", input.x_source(), input.x_axis(), input.x_interval())
            y_cfg = _collect_axis_config("y", input.y_source(), input.y_axis(), input.y_interval())
            z_metric = input.z_axis()
            z_cfg = _collect_axis_config("z", input.z_source(), z_metric, input.z_interval()) if z_metric != "None" else None

            payload = {
                "symbols": symbols,
                "x": x_cfg,
                "y": y_cfg,
                "z": z_cfg,
            }

            if show_progress:
                with ui.Progress(min=0, max=100) as p:
                    p.set(message="Computing Market Radar...")
                    res = requests.post(f"{API_BASE_URL}/market-radar/snapshot", json=payload, timeout=120)
            else:
                res = requests.post(f"{API_BASE_URL}/market-radar/snapshot", json=payload, timeout=120)

            if res.status_code == 200:
                resp_json = res.json()
                results = resp_json.get("metrics", [])
                if not results:
                    ui.notification_show("Failed to compute metrics for any symbols.", type="error")
                    return
                res_df = pd.DataFrame(results)
                active_axes_meta.set({
                    "x_label": _axis_label(x_cfg["source"], x_cfg["metric"], x_cfg["interval"]),
                    "y_label": _axis_label(y_cfg["source"], y_cfg["metric"], y_cfg["interval"]),
                    "z_label": _axis_label(z_cfg["source"], z_cfg["metric"], z_cfg["interval"]) if z_cfg else "None",
                    "z_active": z_cfg is not None,
                })
                snapshot_data.set(res_df)
                if resp_json.get("rate_limited"):
                    cd = resp_json.get("cooldown_sec", 0)
                    ui.notification_show(f"Binance API rate-limited (cooldown {cd}s). Orderbook/API metrics may show 0 until cooldown ends.", type="warning")
                elif show_progress:
                    hits = resp_json.get("cache_hits", 0)
                    misses = resp_json.get("cache_misses", 0)
                    ui.notification_show(f"Updated {len(res_df)} assets (cache hits: {hits}, computed: {misses})", type="success")
            else:
                ui.notification_show(f"Calculation error: {res.text}", type="error")
        except Exception as e:
            logger.log("Market Radar", "ERROR", f"Snapshot error: {str(e)}")
            ui.notification_show(f"Calculation error: {str(e)}", type="error")

    @reactive.effect
    @reactive.event(input.btn_calc_snapshot)
    async def _():
        await _run_snapshot_fetch(show_progress=True)

    @render.ui
    def radar_universe_badge():
        syms = selected_symbols_radar.get() or set(global_universe.get() or [])
        count = len(syms) if syms else 0
        return ui.div(
            ui.span(
                fa.icon_svg("globe"),
                f"Global: {count} assets",
                class_="badge bg-info text-dark px-3 py-1 d-inline-flex align-items-center justify-content-center gap-1 font-monospace shadow-sm",
                style="font-size: 0.8rem; letter-spacing: 0.5px;"
            ),
            class_="d-flex justify-content-center w-100 my-2 text-center"
        )

    @render.ui
    def rpg_universe_badge():
        syms = selected_symbols_rpg.get() or set(global_universe.get() or [])
        count = len(syms) if syms else 0
        return ui.div(
            ui.span(
                fa.icon_svg("globe"),
                f"Global: {count} assets",
                class_="badge bg-info text-dark px-3 py-1 d-inline-flex align-items-center justify-content-center gap-1 font-monospace shadow-sm",
                style="font-size: 0.8rem; letter-spacing: 0.5px;"
            ),
            class_="d-flex justify-content-center w-100 my-2 text-center"
        )

    @render.text
    def snapshot_ready():
        return "true" if not snapshot_data.get().empty else "false"

    @render.text
    def rpg_ready():
        return "true" if not rpg_data.get().empty else "false"
    
    @reactive.calc
    def filtered_snapshot_df():
        df = snapshot_data.get().copy()
        if df.empty:
            return df
        
        active_syms = selected_symbols_radar.get()
        if active_syms:
            df = df[df['symbol'].isin(active_syms)]
            
        if input.drop_zeros():
            x_nonzero = pd.to_numeric(df['x_val'], errors='coerce').abs().fillna(0) > 1e-12
            y_nonzero = pd.to_numeric(df['y_val'], errors='coerce').abs().fillna(0) > 1e-12
            df = df[x_nonzero & y_nonzero]
            
        x_norm = input.x_norm()
        y_norm = input.y_norm()

        if 'x_val' in df.columns:
            if x_norm == "Rank":
                df['x_val'] = pd.to_numeric(df['x_val'], errors='coerce').rank(pct=True)
            elif x_norm == "Z-Score":
                series = pd.to_numeric(df['x_val'], errors='coerce')
                std_x = series.std()
                df['x_val'] = ((series - series.mean()) / std_x) if (std_x and std_x > 1e-9) else 0.0

        if 'y_val' in df.columns:
            if y_norm == "Rank":
                df['y_val'] = pd.to_numeric(df['y_val'], errors='coerce').rank(pct=True)
            elif y_norm == "Z-Score":
                series = pd.to_numeric(df['y_val'], errors='coerce')
                std_y = series.std()
                df['y_val'] = ((series - series.mean()) / std_y) if (std_y and std_y > 1e-9) else 0.0
            
        return df

    @reactive.calc
    def snapshot_regression_params():
        plot_df = filtered_snapshot_df()
        if len(plot_df) <= 1 or 'x_val' not in plot_df.columns or 'y_val' not in plot_df.columns:
            return None
        
        try:
            reg_x = pd.to_numeric(plot_df['x_val'], errors='coerce').values
            reg_y = pd.to_numeric(plot_df['y_val'], errors='coerce').values
            mask = np.isfinite(reg_x) & np.isfinite(reg_y)
            reg_x, reg_y = reg_x[mask], reg_y[mask]
            
            if len(reg_x) > 1 and np.ptp(reg_x) > 1e-9:
                slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)
                x_range = np.linspace(reg_x.min(), reg_x.max(), 100)
                y_range = intercept + slope * x_range
                return {
                    "x_range": x_range,
                    "y_range": y_range,
                    "r_squared": r_value**2,
                    "intercept": intercept,
                    "slope": slope
                }
        except Exception as e:
            logger.log("Market Radar", "ERROR", f"Regression calculation error: {e}")
            
        return None

    def _axis_label(source_key, metric_key, interval_key=None):
        m_label = METRIC_LABELS.get(metric_key, metric_key)
        if source_key == "snapshot":
            return m_label
        s_label = RADAR_SOURCES.get(source_key, source_key)
        int_str = f" [{interval_key}]" if interval_key else ""
        return f"{s_label}{int_str} · {m_label}"

    @render.data_frame
    def snapshot_table():
        df = filtered_snapshot_df()
        if df.empty:
            return df
        meta = active_axes_meta.get()
        x_lbl = meta.get("x_label", "X")
        y_lbl = meta.get("y_label", "Y")
        cols = ['symbol', 'x_val', 'y_val']
        rename_map = {
            'symbol': 'Symbol',
            'x_val': x_lbl,
            'y_val': y_lbl,
        }
        if meta.get("z_active") and 'z_val' in df.columns:
            cols.append('z_val')
            rename_map['z_val'] = meta.get("z_label", "Z")
        out = df[[c for c in cols if c in df.columns]].copy()
        for num_col in out.columns:
            if num_col != 'symbol':
                out[num_col] = pd.to_numeric(out[num_col], errors='coerce').round(6)
        return out.rename(columns=rename_map)

    @render_widget
    def snapshot_chart():
        plot_df = filtered_snapshot_df().copy()
        if plot_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data matching filters or symbols not loaded", showarrow=False, font=dict(size=20))
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0b3d91", plot_bgcolor="#0b3d91")
            return fig

        meta = active_axes_meta.get()
        z_metric = "active" if meta.get("z_active") else "None"
        x_label = meta.get("x_label", "X")
        y_label = meta.get("y_label", "Y")
        z_label = meta.get("z_label", "None")

        for col in ['x_val', 'y_val']:
            if col not in plot_df.columns:
                plot_df[col] = 0.0
            else:
                plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce').fillna(0.0)

        if z_metric != "None" and 'z_val' in plot_df.columns:
            plot_df['z_val'] = pd.to_numeric(plot_df['z_val'], errors='coerce').fillna(0.0)
            denom = plot_df['z_val'].max() - plot_df['z_val'].min()
            if denom > 1e-9:
                z_norm = (plot_df['z_val'] - plot_df['z_val'].min()) / denom
                plot_df['z_marker_size'] = (z_norm * 14 + 6).fillna(8.0)
            else:
                plot_df['z_marker_size'] = 8.0
            plot_df['z_marker_size'] = plot_df['z_marker_size'].replace([np.inf, -np.inf], np.nan).fillna(8.0)
            
            fig = px.scatter(
                plot_df, x='x_val', y='y_val', size='z_marker_size', color='z_val',
                hover_name='symbol',
                color_continuous_scale='Spectral_r',
                template="plotly_dark",
                custom_data=['symbol', 'z_val']
            )
            fig.update_coloraxes(
                colorbar=dict(
                    title=dict(text=z_label, side="right"),
                    thickness=14,
                    len=0.88,
                    x=1.01
                )
            )
        else:
            fig = px.scatter(
                plot_df, x='x_val', y='y_val', hover_name='symbol',
                template="plotly_dark",
                custom_data=['symbol']
            )
            fig.update_traces(marker=dict(color='#00F5FF', size=8))
            
        fig.update_layout(
            autosize=True,
            margin=dict(l=70, r=35, t=55, b=65),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
                bgcolor="rgba(0,0,0,0)"
            ),
            xaxis_title=x_label,
            yaxis_title=y_label,
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            height=620,
            font=dict(family="Space Mono", color="white"),
            clickmode='event+select'
        )

        fig.update_xaxes(
            title_standoff=12,
            gridcolor="rgba(255, 255, 255, 0.2)",
            zerolinecolor="rgba(255, 255, 255, 0.5)",
            zerolinewidth=1.5,
            linecolor="white",
            tickcolor="white"
        )
        fig.update_yaxes(
            title_standoff=12,
            gridcolor="rgba(255, 255, 255, 0.2)",
            zerolinecolor="rgba(255, 255, 255, 0.5)",
            zerolinewidth=1.5,
            linecolor="white",
            tickcolor="white"
        )
                               
        focus_sym = (input.focus_symbol() or "").strip().upper()
        selected_points = None
        unselected_opacity = 0.2

        if focus_sym and focus_sym in plot_df['symbol'].str.upper().values:
            pos_idx = list(plot_df['symbol'].str.upper()).index(focus_sym)
            selected_points = [pos_idx]

        hover_tpl = (
            '<b>%{hovertext}</b><br>' +
            f'{x_label}: %{{x:.4f}}<br>' +
            f'{y_label}: %{{y:.4f}}<br>' +
            (f'{z_label}: %{{customdata[1]:.4f}}<br>' if z_metric != "None" else '') +
            '<extra></extra>'
        )

        fig.update_traces(
            hovertemplate=hover_tpl,
            selectedpoints=selected_points,
            marker=dict(
                line=dict(
                    color="white",
                    width=1
                )
            ),
            selected=dict(
                marker=dict(opacity=1)
            ),
            unselected=dict(
                marker=dict(opacity=unselected_opacity)
            )
        )

        # fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)", linecolor="white", tickcolor="white")
        # fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)", linecolor="white", tickcolor="white")
 
        # --- Regression Line ---
        reg = snapshot_regression_params()
        if input.show_regression(): 
            if reg:
                fig.add_trace(go.Scatter(
                    x=reg["x_range"],
                    y=reg["y_range"],
                    mode='lines',
                    name=f'Fit (R²={reg["r_squared"]:.3f} | intercept={reg["intercept"]:.3f} | slope={reg["slope"]:.3f})',
                    line=dict(color='orange', width=2, dash='dash'),
                    hoverinfo='skip'
                ))

        return fig
    
    @reactive.effect
    @reactive.event(input.snapshot_chart_click)
    def _handle_chart_click():
        """Handle click events on the snapshot chart"""
        click_data = input.snapshot_chart_click()
        
        if click_data is None:
            return
        
        logger.log("Market Radar", "INFO", f"Chart click event: {click_data}")
        
        # Extract the clicked point data
        try:
            # Plotly click data structure: {'points': [{'customdata': [...], ...}]}
            if 'points' in click_data and len(click_data['points']) > 0:
                point = click_data['points'][0]
                
                # Get symbol from customdata or hovertext
                if 'customdata' in point and point['customdata']:
                    symbol = point['customdata'][0]
                elif 'hovertext' in point:
                    symbol = point['hovertext']
                else:
                    logger.log("Market Radar", "WARNING", "No symbol found in click data")
                    return
                
                logger.log("Market Radar", "INFO", f"Selected symbol: {symbol}")
                sym_upper = str(symbol).strip().upper()
                selected_symbol_data.set(sym_upper)
                
                # Update Focus Symbol input box & Launch Assets taskbar
                ui.update_selectize("focus_symbol", selected=sym_upper)
                ui.update_selectize("quick_symbol", selected=[sym_upper])
        except Exception as e:
            logger.log("Market Radar", "ERROR", f"Error handling chart click: {e}")

    @render.ui
    def selected_symbol_info():
        """Display information about the selected symbol from chart click"""
        selected = selected_symbol_data.get()
        
        if selected is None:
            return ui.div(
                ui.p("Click on a point in the chart above to see detailed metrics", 
                     class_="text-muted text-center",
                     style="padding: 20px;")
            )
        
        # Get the full row data for the selected symbol
        df = snapshot_data.get()
        if df.empty:
            return ui.div(ui.p("No data available", class_="text-muted"))
        
        symbol_row = df[df['symbol'] == selected]
        if symbol_row.empty:
            return ui.div(ui.p(f"Symbol {selected} not found", class_="text-muted"))
        
        row = symbol_row.iloc[0]
        
        # TradingView interval mapping
        tv_intervals = {
            '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
            '1h': '60', '2h': '120', '4h': '240', '6h': '360', '8h': '480', '12h': '720',
            '1d': 'D', '3d': '3D', '1w': 'W'
        }
        tv_int = tv_intervals.get(input.radar_interval(), '60')

        # Create a formatted display of all metrics
        metrics_html = f"""
        <div style="padding: 15px;">
            <h4 style="color: #FFD700; margin-bottom: 15px;">
                {selected}
                <a href="{TRADINGVIEW_URL}?symbol=BINANCE:{selected}.P&interval={tv_int}" target="_blank" 
                   style="font-size: 0.8em; margin-left: 10px; color: #1f77b4; text-decoration: none;">
                    📈 View on TradingView
                </a>
            </h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
        """
        
        # Add all metrics
        for col in df.columns:
            if col != 'symbol':
                value = row[col]
                label = METRIC_LABELS.get(col, col)
                
                # Format the value
                if isinstance(value, (int, float)):
                    if abs(value) < 0.01:
                        formatted_value = f"{value:.6f}"
                    elif abs(value) < 1:
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = f"{value:.2f}"
                    
                    # Color code based on value
                    if value > 0:
                        color = "#4ade80"  # green
                    elif value < 0:
                        color = "#f87171"  # red
                    else:
                        color = "#94a3b8"  # gray
                else:
                    formatted_value = str(value)
                    color = "#94a3b8"
                
                metrics_html += f"""
                <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px;">
                    <div style="font-size: 0.85em; color: #94a3b8;">{label}</div>
                    <div style="font-size: 1.1em; font-weight: bold; color: {color}; font-family: 'Space Mono', monospace;">
                        {formatted_value}
                    </div>
                </div>
                """
        
        metrics_html += """
            </div>
        </div>
        """
        
        return ui.HTML(metrics_html)

    @reactive.effect
    @reactive.event(input.btn_gen_rpg)
    async def _():
        logger.log("Market Radar", "INFO", "RPG calculation triggered")
        try:
            interval = input.rpg_interval()
            with reactive.isolate():
                syms = list(selected_symbols_rpg.get() or global_universe.get() or MANDATORY_CRYPTO)
            compare_symbols = sorted([s for s in syms if s not in IGNORED_CRYPTO])
            if not compare_symbols:
                ui.notification_show("Select symbols in Asset Filter to compare", type="error")
                return
            
            logger.log("Market Radar", "INFO", f"Comparing {len(compare_symbols)} symbols: {compare_symbols}")
            
            step_size = input.step_size()
            max_points = input.max_points()
            x_metric = input.rpg_x()
            y_metric = input.rpg_y()
            
            # 1. API Call for RPG
            with ui.Progress(min=0, max=100) as p:
                p.set(message="Calculating trajectories...")
                
                payload = {
                    "symbols": compare_symbols,
                    "interval": interval,
                    "filter_window": input.filter_window(),
                    "step_size": step_size,
                    "max_points": max_points,
                    "x_metric": get_metric_key(x_metric),
                    "y_metric": get_metric_key(y_metric)
                }
                
                try:
                    res = requests.post(f"{API_BASE_URL}/market-radar/path", json=payload)
                    if res.status_code == 200:
                        results = res.json()["data"]
                        if results:
                            combined_df = pd.DataFrame(results)
                            logger.log("Market Radar", "INFO", f"Trajectory gen complete. Total rows: {len(combined_df)}")
                            rpg_data.set(combined_df)
                            ui.notification_show("Trajectory update complete!", type="success")
                        else:
                            ui.notification_show("No trajectory data generated", type="warning")
                    else:
                        ui.notification_show(f"Calculation error: {res.text}", type="error")
                except Exception as e:
                    ui.notification_show(f"API Connection error: {str(e)}", type="error")
                    
        except Exception as e:
            logger.log("Market Radar", "ERROR", f"RPG error: {str(e)}")
            ui.notification_show(f"RPG calculation failed: {str(e)}", type="error")

    @render.data_frame
    def rpg_table():
        df = rpg_data.get()
        if df.empty: return None
        return df

    @render_widget
    def rpg_chart():
        df = rpg_data.get().copy()
        if df.empty: return go.Figure()
        
        focus_sym = (input.rpg_focus_symbol() or "").strip().upper()
        has_focus = focus_sym in df['Symbol'].str.upper().values
        
        x_norm = input.rpg_x_norm()
        y_norm = input.rpg_y_norm()

        if 'X_Value' in df.columns:
            if x_norm == "Rank":
                df['X_Value'] = df['X_Value'].rank(pct=True)
            elif x_norm == "Z-Score":
                df['X_Value'] = (df['X_Value'] - df['X_Value'].mean()) / df['X_Value'].std()

        if 'Y_Value' in df.columns:
            if y_norm == "Rank":
                df['Y_Value'] = df['Y_Value'].rank(pct=True)
            elif y_norm == "Z-Score":
                df['Y_Value'] = (df['Y_Value'] - df['Y_Value'].mean()) / df['Y_Value'].std()
            
        fig = px.line(
            df, x='X_Value', y='Y_Value', color='Symbol',
            markers=True,
            template="plotly_dark", line_shape='spline'
        )
        
        # Hide legend and set proper axis titles
        x_label = METRIC_LABELS.get(input.rpg_x(), input.rpg_x())
        y_label = METRIC_LABELS.get(input.rpg_y(), input.rpg_y())
        
        fig.update_layout(
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis_title=x_label,
            yaxis_title=y_label,
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            height=600,
            # width=1470,
            font=dict(family="Space Mono", color="white")
        )

        fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.3)")
        fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.3)")

        for trace in fig.data:
            if not isinstance(trace, go.Scatter) or trace.name not in df['Symbol'].unique():
                continue
                
            is_focus = has_focus and trace.name.upper() == focus_sym
                
            # Line Styling
            if has_focus:
                trace.line.width = 4 if is_focus else 1
                trace.opacity = 1.0 if is_focus else 0.15
            else:
                trace.line.width = 2
                trace.opacity = 0.8

            # Marker Styling
            sym_df = df[df['Symbol'] == trace.name]
            trace.marker.size = sym_df['Marker_Size'] * (1.5 if is_focus else 1.0)
            trace.marker.opacity = sym_df['Order'] * (trace.opacity)
            
            if is_focus:
                trace.marker.line = dict(color='white', width=1)
                
                # Add a distinct star marker for the latest point
                fig.add_trace(go.Scatter(
                    x=[sym_df['X_Value'].iloc[-1]], 
                    y=[sym_df['Y_Value'].iloc[-1]],
                    mode='markers+text',
                    text=[trace.name],
                    textposition="top right",
                    marker=dict(
                        size=18,
                        color=trace.line.color,
                        line=dict(color='white', width=2),
                        symbol='star'
                    ),
                    name=f"{trace.name} Focus",
                    showlegend=False
                ))
        
        return fig

def get_metric_key(m):
    return 'return' if m == 'metric_return' else m
