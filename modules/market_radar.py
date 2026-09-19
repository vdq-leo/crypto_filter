from shiny import ui, render, reactive
import faicons as fa
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.config import METRIC_LABELS, BENCHMARK_SYMBOL, TRADINGVIEW_URL, ALL_METRICS, AVAILABLE_INTERVALS, MANDATORY_CRYPTO, IGNORED_CRYPTO, API_BASE_URL
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
                        "Load Data",
                        class_="btn-primary w-100 mb-2"
                    ),
                    ui.input_selectize(
                        "focus_symbol",
                        "Focus Symbol",
                        options={"placeholder": "Select a symbol"},
                        choices=[],
                        multiple=False
                    ),

                    ui.input_select(
                        "radar_interval",
                        "Interval",
                        selected="1h",
                        choices=AVAILABLE_INTERVALS
                    ),

                    ui.input_numeric(
                        "filter_window",
                        "Window Size",
                        value=40,
                        min=5,
                        max=500,
                        step=5
                    ),

                    ui.hr(class_="mt-2 mb-2"),

                    # ----- AXES (each own row) -----
                    ui.input_select(
                        "x_axis",
                        "X Axis",
                        choices={m: METRIC_LABELS.get(m, m) for m in ALL_METRICS},
                        selected="mr_prob"
                    ),

                    ui.input_select(
                        "y_axis",
                        "Y Axis",
                        choices={m: METRIC_LABELS.get(m, m) for m in ALL_METRICS},
                        selected="breakout_score_dist"
                    ),

                    ui.input_select(
                        "z_axis",
                        "Z Axis",
                        choices={"None": "None", **{m: METRIC_LABELS.get(m, m) for m in ALL_METRICS}},
                        selected="volume_imbalance"
                    ),

                    # ----- LOG SCALE (single row) -----
                    ui.layout_columns(
                        ui.input_checkbox("x_log", "Rank X"),
                        ui.input_checkbox("y_log", "Rank Y"),
                        # ui.input_checkbox("z_log", "Log Z"),
                        col_widths=[5, 5]
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
                    ui.input_text(
                        "n_assets_radar",
                        "Top Volume",
                        value="50",
                        placeholder="e.g. 20",
                        update_on="blur"
                    ),
                    ui.input_selectize(
                        "radar_symbols",
                        "Select Symbols",
                        choices=[],
                        selected=[],
                        multiple=True
                    )
                ),
                ui.div(
                    # Hidden markers to ensure reactive outputs are transmitted
                    ui.div(ui.output_text("snapshot_ready"), class_="d-none"),
                    
                    ui.panel_conditional(
                        "input.btn_calc_snapshot == 0",
                        ui.div(ui.h4("Click 'Load Data' to see data"), class_="text-center mt-5")
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
                        choices={m: METRIC_LABELS.get(m, m) for m in ALL_METRICS},
                        selected="rel_strength_z"
                    ),

                    ui.input_select(
                        "rpg_y",
                        "Y Axis",
                        choices={m: METRIC_LABELS.get(m, m) for m in ALL_METRICS},
                        selected="breakout_score_dist"
                    ),

                    # ----- LOG SCALE (single row) -----
                    ui.layout_columns(
                        ui.input_checkbox("rpg_x_log", "Rank X"),
                        ui.input_checkbox("rpg_y_log", "Rank Y"),
                        col_widths=[5, 5]
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

                    ui.input_text(
                        "n_assets_rpg",
                        "Top Volume",
                        value="5",
                        placeholder="e.g. 20",
                        update_on="blur"
                    ),
                    ui.input_selectize(
                        "rpg_symbols",
                        "Compare Symbols",
                        choices=[],
                        selected=MANDATORY_CRYPTO,
                        multiple=True
                    )
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

def market_radar_server(input, output, session, global_interval):
    snapshot_data = reactive.Value(pd.DataFrame())
    rpg_data = reactive.Value(pd.DataFrame())
    selected_symbol_data = reactive.Value(None)
    
    # Track selected symbols for snapshot and RPG
    selected_symbols_radar = reactive.Value(set())
    selected_symbols_rpg = reactive.Value(set())
    
    # Track focus symbol for RPG specifically
    selected_rpg_focus_data = reactive.Value(None)

    logger.log("Market Radar", "INFO", "Server initialized")

    @reactive.effect
    @reactive.event(input.btn_calc_snapshot)
    def _initialize_symbols():
        # Only initialize with mandatory if we don't have symbols yet
        if not selected_symbols_radar.get():
            try:
                n = int(input.n_assets_radar() or 20)
                res = requests.get(f"{API_BASE_URL}/data/universe", params={"top_n": n})
                syms = res.json()["symbols"] if res.status_code == 200 else []
            except Exception as e:
                logger.log("Market Radar", "ERROR", f"Initial symbol sync failed: {e}")
                syms = []
            
            new_syms = set(MANDATORY_CRYPTO).union(syms)
            new_syms = {s for s in new_syms if s not in IGNORED_CRYPTO}
            selected_symbols_radar.set(new_syms)

    @reactive.effect
    @reactive.event(input.n_assets_radar)
    def _update_radar_symbols_list():
        try:
            n = int(input.n_assets_radar() or 20)
            try:
                res = requests.get(f"{API_BASE_URL}/data/universe", params={"top_n": n})
                syms = res.json()["symbols"] if res.status_code == 200 else []
            except Exception as e:
                logger.log("Market Radar", "ERROR", f"Radar volume filter failed: {e}")
                syms = []
            
            new_syms = set(MANDATORY_CRYPTO).union(syms)
            # Remove ignored symbols
            new_syms = {s for s in new_syms if s not in IGNORED_CRYPTO}
            
            # This triggers the effect above to update selected_symbols_radar
            ui.update_text("n_assets_radar", value=str(n)) 
            selected_symbols_radar.set(new_syms)
            # Fetch full universe for dropdown choices
            try:
                res_all = requests.get(f"{API_BASE_URL}/data/universe")
                all_syms = res_all.json()["symbols"] if res_all.status_code == 200 else list(new_syms)
            except:
                all_syms = list(new_syms)
            ui.update_selectize("radar_symbols", choices=all_syms, selected=sorted(list(new_syms)))
        except:
            pass

    @reactive.effect
    @reactive.event(input.n_assets_rpg)
    def _update_rpg_symbols_list():
        try:
            val = input.n_assets_rpg()
            n = int(val) if val else 20
            try:
                res = requests.get(f"{API_BASE_URL}/data/universe", params={"top_n": n})
                syms = res.json()["symbols"] if res.status_code == 200 else []
            except Exception as e:
                logger.log("Market Radar", "ERROR", f"RPG volume filter failed: {e}")
                syms = []
            
            new_syms = set(MANDATORY_CRYPTO).union(syms)
            # Remove ignored symbols
            new_syms = {s for s in new_syms if s not in IGNORED_CRYPTO}
            
            # This triggers the effect above to update selected_symbols_rpg
            ui.update_text("n_assets_rpg", value=str(n))
            selected_symbols_rpg.set(new_syms)
            try:
                res_all = requests.get(f"{API_BASE_URL}/data/universe")
                all_syms = res_all.json()["symbols"] if res_all.status_code == 200 else list(new_syms)
            except:
                all_syms = list(new_syms)
            ui.update_selectize("rpg_symbols", choices=all_syms, selected=sorted(list(new_syms)))
        except:
            pass


    @reactive.effect
    def _populate_symbols_on_tab():
        try:
            if input.main_nav() == "MARKET_RADAR":
                if input.radar_nav() == "tab_rpg":
                    _update_rpg_symbols_list()
                else:
                    _update_radar_symbols_list()
        except:
            pass

    @reactive.effect
    @reactive.event(input.radar_interval, ignore_init=True)
    def _update_symbol_choices():
        try:
            res = requests.get(f"{API_BASE_URL}/data/universe")
            all_syms = res.json()["symbols"] if res.status_code == 200 else []
        except:
            all_syms = []
        curr_sel = sorted(list(selected_symbols_radar.get()))
        ui.update_selectize("radar_symbols", choices=all_syms, selected=curr_sel)
        ui.update_selectize("focus_symbol", choices=[""] + curr_sel)
        ui.update_selectize("rpg_focus_symbol", choices=[""] + curr_sel)

    @reactive.effect
    @reactive.event(input.btn_calc_snapshot)
    def _handle_radar_sync():
        # Strictly gate both symbol population and data syncing behind the button
        try:
            val = input.n_assets_radar()
            n_assets = int(val) if val else 20
        except ValueError:
            n_assets = 20
            
        interval = input.radar_interval()
        
        with ui.Progress(min=0, max=100) as p:
            # 1. Populate Symbols
            p.set(5, message="Refreshing symbols...", detail=f"Fetching top {n_assets} high-volume assets")
            try:
                res = requests.get(f"{API_BASE_URL}/data/universe", params={"top_n": n_assets})
                new_syms = res.json()["symbols"] if res.status_code == 200 else []
            except Exception as e:
                ui.notification_show(f"Market Data Error: {str(e)}", type="error")
                new_syms = []
            syms = sorted(list(set(MANDATORY_CRYPTO).union(new_syms)))
            # Filter ignored
            syms = [s for s in syms if s not in IGNORED_CRYPTO]
            
            selected_symbols_radar.set(set(syms))
            
            # 2. Update UI
            try:
                res_all = requests.get(f"{API_BASE_URL}/data/universe")
                all_syms = res_all.json()["symbols"] if res_all.status_code == 200 else syms
            except:
                all_syms = syms
            ui.update_selectize("radar_symbols", choices=all_syms, selected=syms)
            ui.update_selectize("focus_symbol", choices=[""] + syms)
            ui.update_selectize("rpg_focus_symbol", choices=[""] + syms)
            
            # 3. Sync data for these symbols (if needed/optional)
            p.set(20, message="Syncing data...", detail="Ensuring cache is up-to-date")
            # In Market Radar, we typically load on-demand during calculation, 
            # but we can do a quick check here if desired.
            
            p.set(100, message="Sync complete")

    @reactive.effect
    @reactive.event(input.btn_gen_rpg)
    def _handle_rpg_sync():
        # Strictly gate both symbol population and data syncing behind the button
        try:
            val = input.n_assets_rpg()
            n_assets = int(val) if val else 20
        except ValueError:
            n_assets = 20
            
        interval = input.rpg_interval()
        
        with ui.Progress(min=0, max=100) as p:
            # 1. Populate Symbols
            p.set(5, message="Refreshing symbols...", detail=f"Fetching top {n_assets} high-volume assets")
            try:
                res = requests.get(f"{API_BASE_URL}/data/universe", params={"top_n": n_assets})
                new_syms = res.json()["symbols"] if res.status_code == 200 else []
            except Exception as e:
                ui.notification_show(f"Path Analysis Error: {str(e)}", type="error")
                new_syms = []
            syms = sorted(list(set(MANDATORY_CRYPTO).union(new_syms)))
            selected_symbols_rpg.set(set(syms))
            
            # 2. Update UI
            try:
                res_all = requests.get(f"{API_BASE_URL}/data/universe")
                all_syms = res_all.json()["symbols"] if res_all.status_code == 200 else syms
            except:
                all_syms = syms
            ui.update_selectize("rpg_symbols", choices=all_syms, selected=syms)
            ui.update_selectize("rpg_focus_symbol", choices=[""] + syms)
            ui.update_selectize("focus_symbol", choices=[""] + syms)
            
            # 3. Sync data for these symbols (if needed/optional)
            p.set(20, message="Syncing data...", detail="Ensuring cache is up-to-date")
            # In Market Radar, we typically load on-demand during calculation, 
            # but we can do a quick check here if desired.
            
            p.set(100, message="Sync complete")


    @reactive.effect
    def _sync_focus_symbol():
        symbol = input.focus_symbol()
        if symbol:
            selected_symbol_data.set(symbol.upper())

    @reactive.effect
    def _sync_rpg_focus_symbol():
        symbol = input.rpg_focus_symbol()
        if symbol:
            selected_rpg_focus_data.set(symbol.upper())

    @reactive.effect
    @reactive.event(input.btn_calc_snapshot)
    async def _():
        logger.log("Market Radar", "INFO", "Snapshot calculation triggered")
        try:
            interval = input.radar_interval()
            logger.log("Market Radar", "INFO", f"Using interval: {interval}")
            
            symbols = list(input.radar_symbols() or [])
            if not symbols:
                sym_list = list(selected_symbols_radar.get() or MANDATORY_CRYPTO)
                symbols = sorted(sym_list)
                symbols = [s for s in symbols if s not in IGNORED_CRYPTO]
            logger.log("Market Radar", "INFO", f"Calculating metrics for {len(symbols)} symbols")
            
            if not symbols:
                ui.notification_show("Please select symbols for analysis.", type="warning")
                return

            with ui.Progress(min=0, max=100) as p:
                p.set(message="Analyzing...")
                
                payload = {
                    "symbols": symbols,
                    "interval": interval,
                    "filter_window": input.filter_window()
                }
                
                try:
                    res = requests.post(f"{API_BASE_URL}/market-radar/snapshot", json=payload)
                    if res.status_code == 200:
                        results = res.json()["metrics"]
                        if not results:
                            ui.notification_show("Failed to compute metrics for any symbols.", type="error")
                            return
                        res_df = pd.DataFrame(results)
                        snapshot_data.set(res_df)
                        ui.notification_show("Market Snapshot updated!", type="success")
                    else:
                        ui.notification_show(f"Calculation error: {res.text}", type="error")
                except Exception as e:
                    ui.notification_show(f"API Connection error: {str(e)}", type="error")
                    return
                
        except Exception as e:
            logger.log("Market Radar", "ERROR", f"Snapshot error: {str(e)}")
            ui.notification_show(f"Calculation error: {str(e)}", type="error")

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
        
        # Apply Exclude Symbols filter reactively
        # Now we use radar_symbols as positive inclusion
        selected = list(input.radar_symbols())
        if selected:
            df = df[df['symbol'].isin(selected)]
            
        if input.drop_zeros():
            df = df[df['volatility'] > 1e-9]
            
        x = input.x_axis()
        y = input.y_axis()
        
        if input.x_log() and x in df.columns:
            df[x] = pd.to_numeric(df[x], errors='coerce').rank(pct=True)
        if input.y_log() and y in df.columns:
            df[y] = pd.to_numeric(df[y], errors='coerce').rank(pct=True)
            
        return df

    @reactive.calc
    def snapshot_regression_params():
        plot_df = filtered_snapshot_df()
        if len(plot_df) <= 1:
            return None
            
        x, y = input.x_axis(), input.y_axis()
        
        try:
            # Ensure selected columns are numeric
            reg_x = pd.to_numeric(plot_df[x], errors='coerce').values
            reg_y = pd.to_numeric(plot_df[y], errors='coerce').values
            
            # Filter out NaNs or Infs
            mask = np.isfinite(reg_x) & np.isfinite(reg_y)
            reg_x, reg_y = reg_x[mask], reg_y[mask]
            
            if len(reg_x) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(reg_x, reg_y)
                
                # Generate line points
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

    @render.data_frame
    def snapshot_table():
        return filtered_snapshot_df()

    @render_widget
    def snapshot_chart():
        plot_df = filtered_snapshot_df()
        if plot_df.empty:
            # Return empty figure with message if possible, or just empty
            fig = go.Figure()
            fig.add_annotation(text="No data matching filters or symbols not loaded", showarrow=False, font=dict(size=20))
            fig.update_layout(template="plotly_dark")
            return fig

        x, y, z = input.x_axis(), input.y_axis(), input.z_axis()
        
        # Ensure selected columns are numeric and handle NaNs for Plotly
        for col in [x, y]:
            if col in plot_df.columns:
                plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce').fillna(0)
        if z != "None" and z in plot_df.columns:
            plot_df[z] = pd.to_numeric(plot_df[z], errors='coerce').fillna(0)
            # px.scatter size must be positive
            z_norm = (plot_df[z] - plot_df[z].min()) / (plot_df[z].max() - plot_df[z].min())
            plot_df['z_marker_size'] = z_norm + 1
            
            fig = px.scatter(
                plot_df, x=x, y=y, size='z_marker_size', color=z,
                hover_name='symbol',

                color_continuous_scale='Spectral_r',
                template="plotly_dark",
                custom_data=['symbol']  # Add symbol to custom data for click events
            )
        else:
            fig = px.scatter(
                plot_df, x=x, y=y, hover_name='symbol',

                template="plotly_dark",
                custom_data=['symbol']  # Add symbol to custom data for click events
            )
            fig.update_traces(marker=dict(color='#00F5FF'))
            
        fig.update_layout(
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.12,
                xanchor="center",
                x=0.5
            ),
            xaxis_title=METRIC_LABELS.get(x, x),
            yaxis_title=METRIC_LABELS.get(y, y),
            paper_bgcolor="#0b3d91",
            plot_bgcolor="#0b3d91",
            height=600,
            # width=1470,
            font=dict(family="Space Mono", color="white"),
            clickmode='event+select'  # Enable click events
        )

        fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)", linecolor="white", tickcolor="white")
        fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.3)", zerolinecolor="rgba(255, 255, 255, 0.5)", linecolor="white", tickcolor="white")
                               
        focus_sym = (input.focus_symbol() or "").strip().upper()
        selected_points = None
        unselected_opacity = 0.2
        selection_size = 10

        if focus_sym and focus_sym in plot_df['symbol'].str.upper().values:
            pos_idx = list(plot_df['symbol'].str.upper()).index(focus_sym)
            selected_points = [pos_idx]
            
        if z != "None" and 'z_marker_size' in plot_df.columns:
            size_max = 20
            min_visible = 8

            mask = plot_df['symbol'].astype(str).str.strip().str.upper() == focus_sym

            if mask.any():
                max_val = plot_df['z_marker_size'].max()
                current_val = plot_df.loc[mask, 'z_marker_size'].iloc[0]

                if max_val > 0:
                    sizeref = 2 * max_val / (size_max ** 2)
                    pixel_size = np.sqrt(current_val / sizeref)

                    selection_size = pixel_size if pixel_size >= min_visible else min_visible

        fig.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>' +
                        f'{METRIC_LABELS.get(x, x)}: %{{x}}<br>' +
                        f'{METRIC_LABELS.get(y, y)}: %{{y}}<br>' +
                        '<extra></extra>',
            selectedpoints=selected_points,
            marker=dict(
                line=dict(
                    color="white",
                    width=1
                )
            ),
            selected=dict(
                marker=dict(
                    # color='#FF3B3B',
                    # size=selection_size,
                    opacity=1
                )
            ),
            unselected=dict(
                marker=dict(
                    opacity=unselected_opacity
                )
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
                selected_symbol_data.set(symbol)
                
                # Update Focus Symbol input box
                ui.update_selectize("focus_symbol", selected=symbol)
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

    @reactive.Effect
    def _populate_initial_symbols():
        try:
            res = requests.get(f"{API_BASE_URL}/data/universe")
            all_syms = res.json()["symbols"] if res.status_code == 200 else []
        except:
            all_syms = []
            
        n = int(input.n_assets_radar() or 20)
        try:
            res = requests.get(f"{API_BASE_URL}/data/universe", params={"top_n": n})
            syms = res.json()["symbols"] if res.status_code == 200 else []
        except Exception as e:
            logger.log("Market Radar", "ERROR", f"Initial pop-up sync failed: {e}")
            syms = []
            
        new_syms = set(MANDATORY_CRYPTO).union(syms)
        new_syms = {s for s in new_syms if s not in IGNORED_CRYPTO}
        ui.update_selectize("radar_symbols", choices=all_syms, selected=sorted(list(new_syms)), server=True)
        ui.update_selectize("rpg_symbols", choices=all_syms, selected=sorted(list(new_syms)), server=True)

    @reactive.effect
    @reactive.event(input.btn_gen_rpg)
    async def _():
        logger.log("Market Radar", "INFO", "RPG calculation triggered")
        try:
            interval = input.rpg_interval()
            compare_symbols = list(input.rpg_symbols())
            if not compare_symbols:
                ui.notification_show("Select symbols to compare", type="error")
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
        
        if input.rpg_x_log() and 'X_Value' in df.columns:
            df['X_Value'] = df['X_Value'].rank(pct=True)
        if input.rpg_y_log() and 'Y_Value' in df.columns:
            df['Y_Value'] = df['Y_Value'].rank(pct=True)
            
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
