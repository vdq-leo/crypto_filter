import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import src and ml_engine
sys.path.append(str(Path(__file__).parent))

from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import faicons as fa

from shinywidgets import output_widget, render_widget
from src.config import APP_TITLE, APP_ICON, WELCOME_TITLE, WELCOME_TEXT, SIDEBAR_INFO, THEME, BG_COLOR, TRADINGVIEW_URL
import requests
import webbrowser

from modules.data_loader import data_loader_ui, data_loader_server
from modules.market_radar import market_radar_ui, market_radar_server
from modules.predictive import predictive_ui, predictive_server
from modules.multivariate_analysis import multivariate_analysis_ui, multivariate_analysis_server
from modules.activity_logs import activity_logs_ui, activity_logs_server
from modules.symbol_diagnostics import symbol_diagnostics_ui, symbol_diagnostics_server
from modules.pair_radar import pair_radar_ui, pair_radar_server
from src.config import BENCHMARK_SYMBOL, API_BASE_URL
from datetime import datetime, timedelta

# UI definition
app_ui = ui.page_navbar(
    ui.head_content(
        ui.include_css(Path(__file__).parent / "www" / "custom.css"),
        ui.tags.script("""
            document.addEventListener("keydown", function(e) {
                const decomp_n_assets_box = document.querySelector('input[id$="decomp_n_assets"]');
                if (decomp_n_assets_box && document.activeElement === decomp_n_assets_box && e.key === "Enter") {
                    document.activeElement.blur();
                }
            });
            $(document).on('bslib.card.expand', function(event) {
                Shiny.setInputValue('card_expanded', true);
            });
            $(document).on('bslib.card.collapse', function(event) {
                Shiny.setInputValue('card_expanded', false);
            });
        """)
    ),

    ui.nav_panel("HOME", 
        ui.div(
            ui.div(
                ui.h1("QUANT_TOOLS", class_="main-header"),
                ui.HTML(
                    '<p class="lead mb-5" style="color:white;">> Institutional Predictive Analytics for '
                    '<span style="font-weight:bold ; font-size: 1.2em; text-decoration: underline; text-underline-offset: 0.5em; color:#FCC780;">BINANCE_PERPETUAL</span></p>'
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Diagnostics"),
                        ui.p("Deep Quantitative Symbol Diagnostics"),
                        ui.input_action_button("go_diagnostics", "open_diagnostics", class_="btn-primary w-100")
                    ),
                    ui.card(
                        ui.card_header("Market Radar"),
                        ui.p("Real-time relative performance & metrics tracking."),
                        ui.input_action_button("go_radar", "open_radar", class_="btn-primary w-100")
                    ),
                    ui.card(
                        ui.card_header("Multivariate"),
                        ui.p("Cross-asset correlation & Multivariate construction."),
                        ui.input_action_button("go_multivariate", "multivariate_analysis", class_="btn-primary w-100")
                    ),
                    ui.card(
                        ui.card_header("Predictive"),
                        ui.p("ML forecasting with meta-labeling verification protocols."),
                        ui.input_action_button("go_predictive", "explore_models", class_="btn-primary w-100")
                    ),
                    ui.card(
                        ui.card_header("Pair Radar"),
                        ui.p("Statistical arbitrage & pair trading analytics."),
                        ui.input_action_button("go_pair_radar", "open_pair_radar", class_="btn-primary w-100")
                    ),
                    col_widths=[3,3,3,3]
                ),
                class_="hero-container text-center py-5"
            ),
            class_="container mt-5"
        )
    ),

    ui.nav_panel("DATA_MANAGER", data_loader_ui()),
    ui.nav_panel("DIAGNOSTICS", symbol_diagnostics_ui()),
    ui.nav_panel("MARKET_RADAR", market_radar_ui()),
    ui.nav_panel("MULTIVARIATE", multivariate_analysis_ui()),
    ui.nav_panel("PAIR_RADAR", pair_radar_ui()),
    ui.nav_panel("PREDICTIVE", predictive_ui()),
    ui.nav_panel("ACTIVITY_LOGS", activity_logs_ui()),

    # ui.nav_spacer(),
    # ui.nav_control(ui.output_ui("data_status_")),
    ui.nav_spacer(),
    ui.nav_control(
        ui.div(
            ui.div(
                ui.input_selectize(
                    "quick_symbol",
                    None,
                    choices=[],
                    multiple=True,
                    options={"placeholder": "Launch Assets"}
                ),
                class_="flex-grow-1"
            ),
            ui.div(
                ui.input_action_button(
                    "btn_quick_go",
                    fa.icon_svg("paper-plane"),
                    class_="btn-primary btn-sm"  # smaller button
                ),
                class_="flex-shrink-0",
                style="margin-top: -14px;"  # move slightly upwards
            ),
            class_="d-flex align-items-center gap-2"
        )
    ),
    id="main_nav"
)


def server(input, output, session):
    # Shared global state if needed
    global_interval = reactive.Value("1h")
    
    diag_data = reactive.Value({})
    data_info = reactive.Value({"global": {"oldest": "-", "latest": "-"}})
    
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
        with ui.Progress(min=0, max=1) as p:
            p.set(0, message="Initializing Market Data...")
            try:
                res = requests.get(f"{API_BASE_URL}/data/universe")
                all_syms = res.json()["symbols"] if res.status_code == 200 else []
            except:
                all_syms = []
            p.set(1, message="Populating Global Selectors...")
            ui.update_selectize("quick_symbol", choices=all_syms, server=True)

        # Set benchmark/global timestamps once
        try:
            interval_val = input.diag_interval()
        except:
            interval_val = "1h"
        global_ts = get_timestamps(BENCHMARK_SYMBOL, interval_val)
        data_info.set({"global": global_ts})
    
    @render.ui
    def data_status_():
        d = data_info.get()
        return ui.HTML(f"""
            <div style="font-size: 0.7rem; opacity: 0.7; color: white;">
                <div>Global Data: {d['global']['oldest']} - {d['global']['latest']}</div>
                <div>Current Time: {(datetime.utcnow() + timedelta(hours=7)).strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
        """)
    
    @reactive.Effect
    @reactive.event(input.go_predictive)
    def _go_predictive():
        ui.update_navset("main_nav", selected="PREDICTIVE")

    @reactive.Effect
    @reactive.event(input.go_radar)
    def _go_radar():
        ui.update_navset("main_nav", selected="MARKET_RADAR")

    @reactive.Effect
    @reactive.event(input.go_multivariate)
    def _go_multivariate():
        ui.update_navset("main_nav", selected="MULTIVARIATE")

    @reactive.Effect
    @reactive.event(input.go_diagnostics)
    def _go_diagnostics():
        ui.update_navset("main_nav", selected="DIAGNOSTICS")

    @reactive.Effect
    @reactive.event(input.go_pair_radar)
    def _go_pair_radar():
        ui.update_navset("main_nav", selected="PAIR_RADAR")

    @reactive.Effect
    @reactive.event(input.btn_quick_go)
    def _quick_link():
        symbols = input.quick_symbol()
        if not symbols:
            return
            
        try:
            interval = input.radar_interval()
        except:
            interval = "1h"

        # TradingView interval mapping
        tv_intervals = {
            '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
            '1h': '60', '2h': '120', '4h': '240', '6h': '360', '8h': '480', '12h': '720',
            '1d': 'D', '3d': '3D', '1w': 'W'
        }
        tv_int = tv_intervals.get(interval, '60')

        if isinstance(symbols, (list, tuple)):
            for sym in symbols:
                url = f"{TRADINGVIEW_URL}?symbol=BINANCE:{sym}.P&interval={tv_int}"
                webbrowser.open(url)
        
        # Clear selection after opening
        ui.update_selectize("quick_symbol", selected=[])

    data_loader_server(input, output, session)
    market_radar_server(input, output, session, global_interval)
    predictive_server(input, output, session)
    multivariate_analysis_server(input, output, session)
    pair_radar_server(input, output, session, global_interval)
    symbol_diagnostics_server(input, output, session, global_interval)
    activity_logs_server(input, output, session)

app = App(app_ui, server)
