from shiny import ui, render, reactive, session
import faicons as fa
import pandas as pd
from datetime import datetime, timedelta
from src.config import AVAILABLE_INTERVALS, BENCHMARK_SYMBOL, METRIC_LABELS, MANDATORY_CRYPTO, IGNORED_CRYPTO, DEFAULT_FETCH_INTERVALS, API_BASE_URL
import asyncio
import requests

def data_loader_ui():
    return ui.page_fluid(

        ui.row(

            ui.column(
                6,
                ui.card(
                    ui.card_header("Symbol Selection"),

                    ui.layout_columns(
                        ui.h6("Manual Adding Tickers"),
                        ui.input_text("manual_ticker", None, placeholder="BTCUSDT..."),
                        ui.input_action_button("btn_add", "Add", class_="btn-secondary w-100"),
                        col_widths=[4, 5, 3]
                    ),

                    ui.layout_columns(
                        ui.h6("Filtering Top Liquidity Tickers"),
                        ui.input_numeric("top_n", None, value=50, min=0, max=200, step=10),
                        ui.input_action_button("btn_filter", "Filter", class_="btn-secondary w-100"),
                        col_widths=[4, 5, 3]
                    ),

                    ui.layout_columns(
                        ui.output_ui("selection_info"),
                        ui.div(
                            ui.input_selectize("manage_list", None, choices=[], multiple=True),
                            style="max-height:120px; overflow-y:auto;"
                        )
                    ),

                    ui.input_action_button("btn_reset", "Reset", class_="btn-danger w-100")
                )
            ),

            ui.column(
                6,
                ui.card(
                    ui.card_header("Fetch Configuration"),

                    ui.layout_columns(
                        ui.h6("Intervals"),
                        ui.input_selectize(
                            "intervals",
                            None,
                            choices=AVAILABLE_INTERVALS,
                            selected=DEFAULT_FETCH_INTERVALS,
                            multiple=True
                        ),
                        col_widths=[3, 9]
                    ),
                    ui.layout_columns(
                        # Column 1 → Fetch mode selector
                        ui.input_select("fetch_mode", "Fetch Mode", ["Range", "Limit"]),
                        
                        # Column 2 → Conditional inputs
                        ui.div(
                            ui.panel_conditional(
                                "input.fetch_mode == 'Range'",
                                ui.input_numeric("days_back", "Days Back", value=30, min=1)
                            ),
                            ui.panel_conditional(
                                "input.fetch_mode == 'Limit'",
                                ui.input_numeric("limit", "Limit", value=1000, min=100, step=100)
                            )
                        ),
                        
                        col_widths=[3, 3]
                    ),

                    ui.input_action_button(
                        "btn_execute",
                        "Execute",
                        class_="btn-primary w-100 mt-1"
                    )
                ),
                ui.card(
                    ui.card_header("Data Management"),
                    ui.layout_columns(
                        ui.input_select("delete_interval", "Select Interval", ["ALL"] + AVAILABLE_INTERVALS),
                        ui.input_action_button("btn_delete", "Delete Data", class_="btn-outline-danger w-100 mt-4"),
                        col_widths=[7, 5]
                    )
                )
            )
        ),
        ui.row(
            ui.column(
                12,
                ui.card(
                    ui.card_header("Activity Logs"),
                    ui.output_code("fetch_logs"),
                    ui.output_ui("fetch_progress_ui")
                )
            )
        )
    )


def data_loader_server(input, output, session):
    selected_symbols = reactive.Value(set(MANDATORY_CRYPTO))
    logs = reactive.Value([])
    is_fetching = reactive.Value(False)
    progress = reactive.Value(0.0)

    @reactive.effect
    @reactive.event(input.btn_filter)
    def _():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Fetching top symbols...", detail="Please wait")
            try:
                res = requests.get(f"{API_BASE_URL}/data/universe", params={"top_n": input.top_n()})
                if res.status_code == 200:
                    new_syms = res.json()["symbols"]
                    combined = set(MANDATORY_CRYPTO).union(new_syms)
                    filtered = {s for s in combined if s not in IGNORED_CRYPTO}
                    selected_symbols.set(filtered)
            except Exception as e:
                ui.notification_show(f"Failed to fetch universe: {str(e)}", type="error")

    @reactive.effect
    @reactive.event(input.btn_add)
    def _():
        val = input.manual_ticker()
        if val:
            new_tickers = [s.strip().upper() for s in val.split(',') if s.strip()]
            selected_symbols.set(selected_symbols.get().union(new_tickers))
            ui.update_text("manual_ticker", value="")

    @reactive.effect
    @reactive.event(input.btn_reset)
    def _():
        selected_symbols.set(set([s for s in MANDATORY_CRYPTO if s not in IGNORED_CRYPTO]))

    @reactive.effect
    def _():
        sorted_syms = sorted(list(selected_symbols.get()))
        ui.update_selectize("manage_list", choices=sorted_syms, selected=sorted_syms)

    @reactive.effect
    @reactive.event(input.manage_list)
    def _():
        current = set(input.manage_list())
        if current != selected_symbols.get():
            selected_symbols.set(current)

    @render.ui
    def selection_info():
        return ui.p(f"Active List: {len(selected_symbols.get())}")

    @render.code
    def fetch_logs():
        return "\n".join(logs.get()[-12:])

    @render.ui
    def fetch_progress_ui():
        if is_fetching.get():
            return ui.div(
                ui.div(
                    ui.div(class_="progress-bar progress-bar-striped progress-bar-animated", role="progressbar", style="width: 100%;"),
                    class_="progress"
                ),
                class_="mt-2"
            )
        return ui.div()

    @reactive.effect
    @reactive.event(input.btn_execute)
    async def execute_sync():
        if not input.intervals():
            ui.notification_show("Select interval", type="error")
            return
        if not selected_symbols.get():
            ui.notification_show("Select symbols", type="error")
            return

        logs.set([])
        is_fetching.set(True)

        all_syms = sorted(list(selected_symbols.get()))
        intervals = input.intervals()

        payload = {
            "symbols": all_syms,
            "intervals": list(intervals),
            "mode": input.fetch_mode(),
            "days_back": input.days_back(),
            "limit": input.limit()
        }

        try:
            res = requests.post(f"{API_BASE_URL}/data/fetch", json=payload)
            if res.status_code != 200:
                ui.notification_show(f"Failed to start fetch: {res.text}", type="error")
                is_fetching.set(False)
                return
        except Exception as e:
            ui.notification_show(f"Error connecting to API: {str(e)}", type="error")
            is_fetching.set(False)
            return

        while True:
            try:
                res = requests.get(f"{API_BASE_URL}/data/fetch-status")
                if res.status_code == 200:
                    data = res.json()
                    status_logs = [f"[{log['timestamp']}] {log['message']}" for log in data.get('logs', [])]
                    logs.set(status_logs)
                    
                    if not data.get('in_progress', False):
                        break
            except:
                break
            await asyncio.sleep(2)
            await reactive.flush()

        is_fetching.set(False)
        ui.notification_show("Data Fetch Complete", type="message")

    @reactive.effect
    @reactive.event(input.btn_delete)
    def _():
        m = ui.modal(
            ui.p(f"Are you sure you want to delete cached data for: {input.delete_interval()}?"),
            ui.p("This action cannot be undone.", class_="text-danger"),
            title="Confirm Deletion",
            footer=ui.div(
                ui.modal_button("Cancel"),
                ui.input_action_button("btn_confirm_delete", "Yes, Delete", class_="btn-danger")
            ),
            easy_close=True
        )
        ui.modal_show(m)

    @reactive.effect
    @reactive.event(input.btn_confirm_delete)
    def _():
        ui.modal_remove()
        interval = input.delete_interval()
        
        try:
            res = requests.delete(f"{API_BASE_URL}/data/cache/{interval}")
            if res.status_code == 200:
                count = res.json().get("count", 0)
                ui.notification_show(f"Successfully deleted {count} files for {interval}", type="message")
                
                # Update logs
                now_str = datetime.now().strftime('%H:%M:%S')
                current_logs = logs.get()
                current_logs.append(f"[{now_str}] Deleted {count} files for {interval}")
                logs.set(current_logs[:])
            else:
                ui.notification_show(f"Deletion error: {res.text}", type="error")
        except Exception as e:
            ui.notification_show(f"Deletion error: {str(e)}", type="error")
