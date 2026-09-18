from shiny import ui, render, reactive, session
import faicons as fa
import pandas as pd
from datetime import datetime, timedelta
from src.data import BinanceFuturesFetcher, DataManager
from src.config import AVAILABLE_INTERVALS, BENCHMARK_SYMBOL, METRIC_LABELS, MANDATORY_CRYPTO, IGNORED_CRYPTO, DEFAULT_FETCH_INTERVALS
from src.shared_state import get_manager, get_engine
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
    fetcher = BinanceFuturesFetcher()
    manager = get_manager()
    
    selected_symbols = reactive.Value(set(MANDATORY_CRYPTO))
    logs = reactive.Value([])
    is_fetching = reactive.Value(False)
    progress = reactive.Value(0.0)

    @reactive.effect
    @reactive.event(input.btn_filter)
    def _():
        with ui.Progress(min=1, max=15) as p:
            p.set(message="Fetching top symbols...", detail="Please wait")
            new_syms = fetcher.get_top_volume_symbols(top_n=input.top_n())
            # Ensure MANDATORY_CRYPTO are always included when filtering, but remove IGNORED_CRYPTO
            combined = set(MANDATORY_CRYPTO).union(new_syms)
            filtered = {s for s in combined if s not in IGNORED_CRYPTO}
            selected_symbols.set(filtered)

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
            val = int(progress.get() * 100)
            return ui.div(
                ui.div(
                    ui.div(class_="progress-bar", role="progressbar", style=f"width: {val}%;", aria_valuenow=val, aria_valuemin=0, aria_valuemax=100),
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
        progress.set(0.0)

        all_syms = sorted(list(selected_symbols.get()))
        intervals = input.intervals()
        total = len(all_syms) * len(intervals)
        count_done = 0
        current_logs = []

        for sym in all_syms:
            for inter in intervals:
                now_str = datetime.now().strftime('%H:%M:%S')
                current_logs.append(f"[{now_str}] {sym} {inter}...")
                logs.set(current_logs[:])
                
                try:
                    first_ts, last_ts = manager.get_cache_range(sym, inter)
                    
                    if input.fetch_mode() == "Range":
                        end_t = datetime.now()
                        requested_start = end_t - timedelta(days=input.days_back())
                        
                        # Generate UTC naive counterparts for comparison against cache values
                        end_t_utc = pd.to_datetime(end_t.timestamp(), unit='s')
                        req_start_utc = pd.to_datetime(requested_start.timestamp(), unit='s')
                        
                        dfs_to_fetch = []
                        
                        # Case 1: No cache exists -> Full Range
                        if not first_ts:
                            df_full = fetcher.fetch_history(sym, inter, start_time=requested_start, end_time=end_t)
                            if not df_full.empty:
                                dfs_to_fetch.append(df_full)
                                mode_info = "Full Range"
                            else:
                                mode_info = "No Data"
                        else:
                            mode_info = "Incremental"
                            # Case 2: Forward Gap
                            if last_ts < end_t_utc - timedelta(minutes=5):
                                start_ts_ms = int(pd.Timestamp(last_ts).tz_localize('UTC').timestamp() * 1000) + 1
                                df_fwd = fetcher.fetch_history(sym, inter, start_time=start_ts_ms, end_time=end_t)
                                if not df_fwd.empty:
                                    dfs_to_fetch.append(df_fwd)
                                    mode_info += " (Forward)"
                            
                            # Case 3: Backward Gap
                            if first_ts > req_start_utc + timedelta(minutes=5):
                                end_ts_ms = int(pd.Timestamp(first_ts).tz_localize('UTC').timestamp() * 1000) - 1
                                df_bwd = fetcher.fetch_history(sym, inter, start_time=requested_start, end_time=end_ts_ms)
                                if not df_bwd.empty:
                                    dfs_to_fetch.append(df_bwd)
                                    mode_info += " (Backward)"

                        if dfs_to_fetch:
                            final_df = pd.concat(dfs_to_fetch)
                            manager.append_data(sym, inter, final_df)
                            current_logs.append(f"  > {mode_info}: Added {len(final_df)} candles")
                        else:
                            current_logs.append(f"  > Up to date")
                    else:
                        # Limit mode: Always fetch latest N
                        df = fetcher.fetch_candles(sym, inter, limit=input.limit())
                        if not df.empty:
                            manager.append_data(sym, inter, df)
                            current_logs.append(f"  > Limit: Synced {len(df)} candles")
                        else:
                            current_logs.append(f"  > Up to date / No data")
                except requests.exceptions.HTTPError as e:
                    status_code = e.response.status_code
                    if status_code in [418, 429]:
                        current_logs.append(f"  ! CRITICAL: Rate Limit reached ({status_code}). Aborting.")
                        logs.set(current_logs[:])
                        is_fetching.set(False)
                        ui.notification_show(f"Aborted: Rate Limit ({status_code})", type="error")
                        return
                    current_logs.append(f"  ! HTTP Error {status_code}: {str(e)}")
                except Exception as e:
                    current_logs.append(f"  ! Error: {str(e)}")

                count_done += 1
                progress.set(count_done / total)
                logs.set(current_logs[:])
                await asyncio.sleep(0.01)  # Yield to event loop for UI updates
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
            count = manager.delete_data(interval)
            ui.notification_show(f"Successfully deleted {count} files for {interval}", type="message")
            
            # Update logs
            now_str = datetime.now().strftime('%H:%M:%S')
            current_logs = logs.get()
            current_logs.append(f"[{now_str}] Deleted {count} files for {interval}")
            logs.set(current_logs[:])
        except Exception as e:
            ui.notification_show(f"Deletion error: {str(e)}", type="error")
