from shiny import ui, render, reactive
import faicons as fa
import requests
import pandas as pd
from src.config import API_BASE_URL

def activity_logs_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Activity Logs"),
            ui.input_action_button("btn_clear_logs", "Clear Logs", class_="btn-warning w-100 mb-3"),
            ui.input_select("log_level_filter", "Filter by Level", 
                          choices=["ALL", "INFO", "WARNING", "ERROR", "DEBUG"], 
                          selected="ALL"),
            ui.input_numeric("log_limit", "Max Entries", value=100, min=10, max=1000)
        ),
        ui.div(
            ui.card(
                ui.card_header(
                    ui.div(
                        fa.icon_svg("list-check"),
                        " Activity Logs",
                        class_="d-flex align-items-center gap-2"
                    )
                ),
                ui.output_data_frame("logs_table"),
                full_screen=True
            )
        )
    )

def activity_logs_server(input, output, session):
    
    @reactive.effect
    @reactive.event(input.btn_clear_logs)
    def _():
        try:
            requests.delete(f"{API_BASE_URL}/logs")
        except:
            pass
    
    @render.data_frame
    def logs_table():
        limit = input.log_limit()
        level_filter = input.log_level_filter()
        
        try:
            res = requests.get(f"{API_BASE_URL}/logs", params={"limit": limit, "level": level_filter})
            if res.status_code == 200:
                data = res.json()
                if not data:
                    return pd.DataFrame()
                df = pd.DataFrame(data)
                return df
        except:
            pass
        return pd.DataFrame()
