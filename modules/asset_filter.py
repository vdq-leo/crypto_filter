from shiny import ui, render, reactive
import pandas as pd
import faicons as fa
from src.config import API_BASE_URL
import requests
import logging
import asyncio

logger = logging.getLogger(__name__)

DEFAULT_TYPES = ["COIN", "INDEX"]
DEFAULT_SUBTYPES = [
    "-", "AI", "Alpha", "Chinese", "Crypto", "DeFi", "Gaming", 
    "Index", "Infrastructure", "Layer-1", "Layer-2", "Meme", 
    "Metaverse", "NFT", "Payment", "PoW", "RWA", "Storage"
]

def asset_filter_ui(id: str = "asset_filter"):
    """UI for the Asset Filter module - Clean, simple, and functional."""
    return ui.layout_sidebar(
        ui.sidebar(
            ui.h4("GLOBAL UNIVERSE", class_="fw-bold mb-2 text-light"),
            ui.hr(class_="my-2"),
            
            # Asset Type Section
            ui.div(
                ui.div(
                    ui.span("Asset Type", class_="fw-bold small text-uppercase text-light"),
                    ui.div(
                        ui.input_action_button(
                            "asset_filter_types_all", 
                            "Select All", 
                            class_="btn-outline-primary btn-sm py-0 px-2", 
                            style="font-size: 0.72rem;"
                        ),
                        ui.input_action_button(
                            "asset_filter_types_clear", 
                            "Clear", 
                            class_="btn-outline-secondary btn-sm py-0 px-2", 
                            style="font-size: 0.72rem;"
                        ),
                        class_="d-flex gap-1"
                    ),
                    class_="d-flex justify-content-between align-items-center mb-1"
                ),
                ui.input_selectize(
                    "asset_filter_types",
                    None,
                    choices=DEFAULT_TYPES,
                    multiple=True,
                    options={"placeholder": "All Types (e.g. COIN, INDEX)"}
                ),
                class_="mb-3"
            ),
            
            # Asset Sub-Type Section
            ui.div(
                ui.div(
                    ui.span("Asset Sub-Type", class_="fw-bold small text-uppercase text-light"),
                    ui.div(
                        ui.input_action_button(
                            "asset_filter_subtypes_all", 
                            "Select All", 
                            class_="btn-outline-primary btn-sm py-0 px-2", 
                            style="font-size: 0.72rem;"
                        ),
                        ui.input_action_button(
                            "asset_filter_subtypes_clear", 
                            "Clear", 
                            class_="btn-outline-secondary btn-sm py-0 px-2", 
                            style="font-size: 0.72rem;"
                        ),
                        class_="d-flex gap-1"
                    ),
                    class_="d-flex justify-content-between align-items-center mb-1"
                ),
                ui.input_selectize(
                    "asset_filter_subtypes",
                    None,
                    choices=DEFAULT_SUBTYPES,
                    multiple=True,
                    options={"placeholder": "All Sub-Types (or remove exceptions)"}
                ),
                class_="mb-3"
            ),
            
            ui.hr(class_="my-2"),
            
            # Volume Ranking Section
            ui.div(
                ui.span("Volume Ranking (Top N)", class_="fw-bold small text-uppercase text-light mb-1 d-block"),
                ui.input_numeric(
                    "asset_filter_top_n",
                    None,
                    value=100,
                    min=10,
                    max=500,
                    step=10
                ),
                ui.div(
                    ui.input_switch(
                        "asset_filter_bottom_vol",
                        "Fetch Lowest Volume Instead",
                        value=False
                    ),
                    class_="mt-2"
                ),
                class_="mb-3"
            ),
            
            ui.hr(class_="my-2"),
            
            # Apply Action
            ui.input_action_button(
                "asset_filter_btn_apply",
                "Apply Global Filter",
                class_="btn-primary w-100 py-2 fw-bold shadow-sm",
                icon=fa.icon_svg("filter")
            ),
            
            ui.output_ui("asset_filter_status"),
            
            width=280
        ),
        ui.card(
            ui.card_header(
                ui.div(
                    ui.div(
                        fa.icon_svg("table-list"),
                        ui.span("Filtered Global Asset Universe", class_="fw-bold"),
                        class_="d-flex align-items-center gap-2"
                    ),
                    ui.div(
                        ui.input_action_button(
                            "asset_table_select_all",
                            "Select All",
                            class_="btn-primary btn-sm py-1 px-3 fw-bold",
                            style="font-size: 0.78rem;"
                        ),
                        ui.input_action_button(
                            "asset_table_deselect_all",
                            "Deselect All",
                            class_="btn-secondary btn-sm py-1 px-3 fw-bold",
                            style="font-size: 0.78rem;"
                        ),
                        ui.output_ui("asset_filter_badge"),
                        class_="d-flex align-items-center gap-2"
                    ),
                    class_="d-flex justify-content-between align-items-center w-100"
                )
            ),
            ui.div(
                ui.div(
                    ui.span("Active Universe (Select Box):", class_="small fw-bold text-uppercase text-light me-2 text-nowrap pt-1"),
                    ui.div(
                        ui.input_selectize(
                            "asset_filter_selected_symbols",
                            None,
                            choices=[],
                            multiple=True,
                            width="100%",
                            options={
                                "placeholder": "Search / pick / remove assets to use globally...",
                                "plugins": ["remove_button"],
                                "maxOptions": 2000
                            }
                        ),
                        class_="flex-grow-1"
                    ),
                    class_="d-flex align-items-center gap-2 px-3 py-2 w-100"
                ),
                style="background-color: var(--panel-blue); border-bottom: 2px solid var(--border-color);"
            ),
            ui.output_data_frame("asset_filter_table"),
            full_screen=True,
            class_="h-100 shadow-sm"
        ),
    )

def asset_filter_server(*args, **kwargs):
    """Server logic for the Asset Filter module."""
    if len(args) == 5 and isinstance(args[0], str):
        _, input, output, session, global_universe = args
    elif len(args) == 4:
        input, output, session, global_universe = args
    else:
        input = kwargs.get("input")
        output = kwargs.get("output")
        session = kwargs.get("session")
        global_universe = kwargs.get("global_universe")

    all_types = reactive.Value(DEFAULT_TYPES)
    all_subtypes = reactive.Value(DEFAULT_SUBTYPES)
    meta_store = reactive.Value({})
    filtered_df = reactive.Value(pd.DataFrame())
    status_msg = reactive.Value("")

    # Mutable sync state to prevent feedback loops between Table, Selectize, and Global Universe
    sync_state = {"bulk_in_progress": False}

    @reactive.effect
    async def _fetch_metadata():
        try:
            res = requests.get(f"{API_BASE_URL}/data/universe/meta", timeout=10)
            if res.status_code == 200:
                data = res.json()
                t_list = data.get("types", []) or DEFAULT_TYPES
                st_list = data.get("subtypes", []) or DEFAULT_SUBTYPES
                all_types.set(t_list)
                all_subtypes.set(st_list)
                meta_store.set(data.get("symbol_meta", {}))
                ui.update_selectize("asset_filter_types", choices=t_list)
                ui.update_selectize("asset_filter_subtypes", choices=st_list)
                if filtered_df.get().empty:
                    await do_fetch()
        except Exception as e:
            logger.error(f"Failed to fetch universe meta: {e}")

    # Select All / Clear Handlers for Asset Types
    @reactive.effect
    @reactive.event(input.asset_filter_types_all)
    def _select_all_types():
        t = all_types.get() or DEFAULT_TYPES
        ui.update_selectize("asset_filter_types", choices=t, selected=t)

    @reactive.effect
    @reactive.event(input.asset_filter_types_clear)
    def _clear_types():
        t = all_types.get() or DEFAULT_TYPES
        ui.update_selectize("asset_filter_types", choices=t, selected=[])

    # Select All / Clear Handlers for Asset Sub-Types
    @reactive.effect
    @reactive.event(input.asset_filter_subtypes_all)
    def _select_all_subtypes():
        st = all_subtypes.get() or DEFAULT_SUBTYPES
        ui.update_selectize("asset_filter_subtypes", choices=st, selected=st)

    @reactive.effect
    @reactive.event(input.asset_filter_subtypes_clear)
    def _clear_subtypes():
        st = all_subtypes.get() or DEFAULT_SUBTYPES
        ui.update_selectize("asset_filter_subtypes", choices=st, selected=[])

    @render.ui
    def asset_filter_status():
        msg = status_msg.get()
        if not msg:
            return ui.div()
        color = "text-success" if msg.startswith("✓") else "text-danger"
        return ui.p(msg, class_=f"{color} fw-bold mt-2 small text-center")

    @render.ui
    def asset_filter_badge():
        df = filtered_df.get()
        total = len(df) if (not df.empty and "Symbol" in df.columns) else 0
        syms = global_universe.get() or []
        count = len(syms)
        if total == 0:
            return ui.span("No Assets", class_="badge bg-secondary fs-6 px-3 py-1 shadow-sm")
        elif count == total:
            return ui.span(f"✓ All {count} Assets Active Globally", class_="badge bg-primary fs-6 px-3 py-1 shadow-sm")
        elif count == 0:
            return ui.span(f"0 / {total} Selected (None Active)", class_="badge bg-warning text-dark fs-6 px-3 py-1 shadow-sm")
        else:
            return ui.span(f"✓ {count} / {total} Assets Active Globally", class_="badge bg-info text-dark fs-6 px-3 py-1 shadow-sm")

    @render.data_frame
    def asset_filter_table():
        df = filtered_df.get()
        if df.empty:
            df = pd.DataFrame(columns=["#", "Symbol", "Type", "Sub-Types", "Volume (24h)"])
        return render.DataGrid(
            df,
            width="100%",
            height=None,
            filters=False,
            summary=False,
            selection_mode="rows",
            styles=[
                {"cols": [0], "style": {"width": "60px", "max-width": "70px", "text-align": "center"}},
                {"cols": [1], "style": {"font-weight": "600", "width": "150px"}},
                {"cols": [2], "style": {"width": "100px", "text-align": "center"}},
                {"cols": [4], "style": {"width": "140px", "text-align": "right", "font-weight": "600"}},
            ]
        )

    async def do_fetch():
        sync_state["bulk_in_progress"] = True
        try:
            top_n = input.asset_filter_top_n()
        except Exception:
            top_n = 100
        try:
            bottom_vol = input.asset_filter_bottom_vol()
        except Exception:
            bottom_vol = False
        try:
            types = input.asset_filter_types()
        except Exception:
            types = None
        try:
            subtypes = input.asset_filter_subtypes()
        except Exception:
            subtypes = None

        params = {
            "top_n": top_n or 100,
            "bottom": str(bottom_vol).lower() if bottom_vol is not None else "false"
        }
        if types:
            params["types"] = list(types)
        if subtypes:
            params["subtypes"] = list(subtypes)

        try:
            res = requests.get(f"{API_BASE_URL}/data/universe", params=params, timeout=15)
            if res.status_code == 200:
                payload = res.json()
                items = payload.get("items", [])
                syms = payload.get("symbols", [])
                
                rows = []
                for i, item in enumerate(items, 1):
                    rows.append({
                        "#": i,
                        "Symbol": item.get("symbol"),
                        "Type": item.get("type", "COIN"),
                        "Sub-Types": ", ".join(item.get("subtypes", [])) if item.get("subtypes") else "-",
                        "Volume (24h)": item.get("volume_formatted", "$0")
                    })

                if rows:
                    df = pd.DataFrame(rows)
                else:
                    df = pd.DataFrame(columns=["#", "Symbol", "Type", "Sub-Types", "Volume (24h)"])

                filtered_df.set(df)
                global_universe.set(syms)
                status_msg.set(f"✓ {len(syms)} symbols loaded.")
                
                # Default: select all rows in table and selectize
                try:
                    ui.update_selectize("asset_filter_selected_symbols", choices=syms, selected=syms)
                except Exception:
                    pass

                try:
                    await asset_filter_table.update_cell_selection("all")
                except Exception:
                    pass

                try:
                    ui.update_selectize("quick_symbol", choices=syms, selected=syms)
                except Exception:
                    pass
            else:
                err = f"API Error {res.status_code}: {res.text}"
                logger.error(err)
                status_msg.set(f"✗ {err}")
                ui.notification_show(err, type="error")
        except Exception as e:
            logger.error(f"Error filtering universe: {e}")
            status_msg.set(f"✗ Error: {e}")
            ui.notification_show(f"Error filtering universe: {e}", type="error")
        finally:
            await asyncio.sleep(0.5)
            sync_state["bulk_in_progress"] = False

    @reactive.effect
    @reactive.event(input.asset_filter_btn_apply, ignore_init=False)
    async def _on_apply():
        await do_fetch()

    # Handlers for Table Select All / Deselect All
    @reactive.effect
    @reactive.event(input.asset_table_select_all)
    async def _on_table_select_all():
        df = filtered_df.get()
        if not df.empty and "Symbol" in df.columns:
            all_syms = df["Symbol"].tolist()
            sync_state["bulk_in_progress"] = True
            try:
                global_universe.set(all_syms)
                try:
                    ui.update_selectize("asset_filter_selected_symbols", choices=all_syms, selected=all_syms)
                except Exception:
                    pass
                try:
                    ui.update_selectize("quick_symbol", choices=all_syms, selected=all_syms)
                except Exception:
                    pass
                try:
                    await asset_filter_table.update_cell_selection("all")
                except Exception:
                    pass
            finally:
                await asyncio.sleep(0.5)
                sync_state["bulk_in_progress"] = False

    @reactive.effect
    @reactive.event(input.asset_table_deselect_all)
    async def _on_table_deselect_all():
        sync_state["bulk_in_progress"] = True
        try:
            df = filtered_df.get()
            all_syms = df["Symbol"].tolist() if (not df.empty and "Symbol" in df.columns) else []
            global_universe.set([])
            try:
                ui.update_selectize("asset_filter_selected_symbols", choices=all_syms, selected=[])
            except Exception:
                pass
            try:
                ui.update_selectize("quick_symbol", choices=[], selected=[])
            except Exception:
                pass
            try:
                await asset_filter_table.update_cell_selection(None)
            except Exception:
                pass
        finally:
            await asyncio.sleep(0.5)
            sync_state["bulk_in_progress"] = False

    # Sync Select Box (selectize) changes to Table and global_universe
    @reactive.effect
    @reactive.event(input.asset_filter_selected_symbols)
    async def _on_selectize_change():
        if sync_state["bulk_in_progress"]:
            return
        selected = list(input.asset_filter_selected_symbols() or [])
        with reactive.isolate():
            current = global_universe.get() or []
            df = filtered_df.get()
        if set(selected) != set(current):
            global_universe.set(selected)
            if not df.empty and "Symbol" in df.columns:
                sym_list = df["Symbol"].tolist()
                rows = [i for i, s in enumerate(sym_list) if s in selected]
                try:
                    await asset_filter_table.update_cell_selection({"type": "row", "rows": tuple(rows)})
                except Exception:
                    pass
            try:
                ui.update_selectize("quick_symbol", choices=selected, selected=selected)
            except Exception:
                pass

    # Sync Table Row Selection to Select Box and global_universe
    @reactive.effect
    @reactive.event(asset_filter_table.cell_selection, ignore_init=True)
    async def _sync_table_selection_to_global():
        if sync_state["bulk_in_progress"]:
            return

        with reactive.isolate():
            df = filtered_df.get()
            current = global_universe.get() or []

        if df.empty or "Symbol" not in df.columns:
            return

        sel = asset_filter_table.cell_selection()
        all_syms = df["Symbol"].tolist()

        if sel is None:
            if current:
                global_universe.set([])
                try:
                    ui.update_selectize("asset_filter_selected_symbols", choices=all_syms, selected=[])
                except Exception:
                    pass
                try:
                    ui.update_selectize("quick_symbol", choices=[], selected=[])
                except Exception:
                    pass
            return

        rows = sel.get("rows", ())
        if not rows:
            if current:
                global_universe.set([])
                try:
                    ui.update_selectize("asset_filter_selected_symbols", choices=all_syms, selected=[])
                except Exception:
                    pass
                try:
                    ui.update_selectize("quick_symbol", choices=[], selected=[])
                except Exception:
                    pass
            return

        valid_rows = [r for r in rows if r < len(df)]
        selected_symbols = df.iloc[valid_rows]["Symbol"].tolist()
        if set(selected_symbols) != set(current):
            global_universe.set(selected_symbols)
            try:
                ui.update_selectize("asset_filter_selected_symbols", choices=all_syms, selected=selected_symbols)
            except Exception:
                pass
            try:
                ui.update_selectize("quick_symbol", choices=selected_symbols, selected=selected_symbols)
            except Exception:
                pass
