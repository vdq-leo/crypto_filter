# Implementation Plan: Centralized Asset Filter

## Objective
Create a centralized "Asset Filter" tab to globally manage and filter the symbol universe using criteria like `underlyingType`, `underlyingSubType`, and `Volume`. This globally filtered universe will sync automatically across all other tools (Market Radar, Diagnostics, Predictive, etc.) to replace their isolated, redundant asset selection menus.

## Phase 1: API & Data Layer Enhancements
1. **`src/data.py` (Data Fetcher)**
   - Enhance the `get_top_volume_symbols` method to accept new filter parameters: `types` (list) and `subtypes` (list).
   - Extract `underlyingType` and `underlyingSubType` from the Binance `/fapi/v1/exchangeInfo` response.
   - Return a filtered list of symbols based on the intersection of volume ranking and type/subtype filters.
   - Implement caching for `exchangeInfo` to prevent redundant API calls when the user interacts with the UI.

2. **`api/routers/data.py` (API Router)**
   - Update the `/universe` endpoint to accept `types` and `subtypes` as query parameters.
   - Fetch the unique list of available `types` and `subtypes` from the data fetcher so the frontend can populate its filter dropdowns dynamically.

## Phase 2: Frontend Global State Setup
1. **`app.py` (Main Application State)**
   - Initialize a global reactive value: `global_universe = reactive.Value([])`.
   - Pass `global_universe` to all module servers (e.g., `market_radar_server(..., global_universe)`).

## Phase 3: Create the Asset Filter Module
1. **`modules/asset_filter.py`**
   - Create a new UI module (`asset_filter_ui` and `asset_filter_server`).
   - **UI Elements:**
     - Multi-select dropdown for `underlyingType` (e.g., COIN, COMMODITY, INDEX).
     - Multi-select dropdown for `underlyingSubType` (e.g., Layer-1, DeFi, Meme).
     - Numeric input for `Volume Rank` (Top N).
     - Toggle for `Bottom Volume`.
     - Data table displaying the dynamically filtered symbols and their metadata.
   - **Server Logic:**
     - On load, fetch available types/subtypes to populate the dropdowns.
     - Reactively fetch the filtered symbol list from the updated `/universe` endpoint whenever a filter changes.
     - Automatically update the `global_universe` reactive value with the filtered results.

## Phase 4: Module Refactoring (Market Radar, Diagnostics, etc.)
1. **Update `modules/market_radar.py`**
   - Remove the local "Top Volume" input (`n_assets_radar`).
   - Remove the local "Select Symbols" manual selectize dropdown (`radar_symbols`).
   - Update `_initialize_symbols` and the reactive calculations to strictly read from `global_universe.get()`.
2. **Update `modules/symbol_diagnostics.py` & `modules/predictive.py`**
   - Sync their asset dropdowns to use `global_universe.get()` as the source of truth for their choices, ensuring the user only selects from the pre-filtered universe.

## Phase 5: Final UI Integration
1. **`app.py` (Navigation)**
   - Add the `asset_filter_ui` to a new `ui.nav_panel("ASSET FILTER")` in the main navbar.
   - Update the "HOME" tab grid to include a card for the Asset Filter.
