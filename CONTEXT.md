# Mekong Station Visualiser — Context

This project is a minimal, single‑page, client‑only viewer for hydromet CSV data across many stations. Everything runs locally in your browser; there is no backend.

## Overview
- Load data from a ZIP of the `DATA/` folder, multiple CSVs, an on‑disk folder (Chromium browsers), or auto‑load via `reports/data_inventory.csv`.
- Auto‑detects variable type (kind), datetime column, value column, and station ID.
- Builds an in‑memory long table and renders:
  - Overview by variable
  - Time‑series charts (Plotly): one per variable, multi‑station lines
  - Diagnostics: missingness (daily row counts) and salinity exceedance table
- Filter by variables, stations, date range; optionally mark salinity exceedances (>=3, >=4 g/L).
- Export the current filtered view as CSV.

## How It Works
- Pure front‑end app built from `index.html`, `style.css`, and `app.js`.
- Libraries loaded via CDN in the page: Plotly (charts), Papa Parse (CSV), JSZip (ZIP reading).
- When data is provided, `app.js` maps files into a long table with rows:
  `{ datetimeDate, datetimeISO, station_id, kind, value, display_name }`.
- UI state (selected kinds/stations, date range, thresholds, toggles) persists in `localStorage` under key `msv_state_v1`.

## Data Inputs
- ZIP: Select a ZIP containing CSV files (e.g., a zipped `DATA/` folder).
- CSVs: Select multiple `.csv` at once.
- Folder picker: Choose a local folder; app reads all `.csv` inside (Chromium‑based browsers via File System Access API).
- Auto‑load: If served via a local web server and `reports/data_inventory.csv` is present, the app fetches and parses the `RelativePath` entries that live under `data/`.

### Auto‑Detection Rules (app.js)
- Kind:
  - `salinity` if name contains “salinity”
  - `water_level` if path contains `\WL\` or name contains “Water Level”
  - `discharge` if path contains `\Q\` or name contains “Discharge”
  - `rain` if path contains `\Pre\` or name contains “Rain”/“Rainfall”
- Datetime column:
  - First header matching one of `datetime, timestamp, date_time, time, date` that parses for ~60%+ of sample rows; otherwise first column that parses sufficiently.
- Value column:
  - Prefer kind‑specific name hints (e.g., salinity: `sal, salinity, ppt, psu, g/l, g\l, ec, ms/cm, value`); fallback to first numeric column that isn’t the datetime.
- Station ID:
  - If filename contains `[Station Name]`, use the bracketed content without spaces (e.g., `[Tan Chau]` → `TanChau`).
  - Else use the stem before a trailing `_YYYY_YYYY` (e.g., `AnDinh_1996_2023` → `AnDinh`).

## UI and Features
- Filters: variable kinds (salinity, water level, discharge, rain), station multi‑select with search, date range, optional salinity thresholds (>=3 and >=4 g/L) and plot markings.
- Overview: points per kind and date coverage.
- Charts: one Plotly time‑series per kind; multi‑line by station; optional threshold lines for salinity.
- Diagnostics: daily row‑count “missingness” chart and a top‑20 salinity exceedance table (counts and fractions >=3/>=4 g/L).
- Export: CSV with columns `datetime, station_id, display_name, kind, value` based on current filters.
- Mock data: Built‑in generator to test UI without real files (3 stations × 2 kinds × 7 days).

## Running Locally
Because browsers restrict `file://` access, serving the folder is recommended:
- VS Code Live Server: “Open with Live Server” on `index.html`.
- Python: `python -m http.server` and visit `http://localhost:8000/`.
- Node: any static server (e.g., `npx serve`).

Once running:
1) Try “Auto‑load from data/” (uses `reports/data_inventory.csv`). If blocked by `file://`, use folder picker or uploads.
2) Optionally load station metadata CSV with columns: `station_id,display_name,lat,lon,river,notes`.
3) Select variables/stations/dates; toggle salinity threshold markers; export filtered CSV as needed.

## Files and Folders
- `index.html` — HTML shell, loads CSS/JS and CDNs, defines controls and chart containers.
- `style.css` — Light UI theme, responsive layout, table and chart sizing.
- `app.js` — All logic: parsing, auto‑detection, state, filtering, rendering, events, and init.
- `reports/data_inventory.csv` — Inventory of CSV files under `data/`, used by auto‑load.
- `reports/data_report.md` — Human‑readable summary (counts, sizes, largest files).
- `DATA/` or `data/` — Source CSVs (various hydromet folders), not required but expected for auto‑load examples.

## Limitations and Notes
- Everything runs in memory; very large datasets/ZIPs may be slow.
- Date parsing relies on `new Date(value)` with minimal fallbacks; unusual formats may need adjustments.
- Auto‑detection can fail with atypical headers; adjust `VALUE_COLUMN_HINTS` and `DATETIME_COLUMN_HINTS` in `app.js` if needed.
- Folder picker requires a Chromium‑based browser.
- Privacy: No data leaves your machine; the only network activity is loading CDN libraries.

## Customisation Pointers
- Add/adjust value column hints per kind in `app.js` (`VALUE_COLUMN_HINTS`).
- Extend datetime header hints in `DATETIME_COLUMN_HINTS`.
- Tweak station ID extraction in `extractStationId` if your filenames differ.
- Change salinity thresholds defaults via `#thr3` and `#thr4` inputs in `index.html` or their initial values in state.

