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

## Cleaned, model-ready datasets (`data/clean/`)
The notebook `01_data_cleaning_and_merging.ipynb` assembles and cleans hourly time series for salinity and drivers, then writes one CSV per station to `data/clean/`:
- One file per salinity station, named `station_<StationName>.csv` (42 stations in the current run).
- Each file has a `datetime` column (timezone-naive, hourly) and a `salinity` column for that station.
- Input features are shared across stations and include discharge (`Q_*`), tidal water level (`H_*`), and rainfall (`rain_*`) series derived from the DENR/telemetry feeds.
- For each Q/WL/Pre series the notebook keeps the numeric `Value` column and its corresponding station code column (e.g., `Q_MyThuan_Value`, `Q_MyThuan_Station Code`), clipped to simple physical ranges and with short gaps (<=3 steps) interpolated for inputs only.
- Rows with missing salinity are dropped so that each per-station CSV is directly usable as supervised learning data for LSTM experiments (target = `salinity`, predictors = all other columns).

## Baseline scripts and reports (`scripts/`, `reports/`)
Beyond the viewer, the repo includes lightweight, reproducible scripts for baselines and event statistics:

- `scripts/persistence_baseline.py` computes multi-horizon persistence baselines from the raw hourly salinity files under `DATA/Hourly_Salinity_Time_Series_44stations`, writing station-level metrics to `reports/persistence_baseline.csv` and horizon-averaged scores to `reports/persistence_macro.csv`.
- `scripts/count_threshold_events.py` counts how often each station exceeds the 3 g/L and 4 g/L salinity thresholds and writes `reports/threshold_event_counts.csv`, used to decide which stations are viable for exceedance classification versus pure regression.
- `scripts/train_lstm.py` trains a simple univariate PyTorch LSTM directly on each raw station salinity series, producing per-station prediction CSVs `reports/pred_<Station>_h{h}.csv` and an aggregate metrics file `reports/models_lstm_metrics.csv`.

These scripts share the same CSV-reading and auto-detection utilities, so they can be re-run as the raw data updates.

## Single-station modelling on cleaned data (`data/clean/`)
The cleaned, per-station datasets are used in a small set of exploratory single-station notebooks (paired `.ipynb` notebooks and `.py` “Jupytext” scripts):

- `01_data_cleaning_and_merging` builds the per-station files described above.
- `02_baseline_lstm_single_station` trains a first 1-step-ahead LSTM on one station, comparing it against a persistence baseline and logging diagnostics and plots.
- `02_linear_baseline_single_station` fits a Ridge regression baseline on the same cleaned data, using recent history of salinity and hydrological drivers; its performance is directly compared to the LSTM and persistence.
- `04_gate_index_lstm_single_station.py` constructs a physics-motivated “gate index” feature that combines tidal coupling and recent rainfall, then trains paired LSTM models with and without this feature to test whether it improves short-horizon forecasts.

Helper scripts:

- `summary_cleaned.py` summarises coverage, feature counts and average missingness across all `data/clean/station_*.csv` files.
- `summary_rain_AnDinh.py` and `summary_rain_all_AnDinh.py` investigate which rainfall series at AnDinh have enough variability and non-zero observations to be useful in the gate-index experiments.

## Viewer compare tool and tests
The pair `viewer/compare.html` + `viewer/compare.js` reuses the same parsing and auto-detection logic as the main viewer to let you overlay variables from two stations (A/B) with separate station pickers but shared variable and date filters. It supports the same data-loading modes as `index.html` (auto-load via `reports/data_inventory.csv`, folder picker, ZIP, and CSV uploads) and can export the combined comparison view as CSV.

End-to-end browser behaviour for the compare view is covered by a small Playwright UI test in `tests/compare.spec.js`, which starts a local `python -m http.server` instance, navigates from `viewer/index.html` to `viewer/compare.html`, auto-loads the data inventory, selects two stations, and verifies that salinity traces are rendered.
