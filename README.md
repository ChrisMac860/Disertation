# Mekong Salinity â€” Repo Quick Start

## Environment (Windows)
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run baselines (outputs to `reports\`)

```bat
python scripts\persistence_baseline.py --root "C:\Users\User\Desktop\Disertation" --horizons 1,3,6,24
python scripts\count_threshold_events.py --root "C:\Users\User\Desktop\Disertation"
```

## Train LSTM models from raw station files

To train a simple univariate LSTM directly on each raw hourly salinity time series under `DATA\Hourly_Salinity_Time_Series_44stations` and write predictions/metrics to `reports\`:

```bat
python scripts\train_lstm.py --root "C:\Users\User\Desktop\Disertation" --horizons 1 3 6 24 --window 24 --epochs 10
```

This produces per-station prediction files `reports\pred_<Station>_h{h}.csv` and an aggregate metrics table `reports\models_lstm_metrics.csv`.

## Weekly docs to keep updated

* `docs\STATUS.md` â€” rolling log (queries, decisions, blockers, next actions)
* `docs\pilot_results.md` â€” 1â€“2 page update for supervisor (export to PDF for email)

## Folders

* `DATA\`           private dataset (do not share)
* `scripts\`        analysis scripts (no notebooks required)
* `reports\`        CSV outputs and short markdown summaries
* `figures\`        PNG charts from pilot models / diagnostics
* `viewer\`         the HTML viewer app (local only)
* `data\clean\`     model-ready, per-station time series for LSTM models (built by `experiments\01_data_cleaning_and_merging.py`)
* `experiments\`    Python-only versions of the original exploratory notebooks
* `src\gnn\`        GNN model, trainer, and graph builder utilities
* `src\lstm_gate\`  gate-index LSTM experiment (Python script)
* `archive\notebooks\`  archived `.ipynb` files (kept for reference; not needed to run)

**Privacy:** Data is private use only. Do not publish raw data or repo.

## Cleaned modelling datasets

The script `experiments\01_data_cleaning_and_merging.py` (Python-only) loads the raw salinity, discharge, water-level and rainfall CSVs under `DATA\`, standardises them to a common hourly time step, applies simple range-based QC and short-gap interpolation for inputs, and then writes one cleaned CSV per salinity station into `data\clean\`. Each file is named `station_<StationName>.csv` and contains:

* `datetime` (timezone-naive, hourly index).
* `salinity` for that station (target, not interpolated).
* Discharge inputs `Q_*_Value` plus station code columns.
* Water-level inputs `H_*_Value` plus station code columns.
* Rainfall inputs `rain_*_Value` plus station code columns.

Rows with missing salinity are dropped so the per-station files can be used directly for supervised learning (e.g. LSTM forecasting) with `salinity` as the target and all other columns as predictors.

Additional scripts build on these cleaned datasets:

* `summary_cleaned.py` summarises coverage, number of inputs and average missingness across all `data\clean\station_*.csv` files.
* `summary_rain_AnDinh.py` / `summary_rain_all_AnDinh.py` explore which rainfall series at AnDinh have enough variability and non-zero values to be useful in downstream models.
* `experiments\02_baseline_lstm_single_station.py` trains a first 1-step-ahead LSTM at a single station and compares it against a persistence baseline.
* `experiments\02_linear_baseline_single_station.py` fits a Ridge regression baseline on the same inputs for a like-for-like comparison with the LSTM.
* `src\lstm_gate\gate_index_lstm.py` defines a physics-motivated "gate index" that combines tidal coupling and recent rainfall, then compares LSTM models with and without this feature on the cleaned data (the original `.ipynb` lives under `archive\notebooks\` if needed for reference).

The single-station scripts (see `experiments\` and `src\lstm_gate\`) expect additional Python packages (e.g. scikit-learn, matplotlib, TensorFlow) beyond the minimal `requirements.txt`.

## Viewer app

The `viewer\` folder contains a local-only web viewer for hydromet CSVs and model outputs:

* `index.html` + `app.js` load CSVs/ZIPs or a folder of files, auto-detect variable type (salinity, water level, discharge, rain), and render time-series charts, overview panels and diagnostics, with CSV export.
* `compare.html` + `compare.js` overlay variables from two stations (A/B) on shared plots, using the same auto-load and filtering mechanisms.
* `tests\compare.spec.js` is a Playwright UI test that starts a local `python -m http.server`, navigates from `index.html` to `compare.html`, auto-loads `reports\data_inventory.csv`, selects two stations and checks that salinity traces are rendered.

See `CONTEXT.md` for a more detailed description of the viewer behaviour, auto-detection rules and data expectations.

