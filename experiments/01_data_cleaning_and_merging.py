"""Prepare cleaned per-station datasets from raw CSVs."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"

alt_raw_dir = BASE_DIR / "DATA"
if not RAW_DIR.exists() and alt_raw_dir.exists():
    RAW_DIR = alt_raw_dir

CLEAN_DIR = BASE_DIR / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

TIME_STEP = "h"

SALINITY_FILE = RAW_DIR / "salinity_all_stations.csv"
SALINITY_TIME_COL = "datetime"


def infer_station_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem

    m_bracket = re.search(r"\[([^\]]+)\]", stem)
    if m_bracket:
        return m_bracket.group(1).replace(" ", "")

    m_range = re.match(r"^(.*)_\d{4}_\d{4}$", stem)
    if m_range:
        return m_range.group(1)

    return stem


SALINITY_DIR = RAW_DIR / "Hourly_Salinity_Time_Series_44stations"
Q_DIR = RAW_DIR / "Q"
WL_DIR = RAW_DIR / "WL"
PRE_DIR = RAW_DIR / "Pre"

DISCHARGE_FILES = {}
if Q_DIR.exists():
    for path in sorted(Q_DIR.glob("*.csv")):
        sid = infer_station_id_from_filename(path.name)
        key = f"Q_{sid}"
        DISCHARGE_FILES[key] = path

TIDE_FILES = {}
if WL_DIR.exists():
    for path in sorted(WL_DIR.glob("*.csv")):
        sid = infer_station_id_from_filename(path.name)
        key = f"H_{sid}"
        TIDE_FILES[key] = path

RAIN_FILES = {}
if PRE_DIR.exists():
    for path in sorted(PRE_DIR.glob("*.csv")):
        sid = infer_station_id_from_filename(path.name)
        key = f"rain_{sid}"
        RAIN_FILES[key] = path

AGG_MAP = {
    "salinity": "mean",
    "discharge": "mean",
    "tide": "mean",
    "rain": "sum",
}

PHYSICAL_RANGES = {
    "salinity": (0.0, 60.0),
    "discharge": (0.0, 50000.0),
    "tide": (-5.0, 5.0),
    "rain": (0.0, 500.0),
}

SALINITY_STATION_COLS = None


def prepare_timeseries(df: pd.DataFrame, time_col: str | None, time_step: str, agg: str = "mean") -> pd.DataFrame:
    df = df.copy()

    col = time_col

    if col is None or str(col).lower() == "auto":
        if SALINITY_TIME_COL in df.columns:
            col = SALINITY_TIME_COL
        else:
            candidates = []
            for c in df.columns:
                cl = str(c).lower()
                if "timestamp" in cl or "date" in cl or cl.startswith("time"):
                    candidates.append(c)
            if candidates:
                col = candidates[0]

    if col is not None and col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df = df.dropna(subset=[col])
        df = df.set_index(col)
    elif isinstance(df.index, pd.DatetimeIndex):
        pass
    else:
        raise KeyError(
            f"Could not find a suitable time column in {list(df.columns)} "
            "and index is not a DatetimeIndex."
        )

    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    df = df.sort_index()

    if time_step is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            empty = df.iloc[0:0].copy()
            empty.index = pd.DatetimeIndex(empty.index)
            return empty
        df_numeric = df[numeric_cols]
        df = df_numeric.resample(time_step).agg(agg)

    return df


def clip_and_nan(df: pd.DataFrame, col_type: str) -> pd.DataFrame:
    if col_type not in PHYSICAL_RANGES:
        raise ValueError(f"Unknown col_type '{col_type}'. Expected one of {list(PHYSICAL_RANGES.keys())}.")

    low, high = PHYSICAL_RANGES[col_type]
    df_clipped = df.copy()

    numeric_cols = df_clipped.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df_clipped[col]
        mask = (series < low) | (series > high)
        if mask.any():
            df_clipped.loc[mask, col] = np.nan

    return df_clipped


def missing_summary(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    n = len(df)
    n_missing = df.isna().sum()
    pct_missing = (n_missing / n * 100) if n > 0 else np.nan

    summary = pd.DataFrame(
        {
            "n": n,
            "n_missing": n_missing,
            "pct_missing": pct_missing,
        }
    )

    print("-" * 60)
    if label:
        print(f"Missingness summary for {label}")
    else:
        print("Missingness summary")
    print(summary.round(2))

    return summary


def interpolate_short_gaps(df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("interpolate_short_gaps expects a DataFrame with a DatetimeIndex.")

    df_interp = df.copy()

    for col in df_interp.columns:
        series = df_interp[col]
        if not np.issubdtype(series.dtype, np.number):
            continue
        df_interp[col] = series.interpolate(
            method="time",
            limit=limit,
            limit_direction="both",
        )

    return df_interp



if SALINITY_FILE.exists():
    print(f"Loading wide salinity file from {SALINITY_FILE}")
    salinity_raw = pd.read_csv(SALINITY_FILE)
else:
    salinity_dir = SALINITY_DIR
    if not salinity_dir.exists():
        raise FileNotFoundError(
            f"Could not find salinity file {SALINITY_FILE} or directory {salinity_dir}. "
            "Please update SALINITY_FILE or RAW_DIR."
        )

    print(f"Building wide salinity table from per-station files in {salinity_dir}")
    sal_frames = []
    for path in sorted(salinity_dir.glob("*.csv")):
        station_id = infer_station_id_from_filename(path.name)
        df_station = pd.read_csv(path)
        if SALINITY_TIME_COL not in df_station.columns:
            raise KeyError(f"Expected time column '{SALINITY_TIME_COL}' in {path.name}")

        value_cols = [c for c in df_station.columns if c != SALINITY_TIME_COL]
        if not value_cols:
            raise ValueError(f"No value columns found in {path.name}")

        numeric_candidates = [
            c for c in value_cols if np.issubdtype(df_station[c].dtype, np.number)
        ]
        val_col = numeric_candidates[0] if numeric_candidates else value_cols[0]

        df_station = df_station[[SALINITY_TIME_COL, val_col]].rename(
            columns={val_col: station_id}
        )
        sal_frames.append(df_station)

    if not sal_frames:
        raise RuntimeError(f"No CSV files found in {salinity_dir}")

    salinity_raw = sal_frames[0]
    for df_station in sal_frames[1:]:
        salinity_raw = salinity_raw.merge(df_station, on=SALINITY_TIME_COL, how="outer")

if SALINITY_TIME_COL not in salinity_raw.columns:
    raise KeyError(f"Expected time column '{SALINITY_TIME_COL}' in salinity data.")

metadata_candidates = {
    SALINITY_TIME_COL.lower(),
    "station",
    "station_id",
    "station_name",
}

SALINITY_STATION_COLS = [
    col for col in salinity_raw.columns
    if col.lower() not in metadata_candidates
]

print("Detected salinity station columns:")
print(SALINITY_STATION_COLS)
print(f"\nRaw salinity shape: {salinity_raw.shape}")
print(salinity_raw.head())

discharge_raw = {}
if not DISCHARGE_FILES:
    print("\nNo discharge files detected. DISCHARGE_FILES is empty.")
else:
    for name, path in DISCHARGE_FILES.items():
        if not Path(path).exists():
            print(f"\n[warning] Discharge file not found for '{name}': {path}")
            continue
        df = pd.read_csv(path)
        discharge_raw[name] = df
        print(f"\nLoaded discharge series '{name}' from {path} with shape {df.shape}")
        print(df.head())

tide_raw = {}
if not TIDE_FILES:
    print("\nNo tide/water-level files detected. TIDE_FILES is empty.")
else:
    for name, path in TIDE_FILES.items():
        if not Path(path).exists():
            print(f"\n[warning] Tide file not found for '{name}': {path}")
            continue
        df = pd.read_csv(path)
        tide_raw[name] = df
        print(f"\nLoaded tide series '{name}' from {path} with shape {df.shape}")
        print(df.head())

rain_raw = {}
if not RAIN_FILES:
    print("\nNo rainfall files detected. RAIN_FILES is empty.")
else:
    for name, path in RAIN_FILES.items():
        if not Path(path).exists():
            print(f"\n[warning] Rain file not found for '{name}': {path}")
            continue
        df = pd.read_csv(path)
        rain_raw[name] = df
        print(f"\nLoaded rain series '{name}' from {path} with shape {df.shape}")
        print(df.head())



salinity_cols = [SALINITY_TIME_COL] + SALINITY_STATION_COLS
salinity_ts = prepare_timeseries(
    salinity_raw[salinity_cols],
    time_col=SALINITY_TIME_COL,
    time_step=TIME_STEP,
    agg=AGG_MAP["salinity"],
)
print(f"Salinity resampled to {TIME_STEP}: shape = {salinity_ts.shape}")

discharge_ts = {}
for name, df_raw in discharge_raw.items():
    df_ts = prepare_timeseries(
        df_raw,
        time_col="auto",
        time_step=TIME_STEP,
        agg=AGG_MAP["discharge"],
    )
    discharge_ts[name] = df_ts
    print(f"Discharge '{name}' resampled to {TIME_STEP}: shape = {df_ts.shape}")

tide_ts = {}
for name, df_raw in tide_raw.items():
    df_ts = prepare_timeseries(
        df_raw,
        time_col="auto",
        time_step=TIME_STEP,
        agg=AGG_MAP["tide"],
    )
    tide_ts[name] = df_ts
    print(f"Tide '{name}' resampled to {TIME_STEP}: shape = {df_ts.shape}")

rain_ts = {}
for name, df_raw in rain_raw.items():
    df_ts = prepare_timeseries(
        df_raw,
        time_col="auto",
        time_step=TIME_STEP,
        agg=AGG_MAP["rain"],
    )
    rain_ts[name] = df_ts
    print(f"Rain '{name}' resampled to {TIME_STEP}: shape = {df_ts.shape}")



salinity_clean = clip_and_nan(salinity_ts, col_type="salinity")
_ = missing_summary(salinity_clean, label="salinity (after range checks, no interpolation)")

discharge_qc = {}
for name, df_ts in discharge_ts.items():
    df_qc = clip_and_nan(df_ts, col_type="discharge")
    discharge_qc[name] = df_qc

tide_qc = {}
for name, df_ts in tide_ts.items():
    df_qc = clip_and_nan(df_ts, col_type="tide")
    tide_qc[name] = df_qc

rain_qc = {}
for name, df_ts in rain_ts.items():
    df_qc = clip_and_nan(df_ts, col_type="rain")
    rain_qc[name] = df_qc




discharge_clean = {}
for name, df_qc in discharge_qc.items():
    df_clean = interpolate_short_gaps(df_qc, limit=3)
    if df_clean.shape[1] == 1:
        df_clean.columns = [name]
    else:
        df_clean.columns = [f"{name}_{col}" for col in df_clean.columns]
    discharge_clean[name] = df_clean
    _ = missing_summary(df_clean, label=f"discharge '{name}' (clean, interpolated)")

tide_clean = {}
for name, df_qc in tide_qc.items():
    df_clean = interpolate_short_gaps(df_qc, limit=3)
    if df_clean.shape[1] == 1:
        df_clean.columns = [name]
    else:
        df_clean.columns = [f"{name}_{col}" for col in df_clean.columns]
    tide_clean[name] = df_clean
    _ = missing_summary(df_clean, label=f"tide '{name}' (clean, interpolated)")

rain_clean = {}
for name, df_qc in rain_qc.items():
    df_clean = interpolate_short_gaps(df_qc, limit=3)
    if df_clean.shape[1] == 1:
        df_clean.columns = [name]
    else:
        df_clean.columns = [f"{name}_{col}" for col in df_clean.columns]
    rain_clean[name] = df_clean
    _ = missing_summary(df_clean, label=f"rain '{name}' (clean, interpolated)")



common_index = salinity_clean.index
X_inputs = pd.DataFrame(index=common_index)

for name, df_clean in discharge_clean.items():
    X_inputs = X_inputs.join(df_clean, how="outer")

for name, df_clean in tide_clean.items():
    X_inputs = X_inputs.join(df_clean, how="outer")

for name, df_clean in rain_clean.items():
    X_inputs = X_inputs.join(df_clean, how="outer")

X_inputs = X_inputs.dropna(how="all")

print(f"Input feature matrix shape after merging: {X_inputs.shape}")
print(X_inputs.head())

station_data = {}
station_summaries = []

for station in SALINITY_STATION_COLS:
    sal_series = salinity_clean[station]

    n_before = len(sal_series)
    pct_missing_before = sal_series.isna().mean() * 100 if n_before > 0 else np.nan

    df_station = pd.DataFrame(index=salinity_clean.index)
    df_station["salinity"] = sal_series

    df_station = df_station.join(X_inputs, how="inner")

    df_station = df_station.dropna(subset=["salinity"])

    station_data[station] = df_station

    n_after = len(df_station)
    input_cols = [c for c in df_station.columns if c != "salinity"]
    if n_after > 0 and input_cols:
        pct_missing_inputs_after = df_station[input_cols].isna().mean().mean() * 100
    else:
        pct_missing_inputs_after = np.nan

    station_summaries.append(
        {
            "station": station,
            "n_before": n_before,
            "pct_missing_salinity_before": pct_missing_before,
            "n_after": n_after,
            "pct_missing_inputs_after": pct_missing_inputs_after,
            "start_after": df_station.index.min(),
            "end_after": df_station.index.max(),
        }
    )

if station_data:
    first_station = next(iter(station_data))
    print(f"\nExample merged dataset for station '{first_station}':")
    print(station_data[first_station].head())



for station, df_station in station_data.items():
    safe_station = "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in str(station)
    )
    out_path = CLEAN_DIR / f"station_{safe_station}.csv"
    df_station.to_csv(out_path, index_label="datetime")
    print(f"Saved cleaned dataset for station {station!r} to {out_path}")

summary_df = pd.DataFrame(station_summaries)

if not summary_df.empty:
    summary_df["start_after"] = summary_df["start_after"].dt.strftime("%Y-%m-%d %H:%M:%S")
    summary_df["end_after"] = summary_df["end_after"].dt.strftime("%Y-%m-%d %H:%M:%S")
    summary_df = summary_df.set_index("station")

    print("\nOverview of per-station datasets (before/after cleaning):")
    print(summary_df[[
        "n_before",
        "pct_missing_salinity_before",
        "n_after",
        "pct_missing_inputs_after",
        "start_after",
        "end_after",
    ]].round(2))
else:
    print("No station datasets were created; please check the salinity inputs and configuration.")
