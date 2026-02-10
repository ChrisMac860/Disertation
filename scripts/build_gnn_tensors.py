"""Build tensor_values.csv and tensor_mask.csv from cleaned station data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def find_clean_dir(root: Path) -> Path:
    candidates = [root / "DATA" / "clean", root / "data" / "clean"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find cleaned data in DATA/clean or data/clean.")


def station_id_from_path(path: Path) -> str:
    name = path.stem
    if name.startswith("station_"):
        return name[len("station_") :]
    return name


def load_station_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, low_memory=False)
    if "datetime" not in df.columns:
        raise KeyError(f"Missing datetime column in {path.name}")
    if "salinity" not in df.columns:
        raise KeyError(f"Missing salinity column in {path.name}")

    dt = pd.to_datetime(df["datetime"], errors="coerce", format="mixed")
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert(None)
    values = pd.to_numeric(df["salinity"], errors="coerce")
    series = pd.Series(values.values, index=dt).dropna()
    series = series[series.index.notna()]
    if series.empty:
        return series
    series = series.groupby(level=0).mean().sort_index()
    return series


def build_tensor_frames(series_map: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_ts = min(s.index.min() for s in series_map.values())
    max_ts = max(s.index.max() for s in series_map.values())
    full_index = pd.date_range(min_ts, max_ts, freq="h")

    stations = sorted(series_map.keys())
    values_df = pd.DataFrame(index=full_index, columns=stations, dtype=float)
    for station, series in series_map.items():
        values_df[station] = series.reindex(full_index)

    mask_df = values_df.notna().astype(int)
    return values_df, mask_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GNN tensor values/mask from cleaned data.")
    ap.add_argument("--root", default=".", help="Repo root (default: current directory)")
    ap.add_argument("--clean-dir", default=None, help="Override cleaned data directory")
    ap.add_argument("--out-dir", default=None, help="Output directory for tensor CSVs")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    clean_dir = Path(args.clean_dir).resolve() if args.clean_dir else find_clean_dir(root)

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = root / "data" if (root / "data").exists() else root / "DATA"

    out_dir.mkdir(parents=True, exist_ok=True)

    series_map: dict[str, pd.Series] = {}
    for path in sorted(clean_dir.glob("station_*.csv")):
        station_id = station_id_from_path(path)
        series = load_station_series(path)
        if series.empty:
            continue
        series_map[station_id] = series

    if not series_map:
        raise RuntimeError(f"No usable station files found in {clean_dir}")

    values_df, mask_df = build_tensor_frames(series_map)

    values_path = out_dir / "tensor_values.csv"
    mask_path = out_dir / "tensor_mask.csv"
    values_df.to_csv(values_path, index_label="datetime")
    mask_df.to_csv(mask_path, index_label="datetime")

    print(f"Wrote {values_path} and {mask_path}")
    print(f"Stations: {len(values_df.columns)}")
    print(f"Time span: {values_df.index.min()} -> {values_df.index.max()}")
    print(f"Total rows: {len(values_df)}")


if __name__ == "__main__":
    main()
