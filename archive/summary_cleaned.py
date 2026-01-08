"""Summarise coverage in cleaned station files."""

import pandas as pd
from pathlib import Path


def main() -> None:
    clean_dir = Path("data/clean")
    files = sorted(clean_dir.glob("station_*.csv"))
    print("n_station_files", len(files))

    rows = []
    for f in files:
        station = f.stem.replace("station_", "")
        df = pd.read_csv(f, parse_dates=["datetime"])
        n = len(df)
        start = df["datetime"].min()
        end = df["datetime"].max()
        cols = list(df.columns)
        n_inputs = len(cols) - 2
        input_cols = [c for c in cols if c not in ("datetime", "salinity")]
        pct_missing_inputs = float(df[input_cols].isna().mean().mean() * 100) if input_cols else 0.0
        rows.append(
            {
                "station": station,
                "n_rows": n,
                "start": start,
                "end": end,
                "n_inputs": n_inputs,
                "pct_missing_inputs": pct_missing_inputs,
            }
        )

    summary = pd.DataFrame(rows).set_index("station").sort_index()
    print(summary.round(2))

    if files:
        ex = files[0]
        df_ex = pd.read_csv(ex, nrows=1)
        print("\nexample_station", ex.stem.replace("station_", ""))
        print("columns:", list(df_ex.columns))


if __name__ == "__main__":
    main()

