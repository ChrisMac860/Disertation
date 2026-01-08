"""List rainfall series columns for AnDinh."""

import pandas as pd
from pathlib import Path


def main() -> None:
    path = Path("data/clean/station_AnDinh.csv")
    df = pd.read_csv(path, index_col="datetime", parse_dates=True)

    rain_cols = [c for c in df.columns if c.startswith("rain_") and c.endswith("_Value")]
    print("Rain value columns:")
    print(rain_cols)

    rows = []
    for col in rain_cols:
        s = df[col]
        nonnull = s.dropna()
        n = len(s)
        nn = len(nonnull)
        nz = int((nonnull != 0).sum())
        std = float(nonnull.std()) if nn else float("nan")
        rows.append(
            {
                "column": col,
                "n_rows": n,
                "n_nonnull": nn,
                "nan_frac": (n - nn) / n if n else float("nan"),
                "n_nonzero": nz,
                "nonzero_frac": nz / nn if nn else float("nan"),
                "std": std,
            }
        )

    summary = pd.DataFrame(rows).set_index("column")
    print("\nRain columns variability summary (sorted by std desc):")
    print(summary.sort_values("std", ascending=False))


if __name__ == "__main__":
    main()

