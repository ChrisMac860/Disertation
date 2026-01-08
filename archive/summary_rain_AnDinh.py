"""Summarise rainfall series for AnDinh."""

import pandas as pd
from pathlib import Path


def main() -> None:
    path = Path("data/clean/station_AnDinh.csv")
    df = pd.read_csv(path, index_col="datetime", parse_dates=True)

    rain_col = "rain_MyTho_Value"
    if rain_col not in df.columns:
        print("Column", rain_col, "not found in", path)
        print("Available columns:", df.columns.tolist())
        return

    rain = df[rain_col]
    print("Rain column:", rain_col)
    print("\nOverall describe:")
    print(rain.describe())
    print("\nNaN fraction:", rain.isna().mean())

    nonnull = rain.dropna()
    if nonnull.empty:
        print("\nNo non-NaN rainfall values.")
        return

    print("\nFirst non-NaN:", nonnull.index.min(), "value:", nonnull.iloc[0])
    print("Last  non-NaN:", nonnull.index.max(), "value:", nonnull.iloc[-1])

    nonzero = nonnull[nonnull != 0]
    print("\nNon-zero count:", len(nonzero))
    print("Non-zero fraction (within non-NaN):", (len(nonzero) / len(nonnull)))

    annual = nonnull.resample("A").count()
    print("\nNon-NaN count per year (first 10 years):")
    print(annual.head(10))

    sal = df["salinity"]
    joined = pd.concat({"rain": rain, "salinity": sal}, axis=1).dropna()
    if not joined.empty:
        print("\nCorrelation rain vs salinity:", joined["rain"].corr(joined["salinity"]))
    else:
        print("\nNo overlapping non-NaN between rain and salinity for correlation.")


if __name__ == "__main__":
    main()

