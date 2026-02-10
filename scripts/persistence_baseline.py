"""Compute persistence baseline metrics."""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


## Helper Functions
## Read in the CSV
def read_csv(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)

##Find the time column
def detect_time_col(df):
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["datetime","timestamp","date_time","time","date"]):
            dt = pd.to_datetime(df[c], errors="coerce", utc=True)## Try to parse by datetime :)
            if dt.notna().any():## We found it!!
                return c, dt
    c = df.columns[0] ## If it doesnt work just pick the first
    dt = pd.to_datetime(df[c], errors="coerce", utc=True)
    return (c, dt) if dt.notna().any() else (None, None) ## If the first column doesnt work just return None none

def pick_sal_col(df, tcol): ## Find the salinity column, same thing to be robust:)
    prefs = ["sal", "salinity", "ppt", "psu", "g/l", "g\\l", "ec", "ms/cm"]
    for c in df.columns:
        if c == tcol: continue
        lc = c.lower()
        if any(p in lc for p in prefs) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    for c in df.columns:
        if c != tcol and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def mae(a,b): 
    v=(a-b).abs(); return float(v.mean()) if len(v) else float("nan") ## Mean absolute Error function

def rmse(a,b):
    v=(a-b)**2; return float(np.sqrt(v.mean())) if len(v) else float("nan") ## Root mean square error function (Big fuck ups)

def main(root, horizons):
    sal_dir = Path(root) / "DATA" / "Hourly_Salinity_Time_Series_44stations" ## Get the salanity values
    out = Path(root) / "reports"; out.mkdir(parents=True, exist_ok=True) ## Find / Create reports dircectory 

    rows = []
    for fp in sorted(sal_dir.glob("*.csv")):## For every station in the csv
        df = read_csv(fp)
        df.columns = [c.strip() for c in df.columns]
        tcol, dt = detect_time_col(df) ## Find datetime
        if tcol is None: 
            continue
        df[tcol] = dt
        df = df.dropna(subset=[tcol]).sort_values(tcol)
        scol = pick_sal_col(df, tcol)
        if scol is None: 
            continue

##Only Keep datetime and salanity.
        ts = df[[tcol, scol]].rename(columns={tcol:"datetime", scol:"salinity"})
        ts["salinity"] = pd.to_numeric(ts["salinity"], errors="coerce")
        ts = ts.dropna(subset=["salinity"])
        if len(ts) < 200: 
            continue

        n = len(ts); split = int(n*0.80)
        test = ts.iloc[split:].copy() ## split into test and train

        for h in horizons: ## Predict for every horizon 
            pred = ts["salinity"].shift(h).iloc[split:]
            y = test["salinity"]
            yhat = pred.reindex_like(y)
            rows.append({
                "station": fp.stem, "file": fp.name, "h": h,
                "n_test": int(y.dropna().shape[0]),
                "MAE": mae(y,yhat), "RMSE": rmse(y,yhat)
            })

    res = pd.DataFrame(rows)
    res.to_csv(out / "persistence_baseline.csv", index=False)
    if len(res):
        res.groupby("h")[ ["MAE","RMSE"] ].mean().reset_index().to_csv(out / "persistence_macro.csv", index=False)
    print (res)

if __name__ == "__main__":
    """ Shit wont work
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--horizons", default="1,3,6,24")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    main(args.root, horizons)"""

    root = Path(__file__).resolve().parents[1]
    horizons = [1,3,6,24]
    main(root, horizons)

