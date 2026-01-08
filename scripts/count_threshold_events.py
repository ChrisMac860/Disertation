"""Count threshold exceedance events."""

import argparse
from pathlib import Path
import pandas as pd

def read_csv(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)

def detect_time_col(df):
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["datetime","timestamp","date_time","time","date"]):
            dt = pd.to_datetime(df[c], errors="coerce", utc=True)
            if dt.notna().any(): return c, dt
    c = df.columns[0]; dt = pd.to_datetime(df[c], errors="coerce", utc=True)
    return (c, dt) if dt.notna().any() else (None, None)

def pick_sal_col(df, tcol):
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

def main(root):
    sal_dir = Path(root) / "DATA" / "Hourly_Salinity_Time_Series_44stations"
    out = Path(root) / "reports"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fp in sorted(sal_dir.glob("*.csv")):
        df = read_csv(fp)
        df.columns = [c.strip() for c in df.columns]
        tcol, dt = detect_time_col(df)
        if tcol is None: 
            continue
        scol = pick_sal_col(df, tcol)
        if scol is None: 
            continue
        df[tcol] = dt
        df = df.dropna(subset=[tcol]).sort_values(tcol)
        s = pd.to_numeric(df[scol], errors="coerce").dropna()
        if s.empty: 
            continue
        rows.append({
            "station": fp.stem,
            "n_obs": int(s.shape[0]),
            "ge3_count": int((s >= 3).sum()),
            "ge4_count": int((s >= 4).sum()),
            "ge3_frac": float((s >= 3).mean()),
            "ge4_frac": float((s >= 4).mean())
        })
    pd.DataFrame(rows).sort_values("ge4_count", ascending=False).to_csv(out / "threshold_event_counts.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    main(args.root)

