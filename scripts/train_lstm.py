import argparse, math, random
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def read_csv(path):
    try: return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError: return pd.read_csv(path, encoding="latin-1", low_memory=False)

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
        if any(p in lc for p in prefs) and pd.api.types.is_numeric_dtype(df[c]): return c
    for c in df.columns:
        if c != tcol and pd.api.types.is_numeric_dtype(df[c]): return c
    return None

def make_supervised(ts: pd.DataFrame, window=24, horizon=24):
    x, y, t = [], [], []
    v = ts["salinity"].to_numpy(dtype=np.float32)
    for i in range(window, len(v)-horizon):
        x.append(v[i-window:i])
        y.append(v[i+horizon])
        t.append(ts["datetime"].iloc[i+horizon])
    # Return timestamps as a Series so .iloc slicing is available downstream
    t_series = pd.Series(pd.to_datetime(t), name="datetime")
    return np.stack(x), np.array(y, dtype=np.float32), t_series

class SeqDS(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): 
        return torch.from_numpy(self.X[i]).unsqueeze(-1), torch.tensor(self.y[i])

class LSTMReg(nn.Module):
    def __init__(self, input_size=1, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out,_ = self.lstm(x)
        return self.fc(out[:,-1,:]).squeeze(-1)

def mae(a,b): return float(np.mean(np.abs(a-b)))
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

def train_one(Xtr,ytr,Xva,yva, hidden=64, layers=2, dropout=0.1, epochs=10, lr=1e-3, bs=128, seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = LSTMReg(hidden=hidden, layers=layers, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dl_tr = DataLoader(SeqDS(Xtr,ytr), batch_size=bs, shuffle=True)
    dl_va = DataLoader(SeqDS(Xva,yva), batch_size=bs, shuffle=False)
    best, patience, no_improve = (1e9, None), 5, 0
    for ep in range(1, epochs+1):
        model.train()
        for xb,yb in dl_tr:
            opt.zero_grad(); pred = model(xb); loss = loss_fn(pred, yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vpred = torch.cat([model(xb) for xb,_ in dl_va])
            vloss = loss_fn(vpred, torch.tensor(yva)).item()
        if vloss < best[0]: best, no_improve = (vloss, model.state_dict()), 0
        else: no_improve += 1
        if no_improve >= patience: break
    model.load_state_dict(best[1]); model.eval()
    return model

def main(root, stations, horizons, window, epochs, seed):
    root = Path(root)
    sal_dir = root / "DATA" / "Hourly_Salinity_Time_Series_44stations"
    reports = root / "reports"; figures = root / "figures"
    reports.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    metrics_rows = []

    files = sorted(sal_dir.glob("*.csv"))
    if stations:
        files = [fp for fp in files if any(fp.stem.lower().startswith(s.lower()) for s in stations)]

    for fp in files:
        df = read_csv(fp); df.columns = [c.strip() for c in df.columns]
        tcol, dt = detect_time_col(df); 
        if tcol is None: continue
        scol = pick_sal_col(df, tcol); 
        if scol is None: continue
        df[tcol] = dt; df = df.dropna(subset=[tcol]).sort_values(tcol)
        ts = df[[tcol, scol]].rename(columns={tcol:"datetime", scol:"salinity"})
        ts["salinity"] = pd.to_numeric(ts["salinity"], errors="coerce"); ts = ts.dropna()
        if len(ts) < window + max(horizons) + 50: continue

        # chronological split
        n = len(ts); i_test = int(n*0.85); i_val = int(n*0.70)
        for h in horizons:
            X, y, t = make_supervised(ts, window=window, horizon=h)
            # align splits after making sequences
            n2 = len(X); i_test2 = max(1, int(n2*0.85)); i_val2 = max(1, int(n2*0.70))
            Xtr, ytr = X[:i_val2], y[:i_val2]
            Xva, yva = X[i_val2:i_test2], y[i_val2:i_test2]
            Xte, yte = X[i_test2:], y[i_test2:]
            # standardise by train and keep float32 for Torch
            mu, sd = Xtr.mean(), Xtr.std() if Xtr.std()>0 else 1.0
            Xtr = ((Xtr - mu)/sd).astype(np.float32)
            Xva = ((Xva - mu)/sd).astype(np.float32)
            Xte = ((Xte - mu)/sd).astype(np.float32)

            model = train_one(Xtr,ytr,Xva,yva, epochs=epochs, seed=seed)
            with torch.no_grad():
                yhat = model(torch.from_numpy(Xte).unsqueeze(-1)).numpy()

            # save preds
            out_pred = pd.DataFrame({"datetime": t.iloc[-len(yhat):].astype(str),
                                     "y_true": yte, "y_pred": yhat})
            out_pred.to_csv(reports / f"pred_{fp.stem}_h{h}.csv", index=False)

            m_mae, m_rmse = mae(yte,yhat), rmse(yte,yhat)
            metrics_rows.append({"station": fp.stem, "h": h, "MAE": m_mae, "RMSE": m_rmse,
                                 "window": window, "epochs": epochs, "seed": seed, "model": "LSTM"})

    pd.DataFrame(metrics_rows).to_csv(reports / "models_lstm_metrics.csv", index=False)
    print("Wrote", reports / "models_lstm_metrics.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stations", nargs="*", default=None, help="Optional list of station name prefixes")
    ap.add_argument("--horizons", nargs="*", type=int, default=[1,3,6,24])
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.root, args.stations, args.horizons, args.window, args.epochs, args.seed)

