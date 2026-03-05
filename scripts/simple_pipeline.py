"""Run a simple, strict 4-model pipeline on cleaned station data.

Pipeline:
1) Run data cleaning/merge script.
2) For each station, build one strict dataset with required drivers.
3) Train/evaluate: Persistence, Ridge, LSTM_baseline, LSTM_with_gate.
4) Write metrics and per-model predictions.
"""

from __future__ import annotations

import argparse
import copy
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_STATIONS = ["AnDinh", "AnLacTay", "AnNinh", "AnThuan"]
TARGET_COL = "salinity"
BASE_FEATURE_COLS = ["salinity", "H_MyTho_Value", "H_VamKenh_Value", "rain_MyTho_Value"]


@dataclass
class SplitSlices:
    train: slice
    val: slice
    test: slice
    n_train: int
    n_val: int
    n_test: int


class Standardizer:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, arr: np.ndarray) -> "Standardizer":
        self.mean_ = arr.mean(axis=0)
        scale = arr.std(axis=0)
        scale[scale == 0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer must be fit before transform.")
        return (arr - self.mean_) / self.scale_

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer must be fit before inverse_transform.")
        return arr * self.scale_ + self.mean_


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom == 0.0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def run_cleaning(root: Path, python_exe: str) -> None:
    cmd = [python_exe, str(root / "experiments" / "01_data_cleaning_and_merging.py")]
    print("[clean] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(root), check=True)


def find_clean_dir(root: Path) -> Path:
    candidates = [root / "data" / "clean", root / "DATA" / "clean"]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError("Could not find cleaned data in data/clean or DATA/clean.")
    with_files = [p for p in existing if list(p.glob("station_*.csv"))]
    return with_files[0] if with_files else existing[0]


def compute_gate_index(
    df: pd.DataFrame,
    local_h_col: str = "H_MyTho_Value",
    tide_h_col: str = "H_VamKenh_Value",
    rain_col: str = "rain_MyTho_Value",
    gate_window_hours: int = 48,
    rain_window_hours: int = 24,
    k_amp: float = 1.0,
    r0: float = 10.0,
) -> pd.Series:
    h_local = df[local_h_col].astype(float)
    h_tide = df[tide_h_col].astype(float)
    rain = df[rain_col].astype(float)

    corr_ht = h_local.rolling(window=gate_window_hours, min_periods=3).corr(h_tide)
    std_local = h_local.rolling(window=gate_window_hours, min_periods=3).std()
    std_tide = h_tide.rolling(window=gate_window_hours, min_periods=3).std()
    amp_ratio = std_local / (std_tide + 1e-6)
    amp_ratio_norm = (amp_ratio / k_amp).clip(lower=0.0, upper=1.0)

    i_tide = corr_ht.clip(lower=0.0).fillna(0.0) * amp_ratio_norm.fillna(0.0)

    rain_nonnull = rain.dropna()
    if rain_nonnull.empty:
        s_rain = pd.Series(1.0, index=df.index)
    else:
        rain_std = float(rain_nonnull.std())
        nonzero_frac = float((rain_nonnull != 0).mean())
        if rain_std <= 1e-3 or nonzero_frac <= 0.01:
            s_rain = pd.Series(1.0, index=df.index)
        else:
            rain_sum = rain.fillna(0.0).rolling(window=rain_window_hours, min_periods=1).sum()
            s_rain = np.exp(-rain_sum / r0)

    return (i_tide * s_rain).rename("gate_index")


def load_station_frame(clean_dir: Path, station: str) -> pd.DataFrame:
    path = clean_dir / f"station_{station}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing station file: {path}")

    df = pd.read_csv(path, parse_dates=["datetime"])
    missing = [c for c in BASE_FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for {station}: {missing}")

    df = df[["datetime"] + BASE_FEATURE_COLS].copy()
    for c in BASE_FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=BASE_FEATURE_COLS).sort_values("datetime")
    if df.empty:
        raise ValueError(f"No complete rows after strict dropna for {station}.")

    df = df.drop_duplicates(subset=["datetime"], keep="last").set_index("datetime")
    df["gate_index"] = compute_gate_index(df).fillna(0.0)
    return df


def create_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    feats = df[feature_cols].to_numpy(dtype=np.float32)
    target = df[TARGET_COL].to_numpy(dtype=np.float32)
    times = df.index

    X, y, t = [], [], []
    upper = len(df) - horizon
    for i in range(lookback - 1, upper):
        X.append(feats[i - lookback + 1 : i + 1])
        y.append(target[i + horizon])
        t.append(times[i + horizon])

    if not X:
        raise ValueError(
            f"Not enough rows ({len(df)}) for lookback={lookback} and horizon={horizon}."
        )

    return np.stack(X), np.asarray(y, dtype=np.float32), pd.DatetimeIndex(t)


def make_splits(n_samples: int, val_frac: float, test_frac: float) -> SplitSlices:
    n_test = int(np.floor(test_frac * n_samples))
    n_val = int(np.floor(val_frac * n_samples))
    n_train = n_samples - n_val - n_test
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes for n_samples={n_samples}, val_frac={val_frac}, test_frac={test_frac}."
        )
    return SplitSlices(
        train=slice(0, n_train),
        val=slice(n_train, n_train + n_val),
        test=slice(n_train + n_val, n_samples),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )


def scale_windows(X_train: np.ndarray, X_other: np.ndarray, scaler: Standardizer) -> np.ndarray:
    n, L, F = X_other.shape
    train_flat = X_train.reshape(-1, F)
    scaler.fit(train_flat)
    return scaler.transform(X_other.reshape(-1, F)).reshape(n, L, F)


def fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    n, d = X.shape
    X1 = np.hstack([np.ones((n, 1), dtype=np.float64), X])
    reg = np.eye(d + 1, dtype=np.float64)
    reg[0, 0] = 0.0
    lhs = X1.T @ X1 + alpha * reg
    rhs = X1.T @ y
    try:
        w = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(lhs) @ rhs
    return w


def predict_ridge(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    X1 = np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), X])
    return (X1 @ w).astype(np.float32)


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    patience: int,
) -> LSTMRegressor:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMRegressor(input_size=X_train.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ds_train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    ds_val = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("LSTM training failed to produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_lstm(model: LSTMRegressor, X: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        preds = model(torch.from_numpy(X).to(device)).cpu().numpy()
    return preds.astype(np.float32)


def evaluate_and_save(
    pred_path: Path,
    station: str,
    model_name: str,
    times: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    pred_df = pd.DataFrame(
        {
            "datetime": times.astype(str),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    pred_df.to_csv(pred_path / f"{station}_{model_name}.csv", index=False)
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }


def run_station_models(
    station: str,
    clean_dir: Path,
    pred_dir: Path,
    lookback: int,
    horizon: int,
    val_frac: float,
    test_frac: float,
    ridge_alpha: float,
    lstm_epochs: int,
    lstm_batch_size: int,
    lstm_lr: float,
    lstm_patience: int,
    seed: int,
) -> list[dict[str, object]]:
    df = load_station_frame(clean_dir, station)

    X_base, y, t = create_sequences(df, BASE_FEATURE_COLS, lookback, horizon)
    gate_cols = BASE_FEATURE_COLS + ["gate_index"]
    X_gate, y_gate, t_gate = create_sequences(df, gate_cols, lookback, horizon)

    if not np.array_equal(y, y_gate):
        raise RuntimeError(f"Target mismatch between base and gate sequences for {station}.")
    if not t.equals(t_gate):
        raise RuntimeError(f"Timestamp mismatch between base and gate sequences for {station}.")

    splits = make_splits(len(y), val_frac=val_frac, test_frac=test_frac)
    y_train = y[splits.train]
    y_val = y[splits.val]
    y_test = y[splits.test]
    t_test = t[splits.test]

    x_scaler_base = Standardizer()
    Xb_train_raw = X_base[splits.train]
    Xb_val_raw = X_base[splits.val]
    Xb_test_raw = X_base[splits.test]
    Xb_train = scale_windows(Xb_train_raw, Xb_train_raw, x_scaler_base)
    Xb_val = scale_windows(Xb_train_raw, Xb_val_raw, x_scaler_base)
    Xb_test = scale_windows(Xb_train_raw, Xb_test_raw, x_scaler_base)

    x_scaler_gate = Standardizer()
    Xg_train_raw = X_gate[splits.train]
    Xg_val_raw = X_gate[splits.val]
    Xg_test_raw = X_gate[splits.test]
    Xg_train = scale_windows(Xg_train_raw, Xg_train_raw, x_scaler_gate)
    Xg_val = scale_windows(Xg_train_raw, Xg_val_raw, x_scaler_gate)
    Xg_test = scale_windows(Xg_train_raw, Xg_test_raw, x_scaler_gate)

    y_scaler = Standardizer().fit(y_train.reshape(-1, 1))
    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)

    rows: list[dict[str, object]] = []

    persist_pred = Xb_test_raw[:, -1, 0]
    m = evaluate_and_save(pred_dir, station, "Persistence", t_test, y_test, persist_pred)
    rows.append(
        {
            "station": station,
            "model": "Persistence",
            "status": "ok",
            "n_rows_strict": len(df),
            "n_samples": len(y),
            "n_train": splits.n_train,
            "n_val": splits.n_val,
            "n_test": splits.n_test,
            **m,
        }
    )

    Xr_train = Xb_train.reshape(Xb_train.shape[0], -1)
    Xr_test = Xb_test.reshape(Xb_test.shape[0], -1)
    w = fit_ridge_closed_form(Xr_train, y_train_s, alpha=ridge_alpha)
    ridge_pred_s = predict_ridge(Xr_test, w)
    ridge_pred = y_scaler.inverse_transform(ridge_pred_s.reshape(-1, 1)).reshape(-1)
    m = evaluate_and_save(pred_dir, station, "Ridge", t_test, y_test, ridge_pred)
    rows.append(
        {
            "station": station,
            "model": "Ridge",
            "status": "ok",
            "n_rows_strict": len(df),
            "n_samples": len(y),
            "n_train": splits.n_train,
            "n_val": splits.n_val,
            "n_test": splits.n_test,
            **m,
        }
    )

    lstm_base = train_lstm(
        X_train=Xb_train.astype(np.float32),
        y_train=y_train_s.astype(np.float32),
        X_val=Xb_val.astype(np.float32),
        y_val=y_val_s.astype(np.float32),
        epochs=lstm_epochs,
        batch_size=lstm_batch_size,
        lr=lstm_lr,
        seed=seed,
        patience=lstm_patience,
    )
    lstm_base_pred_s = predict_lstm(lstm_base, Xb_test.astype(np.float32))
    lstm_base_pred = y_scaler.inverse_transform(lstm_base_pred_s.reshape(-1, 1)).reshape(-1)
    m = evaluate_and_save(pred_dir, station, "LSTM_baseline", t_test, y_test, lstm_base_pred)
    rows.append(
        {
            "station": station,
            "model": "LSTM_baseline",
            "status": "ok",
            "n_rows_strict": len(df),
            "n_samples": len(y),
            "n_train": splits.n_train,
            "n_val": splits.n_val,
            "n_test": splits.n_test,
            **m,
        }
    )

    lstm_gate = train_lstm(
        X_train=Xg_train.astype(np.float32),
        y_train=y_train_s.astype(np.float32),
        X_val=Xg_val.astype(np.float32),
        y_val=y_val_s.astype(np.float32),
        epochs=lstm_epochs,
        batch_size=lstm_batch_size,
        lr=lstm_lr,
        seed=seed,
        patience=lstm_patience,
    )
    lstm_gate_pred_s = predict_lstm(lstm_gate, Xg_test.astype(np.float32))
    lstm_gate_pred = y_scaler.inverse_transform(lstm_gate_pred_s.reshape(-1, 1)).reshape(-1)
    m = evaluate_and_save(pred_dir, station, "LSTM_with_gate", t_test, y_test, lstm_gate_pred)
    rows.append(
        {
            "station": station,
            "model": "LSTM_with_gate",
            "status": "ok",
            "n_rows_strict": len(df),
            "n_samples": len(y),
            "n_train": splits.n_train,
            "n_val": splits.n_val,
            "n_test": splits.n_test,
            **m,
        }
    )

    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simple strict pipeline: clean -> 4 models per station.")
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="Repo root.")
    p.add_argument("--stations", nargs="+", default=DEFAULT_STATIONS, help="Station IDs.")
    p.add_argument("--skip-cleaning", action="store_true", help="Skip cleaning step.")
    p.add_argument("--lookback", type=int, default=48)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--lstm-epochs", type=int, default=30)
    p.add_argument("--lstm-batch-size", type=int, default=64)
    p.add_argument("--lstm-lr", type=float, default=1e-3)
    p.add_argument("--lstm-patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    out_dir = root / "reports" / "simple_pipeline"
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    if not args.skip_cleaning:
        run_cleaning(root, sys.executable)

    clean_dir = find_clean_dir(root)
    print(f"[data] Using cleaned data directory: {clean_dir}")

    rows: list[dict[str, object]] = []
    had_failure = False

    for station in args.stations:
        print(f"[station] {station}")
        try:
            rows.extend(
                run_station_models(
                    station=station,
                    clean_dir=clean_dir,
                    pred_dir=pred_dir,
                    lookback=args.lookback,
                    horizon=args.horizon,
                    val_frac=args.val_frac,
                    test_frac=args.test_frac,
                    ridge_alpha=args.ridge_alpha,
                    lstm_epochs=args.lstm_epochs,
                    lstm_batch_size=args.lstm_batch_size,
                    lstm_lr=args.lstm_lr,
                    lstm_patience=args.lstm_patience,
                    seed=args.seed,
                )
            )
            print(f"[ok] {station}")
        except Exception as exc:
            had_failure = True
            print(f"[fail] {station}: {exc}")
            rows.append(
                {
                    "station": station,
                    "model": "ALL",
                    "status": "failed",
                    "error": str(exc),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    print(f"[done] Metrics -> {out_dir / 'metrics.csv'}")
    print(f"[done] Predictions -> {pred_dir}")
    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
