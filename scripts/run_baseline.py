"""Persistence baseline summary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data") / "tensor_values.csv"
REPORT_PATH = Path("reports") / "baseline_metrics.csv"
HORIZON = 24


def persistence_baseline(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    targets = df.shift(-horizon)
    preds = df.shift(-(horizon + 0))
    actual = df
    pred = df.shift(horizon)

    errors = actual - pred
    mae = errors.abs().mean(skipna=True)
    mse = (errors**2).mean(skipna=True)
    return pd.DataFrame({"station": df.columns, "MAE": mae.values, "MSE": mse.values})


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run Phase 1 export first.")

    raw = pd.read_csv(DATA_PATH)
    if raw.shape[1] < 2:
        raise ValueError("tensor_values.csv must have a datetime column + at least one station.")

    values = raw.iloc[:, 1:]
    metrics = persistence_baseline(values, HORIZON)
    global_mae = metrics["MAE"].mean()
    global_mse = metrics["MSE"].mean()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(REPORT_PATH, index=False)

    print(f"Global MAE (24h persistence): {global_mae:.4f}")
    print(f"Global MSE (24h persistence): {global_mse:.4f}")


if __name__ == "__main__":
    main()
