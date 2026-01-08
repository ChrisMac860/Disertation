"""Gate-index LSTM for single-station forecasting."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras import models, layers, callbacks

ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "data" / "clean"

STATION_NAME = "AnDinh"
STATION_FILE = CLEAN_DIR / f"station_{STATION_NAME}.csv"

LOCAL_H_COL = "H_MyTho_Value"
TIDE_H_COL = "H_VamKenh_Value"
RAIN_COL = "rain_MyTho_Value"


GATE_WINDOW_HOURS = 48
RAIN_WINDOW_HOURS = 24
K_AMP = 1.0
R0 = 10.0

LOOKBACK = 48
VAL_FRACTION = 0.2
TEST_FRACTION = 0.2

BATCH_SIZE = 64
EPOCHS = 30

assert VAL_FRACTION + TEST_FRACTION < 1.0, "Validation + test fraction must be < 1."



if not STATION_FILE.exists():
    raise FileNotFoundError(f"Cleaned station file not found: {STATION_FILE}")

df = pd.read_csv(STATION_FILE, index_col="datetime", parse_dates=True)

print(f"Loaded station file for {STATION_NAME}: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())
print("\nColumns:")
print(df.columns.tolist())

for col in [LOCAL_H_COL, TIDE_H_COL, RAIN_COL, "salinity"]:
    if col not in df.columns:
        raise ValueError(f"Required column {col!r} not found in dataset.")

print("\nTime coverage:")
print("Start:", df.index.min())
print("End:  ", df.index.max())

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
df["salinity"].plot(ax=axes[0])
axes[0].set_ylabel("Salinity")
axes[0].set_title(f"Salinity at {STATION_NAME}")

df[LOCAL_H_COL].plot(ax=axes[1])
axes[1].set_ylabel("Local H")

df[TIDE_H_COL].plot(ax=axes[2])
axes[2].set_ylabel("Boundary H")
axes[2].set_xlabel("Datetime")

plt.tight_layout()
plt.show()


h_local = df[LOCAL_H_COL].astype(float)
h_tide = df[TIDE_H_COL].astype(float)
rain = df[RAIN_COL].astype(float)

W = GATE_WINDOW_HOURS

min_corr_points = 3
corr_HT = h_local.rolling(window=W, min_periods=min_corr_points).corr(h_tide)

std_local = h_local.rolling(window=W, min_periods=min_corr_points).std()
std_tide = h_tide.rolling(window=W, min_periods=min_corr_points).std()
amp_ratio = std_local / (std_tide + 1e-6)
amp_ratio_norm = (amp_ratio / K_AMP).clip(lower=0.0, upper=1.0)

C_tide = corr_HT.clip(lower=0.0).fillna(0.0)
I_tide = C_tide * amp_ratio_norm.fillna(0.0)

rain_nonnull = rain.dropna()
if not rain_nonnull.empty:
    rain_std = float(rain_nonnull.std())
    nonzero_count = int((rain_nonnull != 0).sum())
    nonzero_frac = nonzero_count / len(rain_nonnull)
else:
    rain_std = 0.0
    nonzero_frac = 0.0

if rain_std <= 1e-3 or nonzero_frac <= 0.01:
    print(
        f"Rain series {RAIN_COL!r} has std={rain_std:.4g}, "
        f"nonzero_frac={nonzero_frac:.4g} -> treating rainfall as unusable "
        "and setting S_rain = 1.0."
    )
    S_rain = pd.Series(1.0, index=df.index)
else:
    rain_filled = rain.fillna(0.0)
    R_sum = rain_filled.rolling(window=RAIN_WINDOW_HOURS, min_periods=1).sum()
    S_rain = np.exp(-R_sum / R0)

gate_index = I_tide * S_rain
df["gate_index"] = gate_index

print(df[["salinity", LOCAL_H_COL, TIDE_H_COL, RAIN_COL, "gate_index"]].head(60))

gi = df["gate_index"]
print("\nGate index summary:")
print(gi.describe())
print("\nAny outside [0, 1]?", ((gi < 0) | (gi > 1)).any())
print("NaN fraction:", gi.isna().mean())

print("\nCorrelations:")
print("gate_index vs LOCAL_H:", gi.corr(df[LOCAL_H_COL]))
print("gate_index vs TIDE_H :", gi.corr(df[TIDE_H_COL]))
print("gate_index vs RAIN   :", gi.corr(df[RAIN_COL]))

fig, ax = plt.subplots(figsize=(12, 4))
df["gate_index"].plot(ax=ax)
ax.set_title(f"Gate index at station {STATION_NAME}")
ax.set_ylabel("gate_index (0–1)")
plt.tight_layout()
plt.show()



non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
drop_cols = set(non_numeric)
drop_cols.update([col for col in df.columns if "Station Code" in col])

if drop_cols:
    print("Dropping non-numeric / ID columns:", drop_cols)
    df = df.drop(columns=list(drop_cols))

target_col = "salinity"
if target_col not in df.columns:
    raise ValueError("salinity column not found after cleaning.")

baseline_feature_cols = [
    c for c in df.columns
    if c != target_col and c != "gate_index"
]

df = df.dropna(subset=[target_col])

if baseline_feature_cols:
    before_dropna = len(df)
    df_before = df.copy()
    df = df.dropna(subset=baseline_feature_cols)
    dropped_rows = before_dropna - len(df)
    print(f"\nRows dropped due to NaNs in baseline features: {dropped_rows}")

    if len(df) == 0:
        print(
            "All rows were dropped when requiring complete data for baseline drivers.\n"
            "Falling back to using salinity-only (autoregressive) input, "
            "while retaining gate_index for the augmented model."
        )
        kept_cols = [col for col in [target_col, "gate_index"] if col in df_before.columns]
        df = df_before[kept_cols].dropna(subset=[target_col])
        baseline_feature_cols = []

print("\nFinal cleaned shape and coverage (baseline logic):")
print(f"Shape: {df.shape}")
print(f"Start: {df.index.min()}")
print(f"End:   {df.index.max()}")

if len(df) <= LOOKBACK + 10:
    raise ValueError(
        f"Not enough rows ({len(df)}) after cleaning for lookback={LOOKBACK}."
    )

if "gate_index" not in df.columns:
    raise ValueError("gate_index not present; check gate index construction.")

gate_feature_cols = baseline_feature_cols + ["gate_index"]

print("\nNumber of features (baseline):", len(baseline_feature_cols))
print("Number of features (with gate_index):", len(gate_feature_cols))


def create_sequences(data, target, lookback):
    X, y = [], []
    T = len(target)
    for t in range(lookback - 1, T - 1):
        X.append(data[t - lookback + 1 : t + 1])
        y.append(target[t + 1])
    return np.array(X), np.array(y)


def run_lstm_experiment(X_all, y_all, time_index_all, label):
    print(f"\n=== Running LSTM experiment: {label} ===")

    n_samples = X_all.shape[0]
    n_test = int(np.floor(TEST_FRACTION * n_samples))
    n_val = int(np.floor(VAL_FRACTION * n_samples))
    n_train = n_samples - n_val - n_test

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes with n_samples={n_samples}, "
            f"TEST_FRACTION={TEST_FRACTION}, VAL_FRACTION={VAL_FRACTION}."
        )

    train_end = n_train
    val_end = n_train + n_val

    X_train_raw = X_all[:train_end]
    X_val_raw = X_all[train_end:val_end]
    X_test_raw = X_all[val_end:]

    y_train_raw = y_all[:train_end]
    y_val_raw = y_all[train_end:val_end]
    y_test_raw = y_all[val_end:]

    time_train = time_index_all[:train_end]
    time_val = time_index_all[train_end:val_end]
    time_test = time_index_all[val_end:]

    print("Split sizes (samples):")
    print(f"Train: {X_train_raw.shape[0]}")
    print(f"Val:   {X_val_raw.shape[0]}")
    print(f"Test:  {X_test_raw.shape[0]}")
    print(f"Total: {n_samples}")

    n_features = X_train_raw.shape[2]
    scaler_X = StandardScaler()
    X_train_flat = X_train_raw.reshape(-1, n_features)
    scaler_X.fit(X_train_flat)

    def scale_features(X, scaler):
        n, L, F = X.shape
        X_flat = X.reshape(-1, F)
        X_scaled_flat = scaler.transform(X_flat)
        return X_scaled_flat.reshape(n, L, F)

    X_train_scaled = scale_features(X_train_raw, scaler_X)
    X_val_scaled = scale_features(X_val_raw, scaler_X)
    X_test_scaled = scale_features(X_test_raw, scaler_X)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).reshape(-1)
    y_val_scaled = scaler_y.transform(y_val_raw.reshape(-1, 1)).reshape(-1)
    y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).reshape(-1)

    print("\nScaled shapes:")
    print(f"X_train_scaled: {X_train_scaled.shape}")
    print(f"X_val_scaled:   {X_val_scaled.shape}")
    print(f"X_test_scaled:  {X_test_scaled.shape}")
    print(f"y_train_scaled: {y_train_scaled.shape}")

    model = models.Sequential([
        layers.Input(shape=(LOOKBACK, n_features)),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    es = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=1,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.history["loss"], label="Train loss")
    ax.plot(history.history["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"LSTM training history ({label}) at station {STATION_NAME}")
    ax.legend()
    plt.tight_layout()
    plt.show()

    y_pred_test_scaled = model.predict(X_test_scaled).reshape(-1, 1)
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled).reshape(-1)
    y_test_orig = y_test_raw

    mae = mean_absolute_error(y_test_orig, y_pred_test)
    rmse_val = float(np.sqrt(mean_squared_error(y_test_orig, y_pred_test)))
    r2 = r2_score(y_test_orig, y_pred_test)

    print(f"Test MAE ({label}):  {mae:.4f}")
    print(f"Test RMSE ({label}): {rmse_val:.4f}")
    print(f"Test R² ({label}):   {r2:.4f}")

    return {
        "label": label,
        "mae": mae,
        "rmse": rmse_val,
        "r2": r2,
        "y_test": y_test_orig,
        "y_pred": y_pred_test,
        "time_test": time_test,
    }


all_feature_cols_base = [target_col] + baseline_feature_cols
data_all_base = df[all_feature_cols_base].values.astype("float32")
target_all = df[target_col].values.astype("float32")
X_all_base, y_all_base = create_sequences(data_all_base, target_all, LOOKBACK)
time_index_all = df.index[LOOKBACK:]

print("\nBaseline sequences:")
print("X_all_base shape:", X_all_base.shape, "y_all_base shape:", y_all_base.shape)

df_gate = df.copy()
df_gate["gate_index"] = df_gate["gate_index"].fillna(0.0)
all_feature_cols_gate = [target_col] + gate_feature_cols
data_all_gate = df_gate[all_feature_cols_gate].values.astype("float32")
X_all_gate, y_all_gate = create_sequences(data_all_gate, target_all, LOOKBACK)

print("\nGate-augmented sequences:")
print("X_all_gate shape:", X_all_gate.shape, "y_all_gate shape:", y_all_gate.shape)

baseline_results = run_lstm_experiment(X_all_base, y_all_base, time_index_all, label="LSTM_baseline")

gate_results = run_lstm_experiment(X_all_gate, y_all_gate, time_index_all, label="LSTM_with_gate")

metrics_df = pd.DataFrame(
    [
        {"Model": baseline_results["label"],
         "MAE": baseline_results["mae"],
         "RMSE": baseline_results["rmse"],
         "R2": baseline_results["r2"]},
        {"Model": gate_results["label"],
         "MAE": gate_results["mae"],
         "RMSE": gate_results["rmse"],
         "R2": gate_results["r2"]},
    ]
)
print("\nTest-set metrics comparison (original salinity units):")
print(metrics_df)

y_test = baseline_results["y_test"]
y_pred_base = baseline_results["y_pred"]
y_pred_gate = gate_results["y_pred"]

n_plot = min(500, len(y_test))

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(y_test[:n_plot], label="Observed", linewidth=1)
ax.plot(y_pred_base[:n_plot], label="LSTM baseline", linewidth=1)
ax.plot(y_pred_gate[:n_plot], label="LSTM with gate_index", linewidth=1)
ax.set_title(f"Test period comparison at station {STATION_NAME}")
ax.set_xlabel("Test sample index")
ax.set_ylabel("Salinity")
ax.legend()
plt.tight_layout()
plt.show()

