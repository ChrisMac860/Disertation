"""Baseline single-station LSTM training."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras import models, layers, callbacks

BASE_DIR = Path(__file__).resolve().parents[1]
CLEAN_DIR = BASE_DIR / "data" / "clean"

STATION_NAME = "AnDinh"

LOOKBACK = 48
TEST_FRACTION = 0.2
VAL_FRACTION = 0.2

np.random.seed(42)


station_file = CLEAN_DIR / f"station_{STATION_NAME}.csv"
if not station_file.exists():
    raise FileNotFoundError(f"Could not find cleaned file for station {STATION_NAME!r}: {station_file}")

df = pd.read_csv(station_file, index_col="datetime", parse_dates=True)

print(f"Loaded station file: {station_file}")
print("\nHead:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nTime coverage and shape:")
print(f"Start: {df.index.min()}")
print(f"End:   {df.index.max()}")
print(f"Shape: {df.shape}")

if "salinity" not in df.columns:
    raise KeyError("Expected 'salinity' column in cleaned station dataset.")

plt.figure(figsize=(12, 4))
df["salinity"].plot()
plt.title(f"Salinity time series at station {STATION_NAME}")
plt.ylabel("Salinity")
plt.xlabel("Datetime")
plt.tight_layout()
plt.show()


non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

station_code_cols = [c for c in df.columns if c.endswith("Station Code")]

cols_to_drop = sorted(set(non_numeric_cols).union(station_code_cols))
if cols_to_drop:
    print("Dropping non-numeric / station code columns:")
    print(cols_to_drop)
    df = df.drop(columns=cols_to_drop)

target_col = "salinity"
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found after dropping non-numeric columns.")

feature_cols = [c for c in df.columns if c != target_col]

print(f"\nNumber of feature columns: {len(feature_cols)}")
print("Feature columns:")
print(feature_cols)

missing_pct = df[feature_cols + [target_col]].isna().mean() * 100
print("\nMissingness (% NaN) per column:")
print(missing_pct.sort_values(ascending=False).round(2))


df = df.dropna(subset=[target_col])

if feature_cols:
    before_dropna = len(df)
    df_before_feature_drop = df.copy()
    df = df.dropna(subset=feature_cols)
    dropped_rows = before_dropna - len(df)
    print(f"Rows dropped due to NaNs in features: {dropped_rows}")

    if len(df) == 0:
        print(
            "All rows were dropped when requiring complete data for all drivers.\n"
            "Falling back to using salinity-only (autoregressive) input."
        )
        df = df_before_feature_drop[[target_col]].dropna()
        feature_cols = []

print("\nFinal cleaned shape and coverage:")
print(f"Shape: {df.shape}")
print(f"Start: {df.index.min()}")
print(f"End:   {df.index.max()}")

if len(df) <= LOOKBACK + 10:
    raise ValueError(
        f"Not enough rows ({len(df)}) after cleaning for lookback={LOOKBACK}. "
        "Consider reducing LOOKBACK or relaxing missing-data thresholds."
    )


def create_sequences(data: np.ndarray, target: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    T, F = data.shape
    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    for t in range(lookback - 1, T - 1):
        window = data[t - lookback + 1 : t + 1]
        X_list.append(window)
        y_list.append(target[t + 1])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    return X, y


all_feature_cols = [target_col] + feature_cols
data_all = df[all_feature_cols].values.astype("float32")
target_all = df[target_col].values.astype("float32")

X_all, y_all = create_sequences(data_all, target_all, lookback=LOOKBACK)

time_index_all = df.index[LOOKBACK:]

print("\nSequence shapes:")
print(f"X_all shape: {X_all.shape}  (samples, lookback, n_features)")
print(f"y_all shape: {y_all.shape}")
print(f"Number of timestamps: {len(time_index_all)}")


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


def scale_features(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
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


salinity_feature_index = all_feature_cols.index(target_col)

y_pred_persist = X_test_raw[:, -1, salinity_feature_index]

y_true_test = y_test_raw

mae_persist = mean_absolute_error(y_true_test, y_pred_persist)
rmse_persist = mean_squared_error(y_true_test, y_pred_persist) ** 0.5
r2_persist = r2_score(y_true_test, y_pred_persist)

print("Persistence baseline metrics (test set, original units):")
print(f"MAE:  {mae_persist:.4f}")
print(f"RMSE: {rmse_persist:.4f}")
print(f"R²:   {r2_persist:.4f}")


n_features = X_train_scaled.shape[2]

model = models.Sequential(
    [
        layers.Input(shape=(LOOKBACK, n_features)),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ]
)

model.compile(loss="mse", optimizer="adam")

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

history = model.fit(
    X_train_scaled,
    y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1,
)

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title(f"LSTM training history for station {STATION_NAME}")
plt.legend()
plt.tight_layout()
plt.show()


y_pred_test_scaled = model.predict(X_test_scaled).reshape(-1, 1)

y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled).reshape(-1)
y_true_test = y_test_raw

mae_lstm = mean_absolute_error(y_true_test, y_pred_test)
rmse_lstm = mean_squared_error(y_true_test, y_pred_test) ** 0.5
r2_lstm = r2_score(y_true_test, y_pred_test)

print("LSTM metrics (test set, original units):")
print(f"MAE:  {mae_lstm:.4f}")
print(f"RMSE: {rmse_lstm:.4f}")
print(f"R²:   {r2_lstm:.4f}")

metrics_df = pd.DataFrame(
    {
        "MAE": [mae_persist, mae_lstm],
        "RMSE": [rmse_persist, rmse_lstm],
        "R2": [r2_persist, r2_lstm],
    },
    index=["Persistence", "LSTM"],
)

print("\nTest-set metrics comparison:")
print(metrics_df.round(4))

n_plot = min(500, len(y_true_test))
time_plot = time_test[:n_plot]

plt.figure(figsize=(12, 4))
plt.plot(time_plot, y_true_test[:n_plot], label="Observed", linewidth=1)
plt.plot(time_plot, y_pred_persist[:n_plot], label="Persistence", linewidth=1)
plt.plot(time_plot, y_pred_test[:n_plot], label="LSTM", linewidth=1)
plt.xlabel("Datetime")
plt.ylabel("Salinity")
plt.title(f"Test period comparison at station {STATION_NAME}")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
plt.scatter(y_true_test, y_pred_test, alpha=0.4, s=10)
max_val = max(y_true_test.max(), y_pred_test.max())
min_val = min(y_true_test.min(), y_pred_test.min())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="1:1 line")
plt.xlabel("Observed salinity")
plt.ylabel("Predicted salinity (LSTM)")
plt.title(f"Observed vs predicted salinity at station {STATION_NAME}")
plt.legend()
plt.tight_layout()
plt.show()

