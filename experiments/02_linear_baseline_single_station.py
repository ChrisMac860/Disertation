"""Baseline single-station Ridge regression."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge

BASE_DIR = Path(__file__).resolve().parents[1]
CLEAN_DIR = BASE_DIR / "data" / "clean"

STATION_NAME = "AnDinh"
LOOKBACK = 48
VAL_FRACTION = 0.2
TEST_FRACTION = 0.2

ALPHA = 1.0

if VAL_FRACTION + TEST_FRACTION >= 1.0:
    raise ValueError("VAL_FRACTION + TEST_FRACTION must be < 1.0")

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

print("\nTime coverage and row count:")
print(f"Start: {df.index.min()}")
print(f"End:   {df.index.max()}")
print(f"Number of rows: {len(df)}")

if "salinity" not in df.columns:
    raise KeyError("Expected 'salinity' column in cleaned station dataset.")

plt.figure(figsize=(12, 4))
if len(df) > 1000:
    df["salinity"].iloc[:1000].plot()
    plt.title(f"Salinity at station {STATION_NAME} (first 1000 points)")
else:
    df["salinity"].plot()
    plt.title(f"Salinity at station {STATION_NAME}")
plt.ylabel("Salinity")
plt.xlabel("Datetime")
plt.tight_layout()
plt.show()


non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

station_code_cols = [c for c in df.columns if "Station Code" in c]

cols_to_drop = sorted(set(non_numeric_cols).union(station_code_cols))
if cols_to_drop:
    print("\nDropping non-numeric / station-code columns:")
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


rows_before = len(df)

rows_before_target = len(df)
df = df.dropna(subset=[target_col])
rows_after_target = len(df)
dropped_target = rows_before_target - rows_after_target
print(f"\nRows dropped due to missing target: {dropped_target}")

df_target_only = df[[target_col]].copy()

feature_cols = [c for c in df.columns if c != target_col]

feature_missing_frac = df[feature_cols].isna().mean()
sparse_features = feature_missing_frac[feature_missing_frac > 0.8].index.tolist()

if sparse_features:
    print("Dropping features with >80% missing values:")
    print(sparse_features)
    df = df.drop(columns=sparse_features)
    feature_cols = [c for c in feature_cols if c not in sparse_features]

if not feature_cols:
    raise ValueError(
        "All candidate input features were dropped due to missingness. "
        "Consider loosening the 80% threshold or adding more drivers."
    )

row_missing_frac = df[feature_cols].isna().mean(axis=1)
mask_keep = row_missing_frac <= 0.2
dropped_rows_high_missing = (~mask_keep).sum()
df = df[mask_keep]
print(f"Rows dropped due to >20% missing inputs: {dropped_rows_high_missing}")

rows_before_clean_features = len(df)
df = df.dropna(subset=feature_cols)
dropped_rows_any_missing = rows_before_clean_features - len(df)
print(f"Rows dropped due to remaining NaNs in features: {dropped_rows_any_missing}")

print("\nFinal cleaned shape and coverage:")
print(f"Shape: {df.shape}")
print(f"Start: {df.index.min()}")
print(f"End:   {df.index.max()}")
print(f"Total rows dropped: {rows_before - len(df)}")

if len(df) == 0:
    print(
        "No rows remaining after dropping sparse features and rows with missing inputs.\n"
        "Falling back to salinity-only (autoregressive) input."
    )
    df = df_target_only
    feature_cols = []
    print("\nFallback salinity-only shape and coverage:")
    print(f"Shape: {df.shape}")
    print(f"Start: {df.index.min()}")
    print(f"End:   {df.index.max()}")
elif len(df) <= LOOKBACK + 10:
    raise ValueError(
        f"Not enough rows ({len(df)}) after cleaning for lookback={LOOKBACK}. "
        "Consider reducing LOOKBACK or relaxing missing-data thresholds."
    )


all_feature_cols = [target_col] + feature_cols
data_all = df[all_feature_cols].values.astype("float32")
target = df[target_col].values.astype("float32")


def create_sequences(data: np.ndarray, target: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    T = len(target)

    for t in range(lookback - 1, T - 1):
        X_list.append(data[t - lookback + 1 : t + 1])
        y_list.append(target[t + 1])

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


X_all, y_all = create_sequences(data_all, target, lookback=LOOKBACK)
time_index_all = df.index[LOOKBACK:]

print("\nSequence shapes:")
print(f"X_all shape: {X_all.shape}  (samples, lookback, n_features)")
print(f"y_all shape: {y_all.shape}")


N = X_all.shape[0]
n_test = int(N * TEST_FRACTION)
n_val = int(N * VAL_FRACTION)
n_train = N - n_val - n_test

if n_train <= 0 or n_val <= 0 or n_test <= 0:
    raise ValueError(
        f"Invalid split sizes with N={N}, VAL_FRACTION={VAL_FRACTION}, TEST_FRACTION={TEST_FRACTION}."
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
print(f"Total: {N}")


n_features = X_train_raw.shape[2]

feature_scaler = StandardScaler()
target_scaler = StandardScaler()

X_train_flat_for_fit = X_train_raw.reshape(-1, n_features)
feature_scaler.fit(X_train_flat_for_fit)


def scale_feature_windows(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, L, F = X.shape
    X_flat = X.reshape(-1, F)
    X_scaled_flat = scaler.transform(X_flat)
    return X_scaled_flat.reshape(n, L, F)


X_train_scaled = scale_feature_windows(X_train_raw, feature_scaler)
X_val_scaled = scale_feature_windows(X_val_raw, feature_scaler)
X_test_scaled = scale_feature_windows(X_test_raw, feature_scaler)

y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
y_val_scaled = target_scaler.transform(y_val_raw.reshape(-1, 1)).ravel()
y_test_scaled = target_scaler.transform(y_test_raw.reshape(-1, 1)).ravel()

print("\nScaled shapes:")
print(f"X_train_scaled: {X_train_scaled.shape}")
print(f"X_val_scaled:   {X_val_scaled.shape}")
print(f"X_test_scaled:  {X_test_scaled.shape}")
print(f"y_train_scaled: {y_train_scaled.shape}")


sal_idx = all_feature_cols.index(target_col)

y_pred_persist_scaled = X_test_scaled[:, -1, sal_idx]

y_pred_persist = target_scaler.inverse_transform(y_pred_persist_scaled.reshape(-1, 1)).ravel()
y_test_orig = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


mae_persist = mean_absolute_error(y_test_orig, y_pred_persist)
rmse_persist = rmse(y_test_orig, y_pred_persist)
r2_persist = r2_score(y_test_orig, y_pred_persist)

print("Persistence baseline metrics (test set, original units):")
print(f"MAE:  {mae_persist:.4f}")
print(f"RMSE: {rmse_persist:.4f}")
print(f"R²:   {r2_persist:.4f}")


n_train, L, F = X_train_scaled.shape
n_val = X_val_scaled.shape[0]
n_test = X_test_scaled.shape[0]

X_train_flat = X_train_scaled.reshape(n_train, L * F)
X_val_flat = X_val_scaled.reshape(n_val, L * F)
X_test_flat = X_test_scaled.reshape(n_test, L * F)

ridge = Ridge(alpha=ALPHA)
ridge.fit(X_train_flat, y_train_scaled)

val_r2_scaled = ridge.score(X_val_flat, y_val_scaled)
print(f"\nRidge validation R² (scaled targets): {val_r2_scaled:.4f}")

y_pred_ridge_scaled = ridge.predict(X_test_flat)

y_pred_ridge = target_scaler.inverse_transform(y_pred_ridge_scaled.reshape(-1, 1)).ravel()

mae_ridge = mean_absolute_error(y_test_orig, y_pred_ridge)
rmse_ridge = rmse(y_test_orig, y_pred_ridge)
r2_ridge = r2_score(y_test_orig, y_pred_ridge)

print("Ridge regression metrics (test set, original units):")
print(f"MAE:  {mae_ridge:.4f}")
print(f"RMSE: {rmse_ridge:.4f}")
print(f"R²:   {r2_ridge:.4f}")


metrics_df = pd.DataFrame(
    {
        "MAE": [mae_persist, mae_ridge],
        "RMSE": [rmse_persist, rmse_ridge],
        "R2": [r2_persist, r2_ridge],
    },
    index=["Persistence", "Ridge"],
)

print("\nTest-set metrics comparison (original salinity units):")
print(metrics_df.round(4))

n_plot = min(500, len(y_test_orig))
idx_plot = slice(0, n_plot)

plt.figure(figsize=(12, 4))
plt.plot(time_test[idx_plot], y_test_orig[idx_plot], label="Observed", linewidth=1)
plt.plot(time_test[idx_plot], y_pred_persist[idx_plot], label="Persistence", linewidth=1)
plt.plot(time_test[idx_plot], y_pred_ridge[idx_plot], label="Ridge", linewidth=1)
plt.xlabel("Datetime")
plt.ylabel("Salinity")
plt.title(f"Test period comparison at station {STATION_NAME}")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
plt.scatter(y_test_orig, y_pred_ridge, alpha=0.4, s=10)
min_val = float(min(y_test_orig.min(), y_pred_ridge.min()))
max_val = float(max(y_test_orig.max(), y_pred_ridge.max()))
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="1:1 line")
plt.xlabel("Observed salinity")
plt.ylabel("Predicted salinity (Ridge)")
plt.title(f"Observed vs predicted salinity at station {STATION_NAME}")
plt.legend()
plt.tight_layout()
plt.show()

