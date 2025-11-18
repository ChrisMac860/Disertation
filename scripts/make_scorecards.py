from pathlib import Path
import pandas as pd

ROOT = r"C:\Users\User\Desktop\Disertation"
rep = Path(ROOT) / "reports"
out = rep / "station_scorecards.csv"

base = pd.read_csv(rep / "persistence_baseline.csv")
lstm = pd.read_csv(rep / "models_lstm_metrics.csv")

df = lstm.merge(base, on=["station","h"], suffixes=("_lstm","_persistence"))
df["dMAE"] = df["MAE_lstm"] - df["MAE_persistence"]
df["dRMSE"] = df["RMSE_lstm"] - df["RMSE_persistence"]

df = df[["station","h","MAE_persistence","RMSE_persistence","MAE_lstm","RMSE_lstm","dMAE","dRMSE","window","epochs","seed","model"]]
df.to_csv(out, index=False)
print("Wrote", out)

