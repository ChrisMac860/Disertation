"""GCN-LSTM training pipeline for Mekong tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

TENSOR_VALUES = DATA_DIR / "tensor_values.csv"
TENSOR_MASK = DATA_DIR / "tensor_mask.csv"
ADJ_MATRIX = DATA_DIR / "adj_matrix.csv"

INPUT_LEN = 72
OUTPUT_LEN = 24
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3



class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, M: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.M = torch.from_numpy(M).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx], self.M[idx]


def load_values_and_mask() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TENSOR_VALUES.exists():
        raise FileNotFoundError(f"Missing {TENSOR_VALUES}")

    df_values = pd.read_csv(TENSOR_VALUES, index_col=0, parse_dates=True)

    if TENSOR_MASK.exists():
        raw_mask = pd.read_csv(TENSOR_MASK, index_col=0)
        raw_mask = raw_mask.reindex(columns=df_values.columns, fill_value=0.0)
        raw_mask = raw_mask.reset_index(drop=True)
        if len(raw_mask) < len(df_values):
            pad = np.zeros((len(df_values) - len(raw_mask), raw_mask.shape[1]))
            raw_mask = pd.DataFrame(
                np.vstack([raw_mask.values, pad]),
                columns=df_values.columns,
            )
        df_mask = pd.DataFrame(
            raw_mask.values[: len(df_values)],
            index=df_values.index,
            columns=df_values.columns,
        )
    else:
        df_mask = (~df_values.isna()).astype(float)
        df_mask.to_csv(TENSOR_MASK)
        print(f"No tensor_mask.csv found; generated placeholder mask at {TENSOR_MASK}")

    return df_values, df_mask


def load_adjacency(stations: list[str]) -> torch.Tensor:
    if not ADJ_MATRIX.exists():
        raise FileNotFoundError(f"Missing {ADJ_MATRIX}")

    adj = pd.read_csv(ADJ_MATRIX, index_col=0)
    adj = adj.reindex(index=stations, columns=stations, fill_value=0.0)
    mat = adj.values.astype(np.float32)
    mat += np.eye(len(stations), dtype=np.float32)
    deg_inv = np.diag(1.0 / np.sqrt(mat.sum(axis=1) + 1e-6))
    norm = deg_inv @ mat @ deg_inv
    return torch.from_numpy(norm)


def create_windows(values: np.ndarray, mask: np.ndarray, input_len: int, output_len: int):
    num_steps, num_nodes = values.shape
    total = num_steps - input_len - output_len + 1
    if total <= 0:
        raise ValueError("Not enough rows to create sliding windows.")

    X = np.zeros((total, input_len, num_nodes), dtype=np.float32)
    Y = np.zeros((total, output_len, num_nodes), dtype=np.float32)
    M = np.zeros((total, output_len, num_nodes), dtype=np.float32)

    for i in range(total):
        X[i] = values[i : i + input_len]
        Y[i] = values[i + input_len : i + input_len + output_len]
        M[i] = mask[i + input_len : i + input_len + output_len]
    return X, Y, M


def split_windows(X, Y, M):
    total = X.shape[0]
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    return (
        (X[:train_end], Y[:train_end], M[:train_end]),
        (X[train_end:val_end], Y[train_end:val_end], M[train_end:val_end]),
        (X[val_end:], Y[val_end:], M[val_end:]),
    )



class GraphConv(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, adj: torch.Tensor):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=False)
        self.register_buffer("adj", adj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        return torch.relu(torch.einsum("ij,bjk->bik", self.adj, h))


class GCNLSTM(nn.Module):
    def __init__(self, num_nodes: int, adj: torch.Tensor, gcn_hidden: int = 32, lstm_hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = OUTPUT_LEN
        self.gcn = GraphConv(1, gcn_hidden, adj)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(num_nodes * gcn_hidden, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, self.horizon * num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, num_nodes = x.shape
        x = x.unsqueeze(-1)
        gcn_frames = []
        for t in range(seq_len):
            gcn_frames.append(self.gcn(x[:, t]))
        h = torch.stack(gcn_frames, dim=1).reshape(bsz, seq_len, -1)
        h = self.dropout(h)
        lstm_out, _ = self.lstm(h)
        last = lstm_out[:, -1, :]
        pred = self.fc(self.dropout(last))
        return pred.view(bsz, self.horizon, num_nodes)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    return (diff * mask).sum() / (mask.sum() + 1e-6)



def main() -> None:
    df_values, df_mask = load_values_and_mask()
    stations = list(df_values.columns)

    df_filled = df_values.ffill().bfill().fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(df_filled.values[: int(len(df_filled) * TRAIN_RATIO)])
    values_scaled = scaler.transform(df_filled.values).astype(np.float32)
    mask_values = df_mask[stations].values.astype(np.float32)

    X, Y, M = create_windows(values_scaled, mask_values, INPUT_LEN, OUTPUT_LEN)
    (X_train, Y_train, M_train), (X_val, Y_val, M_val), (X_test, Y_test, M_test) = split_windows(X, Y, M)

    loaders = {
        "train": DataLoader(WindowDataset(X_train, Y_train, M_train), batch_size=BATCH_SIZE, shuffle=True),
        "val": DataLoader(WindowDataset(X_val, Y_val, M_val), batch_size=BATCH_SIZE),
        "test": DataLoader(WindowDataset(X_test, Y_test, M_test), batch_size=BATCH_SIZE),
    }

    adj = load_adjacency(stations)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNLSTM(len(stations), adj.to(device)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_state = None
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, mb in loaders["train"]:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            xb = xb + torch.randn_like(xb) * 0.01
            optimizer.zero_grad()
            pred = model(xb)
            loss = masked_mse(pred, yb, mb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(loaders["train"].dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, mb in loaders["val"]:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                val_loss += masked_mse(model(xb), yb, mb).item() * xb.size(0)
        val_loss /= max(1, len(loaders["val"].dataset))

        print(f"Epoch {epoch:02d} | Train {train_loss:.5f} | Val {val_loss:.5f}")
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

    if best_state is None:
        raise RuntimeError("Training failed to produce an improved model.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best_state,
        "stations": stations,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
    }
    out_path = REPORTS_DIR / "best_model.pth"
    torch.save(checkpoint, out_path)
    print(f"Saved best checkpoint (val {best_val:.5f}) -> {out_path}")

    model.load_state_dict(best_state)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb, mb in loaders["test"]:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            test_loss += masked_mse(model(xb), yb, mb).item() * xb.size(0)
    test_loss /= max(1, len(loaders["test"].dataset))
    print(f"Test Loss: {test_loss:.5f}")


if __name__ == "__main__":
    main()
