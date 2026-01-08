"""GCN-LSTM training run for tensor inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
TENSOR_VALUES = DATA_DIR / "tensor_values.csv"
ADJ_MATRIX = DATA_DIR / "adj_matrix.csv"


INPUT_WINDOW = 24
OUTPUT_HORIZON = 12
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3



class SequenceDataset(Dataset):

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def load_tensor_values(path: Path) -> tuple[np.ndarray, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run Phase 1 export first.")
    df = pd.read_csv(path)
    stations = list(df.columns[1:])
    values = df[stations].values.astype(np.float32)
    return values, stations


def load_adjacency(path: Path, stations: list[str]) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; build the graph adjacency first.")
    df = pd.read_csv(path, index_col=0)
    df = df.reindex(index=stations, columns=stations, fill_value=0.0)
    mat = df.values.astype(np.float32)
    mat += np.eye(len(stations), dtype=np.float32)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(mat.sum(axis=1) + 1e-6))
    norm = deg_inv_sqrt @ mat @ deg_inv_sqrt
    return torch.from_numpy(norm)


def create_sequences(data: np.ndarray, input_window: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    num_steps, num_nodes = data.shape
    samples = num_steps - input_window - horizon + 1
    if samples <= 0:
        raise ValueError("Not enough rows to build sliding windows. Add more data or reduce window sizes.")
    X = np.zeros((samples, input_window, num_nodes), dtype=np.float32)
    Y = np.zeros((samples, horizon, num_nodes), dtype=np.float32)
    for i in range(samples):
        X[i] = data[i : i + input_window]
        Y[i] = data[i + input_window : i + input_window + horizon]
    return X, Y


def split_train_val_test(X: np.ndarray, Y: np.ndarray):
    total = X.shape[0]
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    return (
        (X[:train_end], Y[:train_end]),
        (X[train_end:val_end], Y[train_end:val_end]),
        (X[val_end:], Y[val_end:]),
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
    def __init__(self, num_nodes: int, adj: torch.Tensor, gcn_hidden: int = 16, lstm_hidden: int = 128):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = OUTPUT_HORIZON
        self.gcn = GraphConv(1, gcn_hidden, adj)
        self.lstm = nn.LSTM(num_nodes * gcn_hidden, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, self.horizon * num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, num_nodes = x.shape
        x = x.unsqueeze(-1)
        gcn_out = []
        for t in range(seq_len):
            gcn_out.append(self.gcn(x[:, t]))
        h = torch.stack(gcn_out, dim=1)
        h = h.reshape(bsz, seq_len, -1)
        lstm_out, _ = self.lstm(h)
        last = lstm_out[:, -1, :]
        pred = self.fc(last)
        return pred.view(bsz, self.horizon, num_nodes)



def train():
    values, stations = load_tensor_values(TENSOR_VALUES)
    adj = load_adjacency(ADJ_MATRIX, stations)

    scaler = StandardScaler()
    train_split_idx = int(len(values) * TRAIN_RATIO)
    scaler.fit(values[:train_split_idx])
    values_norm = scaler.transform(values)

    X, Y = create_sequences(values_norm, INPUT_WINDOW, OUTPUT_HORIZON)
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_train_val_test(X, Y)

    loaders = {
        "train": DataLoader(SequenceDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True),
        "val": DataLoader(SequenceDataset(X_val, Y_val), batch_size=BATCH_SIZE),
        "test": DataLoader(SequenceDataset(X_test, Y_test), batch_size=BATCH_SIZE),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNLSTM(len(stations), adj.to(device)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_state = None
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(loaders["train"].dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in loaders["val"]:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= max(1, len(loaders["val"].dataset))

        print(f"Epoch {epoch:02d} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

    if best_state is None:
        raise RuntimeError("Training never improved; check dataset size.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = REPORTS_DIR / "best_model.pth"
    torch.save(
        {
            "model_state": best_state,
            "stations": stations,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        model_path,
    )
    print(f"Saved best checkpoint (val {best_val:.5f}) -> {model_path}")

    model.load_state_dict(best_state)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in loaders["test"]:
            xb, yb = xb.to(device), yb.to(device)
            test_loss += criterion(model(xb), yb).item() * xb.size(0)
    test_loss /= max(1, len(loaders["test"].dataset))
    print(f"Test Loss: {test_loss:.5f}")


if __name__ == "__main__":
    train()
