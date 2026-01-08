"""Build adjacency and edge lists from station coordinates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from geopy.distance import geodesic

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TENSOR_VALUES = DATA_DIR / "tensor_values.csv"
ADJ_MATRIX_OUT = DATA_DIR / "adj_matrix.csv"
EDGE_LIST_OUT = DATA_DIR / "edge_list.csv"

DIST_THRESHOLD_KM = 30.0
SIGMA_KM = 12.0

STATION_COORDS = {
    'TanChau': (10.8000, 105.2333),
    'ChauDoc': (10.7000, 105.1167),
    'VamNao': (10.5833, 105.3000),
    'MyThuan': (10.2769, 105.9044),
    'CanTho': (10.0333, 105.7833),
    'DaiNgai': (9.7333, 106.1333),
    'TranDe': (9.5500, 106.2167),
    'MyTho': (10.3500, 106.3500),
    'VamKenh': (10.2700, 106.7400),
    'BinhDai': (10.1970, 106.7110),
    'BenTrai': (9.8810, 106.5290),
    'AnThuan': (9.9760, 106.6050),
    'GanhHao': (9.0310, 105.4190),
    'RachGia': (10.0120, 105.0840),
    'CaMau': (9.1769, 105.1524),
    'BacLieu': (9.2941, 105.7278),
    'SocTrang': (9.6033, 105.9722),
    'TraVinh': (9.9342, 106.3456),
    'BenLuc': (10.6386, 106.4825),
    'TanAn': (10.5422, 106.4113),
    'HoaBinh': (10.3000, 106.6000),
    'HungMy': (10.0000, 106.0000),
    'PhuocLong': (9.5667, 105.6000),
    'ViThanh': (9.7840, 105.4670),
    'PhungHiep': (9.8333, 105.8333),
    'MyHoa': (10.0000, 106.0000),
    'SongDoc': (9.0410, 104.8330),
    'XeoRo': (9.8650, 105.1110),
    'LongXuyen': (10.3719, 105.4263),
    'ChoLach': (10.2500, 106.1333),
    'CaiBe': (10.3333, 106.0333),
    'CaiLay': (10.4167, 106.1167),
    'ChoMoi': (10.4833, 105.4667),
    'TanHiep': (10.1180, 105.2850),
    'AnDinh': (9.9500, 106.3000),
    'AnLacTay': (9.9000, 106.2000),
    'AnNinh': (9.8000, 106.1000),
    'Batri': (10.0500, 106.6000),
    'CauNoi': (9.8500, 106.4000),
    'CauQuan': (9.7167, 106.1500),
    'DongTam': (10.3500, 106.3000),
    'GiongTrom': (10.1333, 106.4667),
    'GoQuao': (9.7500, 105.3333),
    'HuongMy': (9.9833, 106.3833),
    'KhanhThanhTan': (9.6000, 105.5000),
    'LocThuan': (10.1667, 106.7167),
    'LongPhu': (9.6500, 106.1333),
    'LuynhQuynh': (9.5000, 105.5000),
    'SonDoc': (10.0500, 106.5000),
    'TamNgan': (9.5000, 105.5000),
    'ThanhPhu': (9.8833, 106.5333),
    'ThoiBinh': (9.3500, 105.1000),
    'TPBacLieu': (9.2941, 105.7278),
    'TraKha': (9.5833, 106.3333),
    'TuyenNhon': (10.8500, 106.3333),
    'VungLiem': (10.1167, 106.1833),
    'XuanKhanh': (10.0333, 105.7667)
}


def load_station_ids(path: Path) -> list[str]:
    df = pd.read_csv(path, nrows=1)
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError("tensor_values.csv must contain datetime plus station columns.")
    return cols[1:]


def ensure_coords(stations: list[str]) -> dict[str, tuple[float, float]]:
    coords = {}
    missing = []
    for sid in stations:
        coord = STATION_COORDS.get(sid)
        if coord is None:
            missing.append(sid)
            coord = (0.0, 0.0)
        coords[sid] = coord
    if missing:
        print(f"Warning: missing coordinates for {len(missing)} stations: {missing}")
    return coords


def pairwise_distances(stations: list[str], coords: dict[str, tuple[float, float]]) -> np.ndarray:
    n = len(stations)
    dist = np.zeros((n, n), dtype=float)
    for i, sid_i in enumerate(stations):
        for j in range(i, n):
            sid_j = stations[j]
            if sid_i == sid_j:
                d = 0.0
            else:
                d = geodesic(coords[sid_i], coords[sid_j]).kilometers
            dist[i, j] = dist[j, i] = d
    return dist


def gaussian_adjacency(distances: np.ndarray, threshold: float, sigma: float) -> np.ndarray:
    mask = (distances > 0) & (distances < threshold)
    weights = np.exp(-(distances**2) / (sigma**2))
    return mask * weights


def export_adj_matrix(stations: list[str], matrix: np.ndarray, out_path: Path) -> None:
    df = pd.DataFrame(matrix, index=stations, columns=stations)
    df.to_csv(out_path, float_format="%.6f")


def export_edge_list(stations: list[str], matrix: np.ndarray, out_path: Path) -> None:
    edges = []
    n = len(stations)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            weight = matrix[i, j]
            if weight <= 0:
                continue
            edges.append({"source": stations[i], "target": stations[j], "weight": weight})
    pd.DataFrame(edges).to_csv(out_path, index=False, float_format="%.6f")


def main() -> None:
    if not TENSOR_VALUES.exists():
        raise FileNotFoundError(f"Missing {TENSOR_VALUES}. Run Phase 1 export first.")

    station_ids = load_station_ids(TENSOR_VALUES)
    coords = ensure_coords(station_ids)
    distances = pairwise_distances(station_ids, coords)
    adj = gaussian_adjacency(distances, DIST_THRESHOLD_KM, SIGMA_KM)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    export_adj_matrix(station_ids, adj, ADJ_MATRIX_OUT)
    export_edge_list(station_ids, adj, EDGE_LIST_OUT)
    print(f"Wrote {ADJ_MATRIX_OUT} and {EDGE_LIST_OUT}")


if __name__ == "__main__":
    main()
