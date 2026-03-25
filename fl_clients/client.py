import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET_COL = "main"
HOUSE_ORDER = ["house1", "house2", "house3", "house4", "house6"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def split_time_series(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def find_house_file(project_root: Path, house_name: str) -> Path:
    candidates = [
        project_root / "data" / "processed" / f"{house_name}_hourly_clean.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot find clean data file for {house_name}. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def load_house_xy(project_root: Path, house_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    csv_path = find_house_file(project_root, house_name)
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if TARGET_COL not in df.columns:
        raise ValueError(f"{csv_path.name} does not contain target column '{TARGET_COL}'")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    if not feature_cols:
        raise ValueError(f"{csv_path.name} has no feature columns")

    train_df, test_df = split_time_series(df, train_ratio=0.8)

    train_valid = train_df.dropna(subset=feature_cols + [TARGET_COL]).copy()
    test_valid = test_df.dropna(subset=feature_cols + [TARGET_COL]).copy()

    x_train = train_valid[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_valid[TARGET_COL].to_numpy(dtype=np.float64)

    x_test = test_valid[feature_cols].to_numpy(dtype=np.float64)
    y_test = test_valid[TARGET_COL].to_numpy(dtype=np.float64)

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError(
            f"Not enough valid train/test rows after dropna for {house_name}. "
            f"train={len(x_train)}, test={len(x_test)}"
        )

    return x_train, y_train, x_test, y_test, feature_cols


class HouseLinearClient(fl.client.NumPyClient):
    """Client Flower cho 1 nhà, dùng LinearRegression để khớp local model."""

    def __init__(self, house_name: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, feature_cols: List[str]) -> None:
        self.house_name = house_name
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.feature_cols = feature_cols

        self.n_features = x_train.shape[1]
        self.model = LinearRegression()
        self._initialize_model()

    def _initialize_model(self) -> None:
        # Khởi tạo tham số để round đầu tiên vẫn get/set được
        self.model.coef_ = np.zeros(self.n_features, dtype=np.float64)
        self.model.intercept_ = np.array(0.0, dtype=np.float64)
        self.model.n_features_in_ = self.n_features

    def _set_parameters(self, parameters: List[np.ndarray]) -> None:
        coef = np.array(parameters[0], dtype=np.float64)
        intercept = np.array(parameters[1], dtype=np.float64)

        if coef.shape[0] != self.n_features:
            raise ValueError(
                f"Feature count mismatch for {self.house_name}: "
                f"received coef size={coef.shape[0]}, expected={self.n_features}"
            )

        self.model.coef_ = coef
        self.model.intercept_ = float(intercept.reshape(-1)[0])
        self.model.n_features_in_ = self.n_features

    def _get_parameters(self) -> List[np.ndarray]:
        coef = np.array(self.model.coef_, dtype=np.float64)
        intercept = np.array([self.model.intercept_], dtype=np.float64)
        return [coef, intercept]

    def get_parameters(self, config):
        return self._get_parameters()

    def fit(self, parameters, config):
        self._set_parameters(parameters)

        self.model.fit(self.x_train, self.y_train)
        y_pred_train = self.model.predict(self.x_train)

        train_metrics = compute_metrics(self.y_train, y_pred_train)
        train_metrics.update(
            {
                "house": self.house_name,
                "num_features": float(self.n_features),
            }
        )

        return self._get_parameters(), len(self.x_train), train_metrics

    def evaluate(self, parameters, config):
        self._set_parameters(parameters)

        y_pred = self.model.predict(self.x_test)
        metrics = compute_metrics(self.y_test, y_pred)
        metrics.update(
            {
                "house": self.house_name,
                "num_features": float(self.n_features),
            }
        )

        # Với regression, dùng RMSE làm loss khi evaluate
        loss = metrics["rmse"]
        return float(loss), len(self.x_test), metrics


def resolve_house_name(client_index: int) -> str:
    if client_index < 0 or client_index >= len(HOUSE_ORDER):
        raise ValueError(
            f"client_index must be in [0, {len(HOUSE_ORDER)-1}], got {client_index}"
        )
    return HOUSE_ORDER[client_index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower client for one house in the smart-home task")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--house",
        type=str,
        choices=HOUSE_ORDER,
        help="House name to run as client",
    )
    group.add_argument(
        "--client-index",
        type=int,
        help="Client-to-house mapping: 0->house1, 1->house2, 2->house3, 3->house4, 4->house6",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load data and run one local fit/evaluate pass without connecting to server",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    house_name = args.house if args.house is not None else resolve_house_name(args.client_index)

    x_train, y_train, x_test, y_test, feature_cols = load_house_xy(project_root, house_name)

    print(f"[CLIENT] Running house={house_name}")
    print(f"[CLIENT] Data columns={feature_cols + [TARGET_COL]}")
    print(f"[CLIENT] Train shape: x={x_train.shape}, y={y_train.shape}")
    print(f"[CLIENT] Test shape : x={x_test.shape}, y={y_test.shape}")

    client = HouseLinearClient(
        house_name=house_name,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_cols=feature_cols,
    )

    if args.dry_run:
        init_params = client.get_parameters(config={})
        updated_params, n_train, train_metrics = client.fit(init_params, config={})
        loss, n_test, test_metrics = client.evaluate(updated_params, config={})

        print("[DRY-RUN] Train samples=", n_train)
        print("[DRY-RUN] Train metrics=", train_metrics)
        print("[DRY-RUN] Test samples=", n_test)
        print("[DRY-RUN] Loss eval (rmse)=", loss)
        print("[DRY-RUN] Test metrics=", test_metrics)
        return

    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
