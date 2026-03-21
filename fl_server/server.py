import flwr as fl
import csv
import os
from datetime import datetime

LOG_FILE = "results/fl_round_logs.csv"

def weighted_average(metrics):
    """Aggregate metrics from all clients."""
    total_examples = sum(num for num, _ in metrics)
    maes = [num * m["mae"] for num, m in metrics]
    rmses = [num * m["rmse"] for num, m in metrics]
    return {
        "mae": sum(maes) / total_examples,
        "rmse": sum(rmses) / total_examples,
    }

def get_evaluate_fn():
    """Return a function that logs metrics after each round."""
    os.makedirs("results", exist_ok=True)

    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "loss", "mae", "rmse", "timestamp"])

    def evaluate(server_round, parameters, config):
        # Placeholder — sẽ dùng thật sau khi có global model
        print(f"[Round {server_round}] Server-side evaluation called")
        return None

    return evaluate

def main() -> None:
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=5,
        min_available_clients=5,
        min_evaluate_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(),
    )

    config = fl.server.ServerConfig(num_rounds=10)

    print("Starting Flower server on 0.0.0.0:8080 ...")
    print(f"Logging rounds to: {LOG_FILE}")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
