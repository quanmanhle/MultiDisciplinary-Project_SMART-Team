import csv
import os
from datetime import datetime
from typing import Any

import flwr as fl

LOG_FILE = "results/fl_round_logs.csv"


def weighted_average(metrics: list[tuple[int, dict[str, Any]]]) -> dict[str, float]:
    """Aggregate client metrics using number of examples as weights."""
    if not metrics:
        return {"mae": 0.0, "rmse": 0.0}

    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"mae": 0.0, "rmse": 0.0}

    weighted_mae = sum(
        num_examples * float(client_metrics.get("mae", 0.0))
        for num_examples, client_metrics in metrics
    )
    weighted_rmse = sum(
        num_examples * float(client_metrics.get("rmse", 0.0))
        for num_examples, client_metrics in metrics
    )

    return {
        "mae": weighted_mae / total_examples,
        "rmse": weighted_rmse / total_examples,
    }


class CsvLoggingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy that logs aggregated evaluation metrics to CSV."""

    def __init__(self, log_file: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.log_file = log_file

        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "loss", "mae", "rmse", "timestamp"])

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate evaluation results and write one row per round to CSV."""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        mae = ""
        rmse = ""

        if aggregated_metrics is not None:
            mae = aggregated_metrics.get("mae", "")
            rmse = aggregated_metrics.get("rmse", "")

        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    server_round,
                    aggregated_loss if aggregated_loss is not None else "",
                    mae,
                    rmse,
                    datetime.now().isoformat(timespec="seconds"),
                ]
            )

        print(
            f"[LOG] Round {server_round} | "
            f"loss={aggregated_loss} | mae={mae} | rmse={rmse}"
        )

        return aggregated_loss, aggregated_metrics


def main() -> None:
    strategy = CsvLoggingFedAvg(
        log_file=LOG_FILE,
        min_fit_clients=5,
        min_available_clients=5,
        min_evaluate_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
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