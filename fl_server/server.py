import flwr as fl


def main() -> None:
    """Start the federated learning server."""

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=5,
        min_available_clients=5,
        min_evaluate_clients=5,
    )

    config = fl.server.ServerConfig(num_rounds=10)

    print("Starting Flower server on 0.0.0.0:8080 ...")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )


if __name__ == "__main__":
    main()