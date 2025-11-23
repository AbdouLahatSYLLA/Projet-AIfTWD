import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics

# Configuration
NUM_CLIENTS = 3  # Server waits for exactly 3 clients


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Aggregates metrics from clients using weighted average

    # Multiply accuracy of each client by number of examples
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]

    # Sum total examples
    examples = [num_examples for num_examples, _ in metrics]

    # Calculate global weighted average
    return {"accuracy": sum(accuracies) / sum(examples)}


def main():
    print(f"Starting Flower Server (Waiting for {NUM_CLIENTS} clients)...")

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # 100% of available clients participate in training
        fraction_evaluate=1.0,  # 100% of available clients participate in evaluation
        min_fit_clients=NUM_CLIENTS,  # Minimum clients required for training
        min_evaluate_clients=NUM_CLIENTS,  # Minimum clients required for evaluation
        min_available_clients=NUM_CLIENTS,  # Wait for N clients before starting
        evaluate_metrics_aggregation_fn=weighted_average
    )

    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=2),  # Number of global updates
        grpc_max_message_length=1024 * 1024 * 1024,
        strategy=strategy
    )


if __name__ == "__main__":
    main()