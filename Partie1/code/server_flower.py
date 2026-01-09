import flwr as fl
from typing import List, Tuple, Dict
from flwr.common import Metrics, Scalar
import pickle
import argparse
import os

# Configuration
NUM_CLIENTS = 3


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


# Function to send round number to clients
def fit_config(server_round: int) -> Dict[str, Scalar]:
    return {"current_round": server_round}


def _save_stats(stats, log_dir):
    filename = os.path.join(log_dir, "server_stats.pkl")
    with open(filename, "ab") as f:
        pickle.dump(stats, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="Run ID (ex: 0)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated rounds")
    args = parser.parse_args()

    # Create logs directory: Partie1/logs/federated/{id}/
    log_dir = os.path.join("Partie1", "logs", "federated", "server", args.run_id)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Server logs will be saved to: {log_dir}")

    # Clean up existing file if necessary
    if os.path.exists(os.path.join(log_dir, "server_stats.pkl")):
        os.remove(os.path.join(log_dir, "server_stats.pkl"))

    print(f"Starting Flower Server (Waiting for {NUM_CLIENTS} clients) for {args.rounds} rounds...")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config  # Send config to clients
    )

    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        grpc_max_message_length=1024 * 1024 * 1024,
        strategy=strategy
    )

    print("Saving Global Server Stats...")

    if "accuracy" in history.metrics_distributed:
        for round_num, acc in history.metrics_distributed["accuracy"]:
            _save_stats({"round": round_num, "global_accuracy": acc}, log_dir)

    for round_num, loss in history.losses_distributed:
        _save_stats({"round": round_num, "global_loss": loss}, log_dir)

    print(f"Server stats saved in {log_dir}")


if __name__ == "__main__":
    main()