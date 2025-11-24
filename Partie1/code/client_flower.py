import flwr as fl
import torch
from torch.utils.data import DataLoader
import sys
import os
import argparse
from collections import OrderedDict
import pickle

# Resolve paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path: sys.path.append(parent_dir)

try:
    from pre_train import client_dataframes
except ImportError:
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    sys.path.append(parent_dir)
    from pre_train import client_dataframes

from Dataset import CBISDataset, data_transforms
from ResNet18 import get_model
from train import train, test


def _save_stats(stats, log_dir, filename):
    filepath = os.path.join(log_dir, filename)
    with open(filepath, "ab") as f:
        pickle.dump(stats, f)


class CBISClient(fl.client.NumPyClient):
    def __init__(self, client_id, device, log_dir, epochs):
        self.client_id = client_id
        self.device = device
        self.log_dir = log_dir
        self.epochs = epochs
        self.stats_filename = f"stats_{self.client_id}.pkl"

        # Clean up on startup
        full_path = os.path.join(self.log_dir, self.stats_filename)
        if os.path.exists(full_path):
            os.remove(full_path)

        print(f"Initializing data for: {self.client_id}")

        if self.client_id not in client_dataframes:
            raise ValueError(f"Client ID '{self.client_id}' unknown.")

        df = client_dataframes[self.client_id]
        full_dataset = CBISDataset(df, transform=data_transforms)

        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        self.train_set, self.test_set = torch.utils.data.random_split(
            full_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.model = get_model().to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Retrieve global round sent by server
        server_round = config["current_round"]
        print(f"Local training ({self.client_id}) - Round {server_round}...")

        self.set_parameters(parameters)
        train_loader = DataLoader(self.train_set, batch_size=32, shuffle=True)

        history = train(self.model, train_loader, epochs=self.epochs, device=self.device)

        # Save with correct round number (important for plotting)
        for stat in history:
            # Replace local 'epoch' (always 1) with global round
            stat['round'] = server_round
            stat['round_type'] = 'fit'
            _save_stats(stat, self.log_dir, self.stats_filename)

        return self.get_parameters(config={}), len(self.train_set), {}

    def evaluate(self, parameters, config):
        print(f"Local evaluation ({self.client_id})...")
        self.set_parameters(parameters)
        test_loader = DataLoader(self.test_set, batch_size=32, shuffle=False)
        loss, accuracy = test(self.model, test_loader, device=self.device)
        return float(loss), len(self.test_set), {"accuracy": float(accuracy)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cid', type=str, required=True, help='Client ID: client{1,2,3}')
    parser.add_argument('--server', type=str, default="127.0.0.1:8080", help='Server address')
    parser.add_argument("--run_id", type=str, required=True, help="Run ID (ex: 0)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of local epochs per round")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Manage logs directory
    log_dir = os.path.join("Partie1", "logs", "federated", "clients", args.run_id)
    # Ensure directory exists
    os.makedirs(log_dir, exist_ok=True)

    print(f"Starting Client {args.cid} on {device} (Logs: {log_dir})")

    client = CBISClient(args.cid, device, log_dir, epochs=args.epochs)
    fl.client.start_numpy_client(server_address=args.server, client=client, grpc_max_message_length=1024 * 1024 * 1024)