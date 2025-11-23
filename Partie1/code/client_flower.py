import flwr as fl
import torch
from torch.utils.data import DataLoader
import sys
import os
import argparse
from collections import OrderedDict

# Resolve paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add parent directory to system path to find Dataset.py
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import prepared data frames
try:
    from pre_train import client_dataframes
except ImportError:
    # Fallback import path
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    sys.path.append(parent_dir)
    from pre_train import client_dataframes

from Dataset import CBISDataset, data_transforms
from ResNet18 import get_model
from train import train, test


class CBISClient(fl.client.NumPyClient):
    def __init__(self, client_id, device):
        self.client_id = client_id
        self.device = device
        print(f"📂 Initializing data for: {self.client_id}")

        # Load Non-IID data specific to this client
        if self.client_id not in client_dataframes:
            raise ValueError(f"Client ID '{self.client_id}' unknown. Use client1, client2 or client3.")

        df = client_dataframes[self.client_id]

        # Create PyTorch Dataset
        full_dataset = CBISDataset(df, transform=data_transforms)

        # Split local Train/Test (80% / 20%)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        self.train_set, self.test_set = torch.utils.data.random_split(
            full_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.model = get_model().to(self.device)

    def get_parameters(self, config):
        # Return model parameters as NumPy arrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # Update local model with parameters from server
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"🔄 Local training ({self.client_id})...")
        self.set_parameters(parameters)

        # Local DataLoader
        train_loader = DataLoader(self.train_set, batch_size=32, shuffle=True)

        # Train model (1 epoch per federated round)
        train(self.model, train_loader, epochs=1, device=self.device)

        return self.get_parameters(config={}), len(self.train_set), {}

    def evaluate(self, parameters, config):
        print(f"📊 Local evaluation ({self.client_id})...")
        self.set_parameters(parameters)

        test_loader = DataLoader(self.test_set, batch_size=32, shuffle=False)

        loss, accuracy = test(self.model, test_loader, device=self.device)
        return float(loss), len(self.test_set), {"accuracy": float(accuracy)}


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Flower Client')
    parser.add_argument('--cid', type=str, required=True, help='Client ID (client1, client2, client3)')
    parser.add_argument('--server', type=str, default="127.0.0.1:8080", help='Server address')
    args = parser.parse_args()

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"🚀 Starting Client {args.cid} on {device}")

    # Start client
    client = CBISClient(args.cid, device)

    fl.client.start_numpy_client(server_address=args.server, client=client, grpc_max_message_length=1024*1024*1024)