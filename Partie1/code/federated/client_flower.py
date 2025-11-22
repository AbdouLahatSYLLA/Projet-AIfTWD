import flwr as fl
import torch
from torch.utils.data import DataLoader
import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from Dataset import CBISDataset, data_transforms
from ResNet18 import get_model
from train import train, test
from pre_train import df_combined


class CBISClient(fl.client.NumPyClient):
    def __init__(self, train_data, test_data, device):
        self.device = device
        self.model = get_model().to(device)

        self.train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    def get_parameters(self, config):
        return [p.cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, new_p in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new_p, dtype=p.dtype, device=self.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_loader, epochs=1, device=self.device)

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.test_loader, device=self.device)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}


def load_client_dataset(client_id, num_clients=3):
    full_dataset = CBISDataset(df_combined, transform=data_transforms)
    dataset_size = len(full_dataset)
    chunk = dataset_size // num_clients

    start = client_id * chunk
    end = (client_id + 1) * chunk

    client_data = torch.utils.data.Subset(full_dataset, list(range(start, end)))

    train_size = int(0.8 * len(client_data))
    test_size = len(client_data) - train_size
    return torch.utils.data.random_split(client_data, [train_size, test_size])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Client ID retrieved from command line
    cid = int(sys.argv[1])

    train_data, test_data = load_client_dataset(cid)

    client = CBISClient(train_data, test_data, device)

    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
