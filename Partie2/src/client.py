import flwr as fl
import torch
from torch.utils.data import DataLoader
from .models import get_model
from .engine import Trainer


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, train_loader, val_loader, device, epochs, lr, mode, mu, dp_settings, model_name):
        self.cid = cid
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.mode = mode
        self.mu = mu
        self.dp_settings = dp_settings
        self.model_name = model_name  # On le stocke

        # On passe model_name à get_model ⬇️
        self.model = get_model(model_name=self.model_name, num_classes=4, use_dp=(mode == 'dp'), device=device)
        self.trainer = Trainer(self.model, device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        server_round = config.get("current_round", 1)
        mu_val = config.get("proximal_mu", self.mu)

        results = self.trainer.train(
            self.train_loader,
            epochs=self.epochs,
            lr=self.lr,
            mode=self.mode,
            mu=mu_val,
            dp_settings=self.dp_settings
        )
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"loss": float(results['loss'])}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.trainer.evaluate(self.val_loader)
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy)}