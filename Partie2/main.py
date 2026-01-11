import argparse
import os
import sys
import torch
import flwr as fl
from typing import List, Tuple, Union, Dict, Optional
from flwr.common import Metrics, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
import ssl
from collections import OrderedDict
import json
import pickle

from torch.utils.data import DataLoader, random_split

# Fix SSL
ssl._create_default_https_context = ssl._create_unverified_context
sys.path.append(os.getcwd())

from src.pre_train import load_data_and_partition
from src.data import CBISDataset, TRANSFORMS_TRAIN, TRANSFORMS_TEST
from src.models import get_model
from src.engine import Trainer
from src.client import FlowerClient


# ==========================================
# 💾 FONCTIONS DE SAUVEGARDE
# ==========================================
def save_model_from_parameters(parameters: Parameters, model_name: str, save_path: str):
    weights = parameters_to_ndarrays(parameters)
    model = get_model(model_name=model_name, num_classes=4, device='cpu')
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    torch.save(model.state_dict(), save_path)
    print(f"💾 Modèle global sauvegardé : {save_path}")


def save_history(history, filename):
    """Sauvegarde l'historique d'entraînement en JSON ou Pickle."""
    os.makedirs("logs", exist_ok=True)
    path = f"logs/{filename}"

    # Si c'est un objet History de Flower, on le pickle
    if isinstance(history, fl.server.history.History):
        with open(path + ".pkl", "wb") as f:
            pickle.dump(history, f)
    else:
        # Sinon (dict standard), on le met en JSON
        with open(path + ".json", "w") as f:
            json.dump(history, f)
    print(f"📊 Historique sauvegardé : {path}")


# --- STRATÉGIES ---
class SaveModelFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, run_id, model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = run_id
        self.model_name = model_name

    def aggregate_fit(self, server_round: int, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            save_path = f"models/{self.run_id}_round_{server_round}.pth"
            latest_path = f"models/{self.run_id}_latest.pth"
            if server_round % 10 == 0:
                save_model_from_parameters(aggregated_parameters, self.model_name, save_path)
            save_model_from_parameters(aggregated_parameters, self.model_name, latest_path)
        return aggregated_parameters, aggregated_metrics


class SaveModelFedAdam(fl.server.strategy.FedAdam):
    def __init__(self, run_id, model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = run_id
        self.model_name = model_name

    def aggregate_fit(self, server_round: int, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            latest_path = f"models/{self.run_id}_latest.pth"
            save_model_from_parameters(aggregated_parameters, self.model_name, latest_path)
        return aggregated_parameters, aggregated_metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    if not examples: return {"accuracy": 0}
    return {"accuracy": sum(accuracies) / sum(examples)}


# ==========================================
# MODE CENTRALISÉ
# ==========================================
def run_centralized(args, full_df, device):
    print(f"\n🚀 --- MODE CENTRALISÉ (ID: {args.train_id}) | MODEL: {args.model} ---")
    dataset_full = CBISDataset(full_df, None, transform=None)
    train_size = int(0.8 * len(dataset_full))
    test_size = len(dataset_full) - train_size
    train_subset, test_subset = random_split(dataset_full, [train_size, test_size],
                                             generator=torch.Generator().manual_seed(42))

    train_set = CBISDataset(full_df.iloc[train_subset.indices], None, transform=TRANSFORMS_TRAIN)
    test_set = CBISDataset(full_df.iloc[test_subset.indices], None, transform=TRANSFORMS_TEST)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = get_model(model_name=args.model, num_classes=4, use_dp=False, device=device)
    trainer = Trainer(model, device=device)

    print(f"🔄 Training for {args.epochs} epochs...")
    # Capture de l'historique
    history = trainer.train(train_loader, epochs=args.epochs, lr=args.lr, mode='standard')

    loss, acc = trainer.evaluate(test_loader)
    print(f"🏆 CENTRALIZED RESULT: Accuracy = {acc * 100:.2f}% | Loss = {loss:.4f}")

    # Sauvegardes
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{args.train_id}.pth")
    save_history(history, f"{args.train_id}_history")


# ==========================================
# MODE FÉDÉRÉ
# ==========================================
def run_federated(args, client_dfs, device):
    print(f"\n🌐 --- MODE FÉDÉRÉ (Algo: {args.algo} | Model: {args.model}) ---")
    os.makedirs("models", exist_ok=True)

    def client_fn(cid: str):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if cid not in client_dfs:
            df_client = client_dfs['0']
        else:
            df_client = client_dfs[cid]

        dataset_full = CBISDataset(df_client, None, transform=None)
        tr_len = int(0.8 * len(dataset_full))

        tr_subset, val_subset = random_split(
            dataset_full,
            [tr_len, len(dataset_full) - tr_len],
            generator=torch.Generator().manual_seed(int(cid) + 42)
        )

        tr_set = CBISDataset(df_client.iloc[tr_subset.indices], None, transform=TRANSFORMS_TRAIN)
        val_set = CBISDataset(df_client.iloc[val_subset.indices], None, transform=TRANSFORMS_TEST)

        client_mode = 'fedprox' if args.algo == 'fedprox' else ('dp' if args.dp else 'standard')

        return FlowerClient(
            cid=cid,
            train_loader=DataLoader(tr_set, batch_size=args.batch_size, shuffle=True, num_workers=0),
            val_loader=DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0),
            device=device,
            epochs=1,
            lr=args.lr,
            mode=client_mode,
            mu=args.mu,
            dp_settings={'noise': args.dp_noise, 'clip': args.dp_clip} if args.dp else None,
            model_name=args.model
        )

    def fit_config(server_round: int):
        config = {"current_round": server_round}
        if args.algo == 'fedprox':
            config["proximal_mu"] = args.mu
        else:
            config["proximal_mu"] = 0.0
        return config

    if args.algo == 'fedadam':
        temp_model = get_model(model_name=args.model, num_classes=4, device='cpu')
        initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in temp_model.state_dict().items()])
        strategy = SaveModelFedAdam(
            run_id=args.train_id,
            model_name=args.model,
            initial_parameters=initial_parameters,
            eta=0.2, eta_l=args.lr, tau=1e-3,
            fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=3,
            on_fit_config_fn=fit_config, evaluate_metrics_aggregation_fn=weighted_average
        )
    else:
        strategy = SaveModelFedAvg(
            run_id=args.train_id,
            model_name=args.model,
            fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=3,
            on_fit_config_fn=fit_config, evaluate_metrics_aggregation_fn=weighted_average
        )

    use_gpu_resource = 1.0 if torch.cuda.is_available() else 0.0
    print(f"🖥️ Configuration Ressources Ray : {use_gpu_resource} GPU par client")

    # Capture de l'objet History retourné par start_simulation
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=args.epochs),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": use_gpu_resource}
    )

    # Sauvegarde
    save_history(hist, f"{args.train_id}_history")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['centralized', 'federated'])
    parser.add_argument('--train_id', type=str, default="exp")
    parser.add_argument('--model', type=str, default="resnet18", choices=['resnet18', 'resnet50', 'resnext50'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--algo', type=str, default="fedavg", choices=['fedavg', 'fedprox', 'fedadam'])
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--dp', action='store_true')
    parser.add_argument('--dp_noise', type=float, default=1.0)
    parser.add_argument('--dp_clip', type=float, default=1.2)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_df, client_partitions = load_data_and_partition()

    if args.mode == 'centralized':
        run_centralized(args, full_df, device)
    else:
        run_federated(args, client_partitions, device)