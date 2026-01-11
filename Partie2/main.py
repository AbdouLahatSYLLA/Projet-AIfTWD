import argparse
import os
import sys
import torch
import flwr as fl
from typing import List, Tuple, Union, Dict, Optional
from flwr.common import Metrics, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
import ssl
from collections import OrderedDict

from torch.utils.data import DataLoader, random_split

# Fix SSL
ssl._create_default_https_context = ssl._create_unverified_context
sys.path.append(os.getcwd())

from src.pre_train import load_data_and_partition
from src.data import CBISDataset, TRANSFORMS_TRAIN, TRANSFORMS_TEST
from src.models import get_model
from src.engine import Trainer
from src.client import FlowerClient


# --- Fonction utilitaire pour sauvegarder le modèle ---
def save_model_from_parameters(parameters: Parameters, model_name: str, save_path: str):
    """Reconstruit le modèle PyTorch depuis les paramètres Flower et le sauvegarde."""
    # 1. Convertir les paramètres Flower (bytes) en Numpy arrays
    weights = parameters_to_ndarrays(parameters)

    # 2. Instancier un modèle vide
    model = get_model(model_name=model_name, num_classes=4, device='cpu')

    # 3. Appliquer les poids
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

    # 4. Sauvegarder
    torch.save(model.state_dict(), save_path)
    print(f"💾 Modèle global sauvegardé : {save_path}")


# --- STRATÉGIES PERSONNALISÉES AVEC SAUVEGARDE ---

class SaveModelFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, run_id, model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = run_id
        self.model_name = model_name

    def aggregate_fit(self, server_round: int, results, failures):
        # Appel de la méthode parent pour faire l'agrégation standard
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Sauvegarde du modèle à chaque round (écrase le précédent ou crée un historique)
            save_path = f"models/{self.run_id}_round_{server_round}.pth"
            # On sauvegarde aussi le "best/last" générique
            latest_path = f"models/{self.run_id}_latest.pth"

            # Sauvegarde seulement tous les 10 rounds ou au dernier round pour économiser l'espace
            if server_round % 10 == 0:
                save_model_from_parameters(aggregated_parameters, self.model_name, save_path)

            # Toujours mettre à jour le "latest"
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


# --- Accuracy moyenne ---
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    if not examples: return {"accuracy": 0}
    return {"accuracy": sum(accuracies) / sum(examples)}


# ==========================================
# MODE CENTRALISÉ
# ==========================================
def run_centralized(args, full_df, device):
    # ... (src identique à avant) ...
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
    trainer.train(train_loader, epochs=args.epochs, lr=args.lr, mode='standard')

    loss, acc = trainer.evaluate(test_loader)
    print(f"🏆 CENTRALIZED RESULT: Accuracy = {acc * 100:.2f}% | Loss = {loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{args.train_id}.pth")


# ==========================================
# MODE FÉDÉRÉ
# ==========================================
def run_federated(args, client_dfs, device):
    print(f"\n🌐 --- MODE FÉDÉRÉ (Algo: {args.algo} | Model: {args.model}) ---")
    os.makedirs("models", exist_ok=True)  # Créer le dossier models

    def client_fn(cid: str):
        # Force GC pour éviter OOM sur ResNet50
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if cid not in client_dfs:
            df_client = client_dfs['0']
        else:
            df_client = client_dfs[cid]

        dataset_full = CBISDataset(df_client, None, transform=None)
        tr_len = int(0.8 * len(dataset_full))
        tr_subset, val_subset = random_split(dataset_full, [tr_len, len(dataset_full) - tr_len])

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

    # --- SÉLECTION DE LA STRATÉGIE AVEC SAUVEGARDE ---
    if args.algo == 'fedadam':
        print("⚙️ Initialisation des poids globaux pour FedAdam...")
        temp_model = get_model(model_name=args.model, num_classes=4, device='cpu')
        initial_parameters = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in temp_model.state_dict().items()]
        )

        strategy = SaveModelFedAdam(
            run_id=args.train_id,
            model_name=args.model,
            initial_parameters=initial_parameters,
            eta=0.2, eta_l=args.lr, tau=1e-3,
            fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=3,
            on_fit_config_fn=fit_config, evaluate_metrics_aggregation_fn=weighted_average
        )
    else:
        # FedAvg et FedProx utilisent SaveModelFedAvg
        strategy = SaveModelFedAvg(
            run_id=args.train_id,
            model_name=args.model,
            fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=3,
            on_fit_config_fn=fit_config, evaluate_metrics_aggregation_fn=weighted_average
        )

    use_gpu_resource = 1.0 if torch.cuda.is_available() else 0.0
    print(f"🖥️ Configuration Ressources Ray : {use_gpu_resource} GPU par client")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=args.epochs),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": use_gpu_resource}
    )


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