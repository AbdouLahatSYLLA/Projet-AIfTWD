import matplotlib.pyplot as plt
import json
import pickle
import os

# ==========================================
# ⚙️ CONFIGURATION DES FICHIERS À TRACER
# ==========================================
# Ajoutez ici les noms des runs que vous avez lancés
# Format : ("Nom Légende", "Chemin fichier", "Type")
# Type = 'central' (json) ou 'fed' (pkl)

EXPERIMENTS = [
    # Baseline Centralisée (Référence)
    ("Centralisé (ResNet50)", "logs/centralized_resnet50_history.json", "central"),

    # Expérience 1 : Robustesse (FedAvg vs FedProx)
    ("Federated FedAvg (Baseline)", "logs/fedavg_noniid_baseline_history.pkl", "fed"),
    ("Federated FedProx (Robust)", "logs/fedprox_noniid_robust_history.pkl", "fed"),

    # Expérience 2 : Privacy
    # ("Federated DP (Privé)", "logs/dp_resnet50_secure_history.pkl", "fed"),
]


def load_data(path, exp_type):
    """Charge les données selon le format."""
    if not os.path.exists(path):
        print(f"⚠️ Fichier introuvable : {path}")
        return None, None, None

    if exp_type == 'central':
        with open(path, 'r') as f:
            data = json.load(f)
            # Centralisé : liste simple [0.8, 0.85, ...]
            # On crée l'axe X (epochs)
            rounds = list(range(1, len(data['accuracy']) + 1))
            return rounds, data['loss'], data['accuracy']

    elif exp_type == 'fed':
        with open(path, 'rb') as f:
            history = pickle.load(f)
            # Fédéré : liste de tuples [(round, value), ...]
            # On sépare X et Y

            # 1. Accuracy (Metrics distributed)
            if 'accuracy' in history.metrics_distributed:
                r_acc, acc = zip(*history.metrics_distributed['accuracy'])
            else:
                r_acc, acc = [], []

            # 2. Loss (Losses distributed)
            # Note: Flower stocke la loss d'entraînement agrégée ici
            if history.losses_distributed:
                r_loss, loss = zip(*history.losses_distributed)
            else:
                # Si pas de loss distribuée, on prend la loss centralisée (si dispo)
                if history.losses_centralized:
                    r_loss, loss = zip(*history.losses_centralized)
                else:
                    r_loss, loss = [], []

            return r_acc, loss, acc


def plot_metrics():
    plt.figure(figsize=(16, 6))

    # --- GRAPHIQUE 1 : LOSS ---
    plt.subplot(1, 2, 1)
    for label, path, exp_type in EXPERIMENTS:
        rounds, loss, _ = load_data(path, exp_type)
        if rounds is None or not loss: continue

        # Astuce : Centralisé a souvent plus de points (par epoch) que Fédéré (par round)
        # On trace tout pour comparer la vitesse de convergence
        linestyle = '--' if exp_type == 'central' else '-'
        linewidth = 2 if exp_type == 'fed' else 1.5

        plt.plot(rounds, loss, label=label, linestyle=linestyle, linewidth=linewidth)

    plt.title("📉 Évolution de la Loss (Erreur)")
    plt.xlabel("Rounds / Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- GRAPHIQUE 2 : ACCURACY ---
    plt.subplot(1, 2, 2)
    for label, path, exp_type in EXPERIMENTS:
        rounds, _, acc = load_data(path, exp_type)
        if rounds is None or not acc: continue

        linestyle = '--' if exp_type == 'central' else '-'
        linewidth = 2 if exp_type == 'fed' else 1.5

        plt.plot(rounds, acc, label=label, linestyle=linestyle, linewidth=linewidth)

    plt.title("📈 Évolution de l'Accuracy (Précision)")
    plt.xlabel("Rounds / Epochs")
    plt.ylabel("Accuracy (0-1)")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)  # Accuracy est toujours entre 0 et 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_metrics()