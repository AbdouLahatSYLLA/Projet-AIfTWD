import pandas as pd
import numpy as np
import os
import sys


def load_data_and_partition():
    """
    Charge le dataset, respecte le split Train/Test officiel,
    et génère des partitions Non-IID pour les clients en gardant les 4 classes.
    """
    # 1. Chargement du CSV
    POSSIBLE_PATHS = [
        "Partie2/dataset/cleaned_dataset.csv",
        "dataset/cleaned_dataset.csv",
        "/kaggle/working/dataset/cleaned_dataset.csv"
    ]

    CLEAN_CSV = None
    for path in POSSIBLE_PATHS:
        if os.path.exists(path):
            CLEAN_CSV = path
            break

    if not CLEAN_CSV:
        print("❌ Erreur: 'cleaned_dataset.csv' introuvable.")
        sys.exit(1)

    print(f"📂 Chargement des données depuis {CLEAN_CSV}...")
    df = pd.read_csv(CLEAN_CSV)

    # 2. SÉPARATION STRICTE TRAIN / TEST OFFICIELLE
    # On s'assure de ne jamais toucher au Test Set pour l'entraînement (Centralisé ou Fédéré)
    if 'split' in df.columns:
        print("ℹ️ Colonne 'split' détectée : Séparation stricte Train/Test.")
        df_train = df[df['split'] == 'train'].copy()
        df_test_official = df[df['split'] == 'test'].copy()
    else:
        print("⚠️ Pas de colonne 'split'. Utilisation de tout le dataset (Risque de Leakage).")
        df_train = df.copy()
        df_test_official = pd.DataFrame()

    print(f"📊 Dataset Global : {len(df)} images")
    print(f"   ↳ Train Set (Distribué aux Clients) : {len(df_train)} images")
    print(f"   ↳ Test Set (Réservé au Serveur/Eval): {len(df_test_official)} images")

    # 3. Création d'une colonne helper pour le tri Non-IID
    # Les classes sont : 0=Calc-Ben, 1=Calc-Mal, 2=Mass-Ben, 3=Mass-Mal
    # On regroupe les Malins (1,3) vs Bénins (0,2) pour simuler le biais Hôpital vs Dépistage.
    df_train['is_malignant'] = df_train['target'].isin([1, 3]).astype(int)

    # 4. Partitionnement logique par Patient (Anti-Leakage)
    # On garantit qu'un patient est entierement chez un seul client
    patient_profiles = df_train.groupby('patient_id')['is_malignant'].agg(['mean', 'count']).reset_index()

    # Un patient est "Malin" s'il a au moins une image maligne, "Bénin" sinon
    malignant_patients = patient_profiles[patient_profiles['mean'] > 0]['patient_id'].tolist()
    benign_patients = patient_profiles[patient_profiles['mean'] == 0]['patient_id'].tolist()

    # 5. Attribution Non-IID (Simulation réaliste)
    np.random.seed(42)

    # --- Client 0 : Oncologie (Cible: Classes 1 & 3) ---
    # Reçoit 70% des patients malades, et seulement 5% des sains
    n_c0_mal = int(0.7 * len(malignant_patients))
    n_c0_ben = int(0.05 * len(benign_patients))

    c0_patients = (
            np.random.choice(malignant_patients, n_c0_mal, replace=False).tolist() +
            np.random.choice(benign_patients, n_c0_ben, replace=False).tolist()
    )

    remaining_mal = list(set(malignant_patients) - set(c0_patients))
    remaining_ben = list(set(benign_patients) - set(c0_patients))

    # --- Client 1 : Dépistage (Cible: Classes 0 & 2) ---
    # Reçoit 80% des patients sains restants
    n_c1_ben = int(0.8 * len(remaining_ben))
    n_c1_mal = int(0.1 * len(remaining_mal))

    c1_patients = (
            np.random.choice(remaining_ben, n_c1_ben, replace=False).tolist() +
            np.random.choice(remaining_mal, n_c1_mal, replace=False).tolist()
    )

    # --- Client 2 : Généraliste (Mixte) ---
    # Prend tout ce qui reste
    c2_patients = list(set(remaining_ben) - set(c1_patients)) + list(set(remaining_mal) - set(c1_patients))

    # 6. Construction des DataFrames Clients (On garde les cibles 0-3 intactes)
    clients = {}
    clients['0'] = df_train[df_train['patient_id'].isin(c0_patients)].copy()
    clients['1'] = df_train[df_train['patient_id'].isin(c1_patients)].copy()
    clients['2'] = df_train[df_train['patient_id'].isin(c2_patients)].copy()

    # 7. Rapport de distribution détaillé (4 Classes)
    print("\n--- Distribution Non-IID (Train Set - 4 Classes) ---")
    classes_name = {0: "Calc-Ben", 1: "Calc-Mal", 2: "Mass-Ben", 3: "Mass-Mal"}

    for cid, cdf in clients.items():
        total = len(cdf)
        # Compte par classe (0, 1, 2, 3)
        counts = cdf['target'].value_counts().sort_index()

        if cid == '0':
            label = "Oncologie"
        elif cid == '1':
            label = "Dépistage"
        else:
            label = "Généraliste"

        print(f"Client {cid} [{label}] - Total: {total}")
        for cls, name in classes_name.items():
            count = counts.get(cls, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            # Affiche une barre visuelle si le pourcentage est élevé
            bar = "█" * int(percentage / 10)
            print(f"   - {name} ({cls}): {count:3d} ({percentage:5.1f}%) {bar}")

    # On retourne uniquement df_train et les partitions clients
    # Le Test Set est écarté ici, il pourra être chargé séparément pour l'évaluation finale si besoin
    return df_train, clients


if __name__ == "__main__":
    load_data_and_partition()