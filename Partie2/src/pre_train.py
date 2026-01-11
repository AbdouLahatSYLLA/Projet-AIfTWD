import pandas as pd
import numpy as np
import os
import sys


def load_data_and_partition():
    """
    Charge le dataset propre et génère des partitions Non-IID basées sur les patients.
    Garantit l'absence de fuite de données entre clients (split par patient_id).
    """
    # 1. Chargement du CSV
    # On cherche le CSV généré par prepare_dataset.py
    POSSIBLE_PATHS = [
        "Partie2/dataset/cleaned_dataset.csv",
        "../dataset/cleaned_dataset.csv",
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

    # 2. Création de la colonne binaire pour l'analyse Non-IID
    # Classes : 0=Calc-Ben, 1=Calc-Mal, 2=Mass-Ben, 3=Mass-Mal
    # Is Malignant = 1 ou 3
    df['is_malignant'] = df['target'].isin([1, 3]).astype(int)

    # 3. Partitionnement logique par Patient
    # On regroupe par patient pour déterminer son profil pathologique dominant
    patient_profiles = df.groupby('patient_id')['is_malignant'].agg(['mean', 'count']).reset_index()

    # Définition des profils patients :
    # - "Cancer Patient" : A au moins une image maligne
    # - "Benign Patient" : N'a que des images bénignes
    malignant_patients = patient_profiles[patient_profiles['mean'] > 0]['patient_id'].tolist()
    benign_patients = patient_profiles[patient_profiles['mean'] == 0]['patient_id'].tolist()

    print(f"📊 Total Patients : {len(patient_profiles)}")
    print(f"   🤒 Patients avec pathologie maligne : {len(malignant_patients)}")
    print(f"   ✅ Patients sains/bénins : {len(benign_patients)}")

    # 4. Attribution Non-IID (Simulation réaliste)
    np.random.seed(42)

    # Client 0 : "Centre d'Oncologie Spécialisé"
    # -> Reçoit une très grosse majorité de cas malins (Label Skew fort)
    # Prend 70% des patients malins disponibles
    n_c0_mal = int(0.7 * len(malignant_patients))
    # Prend très peu de bénins (5%)
    n_c0_ben = int(0.05 * len(benign_patients))

    c0_patients = (
            np.random.choice(malignant_patients, n_c0_mal, replace=False).tolist() +
            np.random.choice(benign_patients, n_c0_ben, replace=False).tolist()
    )

    # Mise à jour des listes de patients restants
    remaining_mal = list(set(malignant_patients) - set(c0_patients))
    remaining_ben = list(set(benign_patients) - set(c0_patients))

    # Client 1 : "Centre de Dépistage de Routine"
    # -> Reçoit une énorme majorité de cas bénins
    # Prend 80% des bénins restants
    n_c1_ben = int(0.8 * len(remaining_ben))
    # Prend peu de malins (10% des restants)
    n_c1_mal = int(0.1 * len(remaining_mal))

    c1_patients = (
            np.random.choice(remaining_ben, n_c1_ben, replace=False).tolist() +
            np.random.choice(remaining_mal, n_c1_mal, replace=False).tolist()
    )

    # Client 2 : "Hôpital Généraliste"
    # -> Reçoit tout ce qui reste (mixte mais souvent déséquilibré)
    c2_patients = list(set(remaining_ben) - set(c1_patients)) + list(set(remaining_mal) - set(c1_patients))

    # 5. Construction des DataFrames finaux
    clients = {}
    clients['0'] = df[df['patient_id'].isin(c0_patients)].copy()
    clients['1'] = df[df['patient_id'].isin(c1_patients)].copy()
    clients['2'] = df[df['patient_id'].isin(c2_patients)].copy()

    # 6. Rapport de distribution (Pour vérifier le Non-IID)
    print("\n--- Distribution Non-IID ---")

    for cid, cdf in clients.items():
        total = len(cdf)
        n_mal = cdf['is_malignant'].sum()
        ratio = n_mal / total if total > 0 else 0

        if cid == '0':
            label = "Oncologie (Cible: Malin)"
        elif cid == '1':
            label = "Dépistage (Cible: Bénin)"
        else:
            label = "Généraliste (Reste)"

        print(f"Client {cid} [{label}]:")
        print(f"   🖼️ {total} images")
        print(f"   📈 {ratio * 100:.1f}% Malignantes ({n_mal} img)")

    return df, clients


if __name__ == "__main__":
    # Test rapide si lancé directement
    try:
        load_data_and_partition()
    except:
        pass