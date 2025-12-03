import os
import shutil
from pre_train import df_combined
from sklearn.model_selection import train_test_split

# Création des dossiers
BASE_DIR = "dataset_yolo"
for split in ["train", "val"]:
    for label in ["benign", "malignant"]:
        os.makedirs(os.path.join(BASE_DIR, split, label), exist_ok=True)

# Split des données
train_df, val_df = train_test_split(df_combined, test_size=0.2, random_state=42)


def move_files(df, split_name):
    print(f"Migration de {len(df)} images vers {split_name}...")
    for _, row in df.iterrows():
        src = row['image file path']
        label_str = "malignant" if row['target'] == 1 else "benign"

        # Nom de fichier unique pour éviter les doublons
        filename = f"{row['patient_id']}_{os.path.basename(src)}"
        dst = os.path.join(BASE_DIR, split_name, label_str, filename)

        try:
            shutil.copy(src, dst)
        except Exception as e:
            print(f"Erreur copie {src}: {e}")


move_files(train_df, "train")
move_files(val_df, "val")
print("Données prêtes pour YOLO !")