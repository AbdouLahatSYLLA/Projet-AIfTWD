import pandas as pd
import os
import shutil
from PIL import Image
from tqdm.auto import tqdm
import sys

# --- CONFIGURATION ---
TARGET_SIZE = (224, 224)
OUTPUT_DIR = "dataset"
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "cleaned_dataset.csv")

# Chemins Kaggle Input possibles (à adapter selon l'environnement)
POSSIBLE_ROOTS = [
    'dataset',
    '.'
]


def find_root():
    for path in POSSIBLE_ROOTS:
        if os.path.exists(os.path.join(path, 'csv')):
            return path
    return None


def prepare():
    root_dir = find_root()
    if not root_dir:
        print("❌ Dataset input introuvable.")
        return

    print(f"📂 Lecture des données depuis {root_dir}...")

    # Chargement CSVs
    csv_dir = os.path.join(root_dir, 'csv')

    try:
        dfs = []
        # On charge les descriptions de cas (Mass et Calc)
        for f in ['mass_case_description_train_set.csv', 'mass_case_description_test_set.csv',
                  'calc_case_description_train_set.csv', 'calc_case_description_test_set.csv']:
            p = os.path.join(csv_dir, f)
            if os.path.exists(p):
                df = pd.read_csv(p)
                # Ajout du type d'anomalie
                df['abnormality_type'] = 'Mass' if 'mass' in f else 'Calc'
                dfs.append(df)

        if not dfs:
            print("❌ Aucun CSV trouvé !")
            return

        df_full = pd.concat(dfs, ignore_index=True)
        print(f"📊 Total entrées brutes : {len(df_full)}")
    except Exception as e:
        print(f"❌ Erreur lecture CSV: {e}")
        return

    # --- INDEXATION DES IMAGES SOURCES ---
    print("🔍 Indexation des fichiers JPEG sources...")
    image_map = {}
    count_found = 0

    # On cherche dans le dossier 'images'
    search_dir = os.path.join(root_dir, 'images')
    if not os.path.exists(search_dir):
        print(f"⚠️ Dossier 'images' introuvable dans {root_dir}")
        return

    # On indexe chaque fichier par le nom de son dossier parent (UID unique dans DDSM)
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.images')):
                folder_uid = os.path.basename(root)
                image_map[folder_uid] = os.path.join(root, file)
                count_found += 1

    print(f"✅ {count_found} images physiques indexées.")

    # --- TRAITEMENT ---
    valid_rows = []

    print("🚀 Traitement et redimensionnement des images (FULL MAMMOGRAMS)...")

    for idx, row in tqdm(df_full.iterrows(), total=len(df_full)):

        # --- CHANGEMENT V2 : ON FORCE L'IMAGE ENTIÈRE ---
        # On ignore 'cropped image file path' et 'ROI mask file path'
        raw_path = str(row['image file path'])

        # Le chemin dans le CSV ressemble à :
        # "Mass-Training_P_00001_LEFT_CC/1.3.6.1.4.1.9590.../1.3.6.1.4.1.9590.../000000.dcm"
        # On doit trouver quel segment correspond à un dossier physique

        real_path = None
        parts = raw_path.split('/')

        # On cherche l'UID dans les parties du chemin
        for part in parts:
            if part in image_map:
                real_path = image_map[part]
                break

        # Si on ne trouve pas directement, on essaie de nettoyer les caractères cachés (souvent \n dans les CSV)
        if not real_path:
            for part in parts:
                clean_part = part.strip()
                if clean_part in image_map:
                    real_path = image_map[clean_part]
                    break

        if real_path:
            try:
                # Ouverture et Resize
                with Image.open(real_path) as img:
                    img = img.convert('RGB')
                    img = img.resize(TARGET_SIZE)

                    # Nouveau nom unique
                    safe_view = str(row['image view']).strip().replace(' ', '_')
                    safe_side = str(row['left or right breast']).strip()
                    pat_id = str(row['patient_id']).strip()

                    new_filename = f"{pat_id}_{safe_side}_{safe_view}_{idx}.jpg"
                    save_path = os.path.join(OUTPUT_IMAGES_DIR, new_filename)

                    img.save(save_path, quality=85)

                    # Mise à jour des infos pour le nouveau CSV
                    row['local_path'] = save_path
                    valid_rows.append(row)
            except Exception as e:
                # print(f"Erreur image {real_path}: {e}")
                pass

    if not valid_rows:
        print("❌ CRITIQUE : Aucune image traitée. Vérifiez les chemins.")
        return

    # --- GÉNÉRATION DU CSV FINAL ---
    df_clean = pd.DataFrame(valid_rows)

    # Création de la cible (Target) 0-3
    def get_label(row):
        # Pathology: BENIGN, MALIGNANT, BENIGN_WITHOUT_CALLBACK
        pathology = str(row['pathology']).upper()
        is_mass = row['abnormality_type'] == 'Mass'

        is_malignant = 'MALIGNANT' in pathology

        if not is_mass and not is_malignant: return 0  # Calc Benign
        if not is_mass and is_malignant: return 1  # Calc Malignant
        if is_mass and not is_malignant: return 2  # Mass Benign
        if is_mass and is_malignant: return 3  # Mass Malignant
        return 0

    df_clean['target'] = df_clean.apply(get_label, axis=1)

    # Sauvegarde
    df_clean.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Dataset généré avec succès !")
    print(f"   📂 Images : {OUTPUT_IMAGES_DIR}")
    print(f"   📄 CSV : {OUTPUT_CSV}")
    print(f"   🖼️ Nombre d'images : {len(df_clean)}")
    print(f"   📊 Distribution des classes :\n{df_clean['target'].value_counts().sort_index()}")


if __name__ == "__main__":
    prepare()