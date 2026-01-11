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

# Chemins Kaggle Input ou local possibles
POSSIBLE_ROOTS = [
    '/kaggle/input/cbis-ddsm-breast-cancer-image-dataset',
    '/kaggle/input/cbis-ddsm',
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
        # On définit explicitement quels fichiers sont TRAIN et lesquels sont TEST
        files_map = {
            'train': ['mass_case_description_train_set.csv', 'calc_case_description_train_set.csv'],
            'test': ['mass_case_description_test_set.csv', 'calc_case_description_test_set.csv']
        }

        found_any = False
        for split_type, filenames in files_map.items():
            for f in filenames:
                p = os.path.join(csv_dir, f)
                if os.path.exists(p):
                    df = pd.read_csv(p)
                    # --- CORRECTION ICI : ON TAGUE L'ORIGINE ---
                    df['abnormality_type'] = 'Mass' if 'mass' in f else 'Calc'
                    df['split'] = split_type  # 'train' ou 'test'
                    dfs.append(df)
                    found_any = True

        if not found_any:
            print("❌ Aucun CSV trouvé !")
            return

        df_full = pd.concat(dfs, ignore_index=True)
        print(f"📊 Total entrées brutes : {len(df_full)}")
        print(f"   - Train set : {len(df_full[df_full['split'] == 'train'])}")
        print(f"   - Test set  : {len(df_full[df_full['split'] == 'test'])}")

    except Exception as e:
        print(f"❌ Erreur lecture CSV: {e}")
        return

    # --- INDEXATION ROBUSTE ---
    print("🔍 Indexation des fichiers JPEG sources...")
    image_map = {}
    count_found = 0

    search_dir = os.path.join(root_dir, 'jpeg')
    if not os.path.exists(search_dir):
        print("⚠️ Dossier 'jpeg' introuvable, recherche recursive à la racine...")
        search_dir = root_dir

    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                folder_uid = os.path.basename(root)
                image_map[folder_uid] = os.path.join(root, file)
                count_found += 1

    print(f"✅ {count_found} images physiques indexées.")

    # --- TRAITEMENT ---
    valid_rows = []
    print("🚀 Traitement et redimensionnement...")

    matches = 0
    failures = 0

    for idx, row in tqdm(df_full.iterrows(), total=len(df_full)):

        raw_path = str(row['image file path'])
        raw_path_clean = raw_path.strip().replace('\n', '').replace('\r', '')

        real_path = None
        parts = [p.strip() for p in raw_path_clean.split('/')]

        for part in parts:
            if part in image_map:
                real_path = image_map[part]
                break

        if real_path:
            try:
                with Image.open(real_path) as img:
                    img = img.convert('RGB')
                    img = img.resize(TARGET_SIZE)

                    safe_view = str(row['image view']).strip().replace(' ', '_')
                    safe_side = str(row['left or right breast']).strip()
                    pat_id = str(row['patient_id']).strip()

                    new_filename = f"{pat_id}_{safe_side}_{safe_view}_{idx}.jpg"
                    save_path = os.path.join(OUTPUT_IMAGES_DIR, new_filename)

                    img.save(save_path, quality=85)

                    row['local_path'] = save_path
                    valid_rows.append(row)
                    matches += 1
            except Exception as e:
                pass
        else:
            failures += 1

    print(f"\n📊 Bilan traitement :")
    print(f"   ✅ Images matchées et traitées : {matches}")
    print(f"   ❌ Images non trouvées : {failures}")

    if not valid_rows:
        print("❌ CRITIQUE : Aucune image traitée.")
        return

    # --- GÉNÉRATION CSV FINAL ---
    df_clean = pd.DataFrame(valid_rows)

    def get_label(row):
        pathology = str(row['pathology']).upper()
        is_mass = row['abnormality_type'] == 'Mass'
        is_malignant = 'MALIGNANT' in pathology

        if not is_mass and not is_malignant: return 0
        if not is_mass and is_malignant: return 1
        if is_mass and not is_malignant: return 2
        if is_mass and is_malignant: return 3
        return 0

    df_clean['target'] = df_clean.apply(get_label, axis=1)
    df_clean.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Dataset prêt !")
    print(f"   📄 {OUTPUT_CSV}")
    print(f"   👉 Colonne 'split' ajoutée : {df_clean['split'].unique()}")


if __name__ == "__main__":
    prepare()