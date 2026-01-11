import pandas as pd
import os
import shutil
from PIL import Image
from tqdm.auto import tqdm
import sys

# --- CONFIGURATION ---
TARGET_SIZE = (224, 224)

# Chemins relatifs robustes (fonctionne depuis n'importe où)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "dataset")

CSV_DIR = os.path.join(BASE_DIR, "csv")
SOURCE_IMAGES_DIR = os.path.join(BASE_DIR, "images")

# Dossier de sortie spécifique pour ne pas mélanger
OUTPUT_IMAGES_DIR = os.path.join(BASE_DIR, "processed_images_cropped")
OUTPUT_CSV = os.path.join(BASE_DIR, "cleaned_dataset.csv")


def prepare():
    # Vérifications
    if not os.path.exists(CSV_DIR):
        print(f"❌ Dossier CSV introuvable : {CSV_DIR}")
        return
    if not os.path.exists(SOURCE_IMAGES_DIR):
        print(f"❌ Dossier Images introuvable : {SOURCE_IMAGES_DIR}")
        return

    # Nettoyage dossier sortie
    if os.path.exists(OUTPUT_IMAGES_DIR):
        shutil.rmtree(OUTPUT_IMAGES_DIR)
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

    print(f"📂 Lecture des données (Mode CROPPED IMAGES)...")

    # --- CHARGEMENT CSV ---
    try:
        dfs = []
        files_map = {
            'train': ['mass_case_description_train_set.csv', 'calc_case_description_train_set.csv'],
            'test': ['mass_case_description_test_set.csv', 'calc_case_description_test_set.csv']
        }
        found_any = False
        for split_type, filenames in files_map.items():
            for f in filenames:
                p = os.path.join(CSV_DIR, f)
                if os.path.exists(p):
                    print(f"   -> Lecture de {f}")
                    df = pd.read_csv(p)
                    df['abnormality_type'] = 'Mass' if 'mass' in f else 'Calc'
                    df['split'] = split_type
                    dfs.append(df)
                    found_any = True

        if not found_any:
            print("❌ Aucun CSV trouvé.")
            return
        df_full = pd.concat(dfs, ignore_index=True)
        print(f"📊 Total entrées brutes : {len(df_full)}")

    except Exception as e:
        print(f"❌ Erreur lecture CSV: {e}")
        return

    # --- INDEXATION FICHIERS ---
    print("🔍 Indexation des images sources...")
    image_map = {}
    count_found = 0
    for root, _, files in os.walk(SOURCE_IMAGES_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.dcm', '.png')):
                folder_uid = os.path.basename(root)
                image_map[folder_uid] = os.path.join(root, file)
                count_found += 1

    print(f"✅ {count_found} images physiques trouvées.")

    # --- TRAITEMENT ---
    valid_rows = []
    print("🚀 Extraction et redimensionnement des CROPS...")

    matches = 0
    failures = 0

    for idx, row in tqdm(df_full.iterrows(), total=len(df_full)):

        # --- CHANGEMENT CLÉ ICI : On prend la colonne CROPPED ---
        raw_path = str(row['cropped image file path'])

        # Nettoyage du chemin
        raw_path_clean = raw_path.strip().replace('\n', '').replace('\r', '')
        parts = [p.strip() for p in raw_path_clean.split('/')]

        real_path = None
        for part in parts:
            if part in image_map:
                real_path = image_map[part]
                break

        if real_path:
            try:
                with Image.open(real_path) as img:
                    img = img.convert('RGB')
                    # On resize les crops car ils ont des tailles très variables (petits ou gros)
                    img = img.resize(TARGET_SIZE)

                    safe_view = str(row['image view']).strip().replace(' ', '_')
                    safe_side = str(row['left or right breast']).strip()
                    pat_id = str(row['patient_id']).strip()

                    # On ajoute "CROP" dans le nom pour être sûr
                    new_filename = f"CROP_{pat_id}_{safe_side}_{safe_view}_{idx}.jpg"
                    save_path = os.path.join(OUTPUT_IMAGES_DIR, new_filename)

                    img.save(save_path, quality=90)

                    row['local_path'] = save_path
                    valid_rows.append(row)
                    matches += 1
            except Exception:
                pass
        else:
            failures += 1

    print(f"\n📊 Bilan traitement Crops :")
    print(f"   ✅ Crops récupérés : {matches}")
    print(f"   ❌ Crops introuvables : {failures}")

    # --- CSV FINAL ---
    if not valid_rows:
        print("❌ Aucune image générée.")
        return

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

    print(f"\n✅ Dataset CROPPED prêt !")
    print(f"   📄 {OUTPUT_CSV}")
    print(f"   📂 {OUTPUT_IMAGES_DIR}")


if __name__ == "__main__":
    prepare()