# import libraries
import pandas as pd
import os
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class CBISDDSMDataPrep:
    DATASET_DIR = ""
    JPEG_DIR = ""

    def __init__(
            self,
            dataset_dir,
            n_clients=3,
            non_iid=True,
            class_ratio_per_client=None,
            client_ratio=None,
            val_ratio=0.15,
            seed=42
        ) :
        self.DATASET_DIR = os.path.join(CURRENT_DIR, dataset_dir)
        self.JPEG_DIR = os.path.join(self.DATASET_DIR, 'images')

        self.appelation = {
            'benign': 0,
            'malignant': 1
        }

        self.n_clients = n_clients
        self.non_iid = non_iid
        self.class_ratio_per_client = class_ratio_per_client
        self.client_ratio = client_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        self.load_and_merge()
        self.index_real_images()
        self.extract_train()
        self.extract_test()

    def load_and_merge(self):
        self.train_csvs = [
            'mass_case_description_train_set.csv',
            'calc_case_description_train_set.csv'
        ]
        self.test_csvs = [
            'mass_case_description_test_set.csv',
            'calc_case_description_test_set.csv'
        ]
    
    def index_real_images(self):
        self.real_image_paths = {}
        for root, _, files in os.walk(self.JPEG_DIR):
            for file in files:
                if file.endswith(".jpg"):
                    folder_uid = os.path.basename(root)
                    self.real_image_paths[folder_uid] = os.path.join(root, file)

    def get_real_path(self, csv_path):
        parts = csv_path.split('/')
        for part in parts:
            if part in self.real_image_paths:
                return self.real_image_paths[part]
        return None

    def extract_train(self):
        self.df_mass_train = pd.read_csv(os.path.join(self.DATASET_DIR, self.train_csvs[0]))
        self.df_calc_train = pd.read_csv(os.path.join(self.DATASET_DIR, self.train_csvs[1]))

        self.df_combined_train = pd.concat([self.df_mass_train, self.df_calc_train], ignore_index=True)
        self.df_combined_train['real_path'] = self.df_combined_train['image file path'].apply(self.get_real_path)
        self.df_combined_train = self.df_combined_train.dropna(subset=['real_path'])
        self.df_combined_train['local_path'] = self.df_combined_train['real_path']

        self.df_combined_train['target'] = self.df_combined_train['pathology'].apply(
            lambda x: 1 if 'MALIGNANT' in str(x).upper() else 0
        )

        self.df_combined_train = self.df_combined_train[['patient_id', 'local_path', 'target']]

    def extract_test(self):
        self.df_mass_test = pd.read_csv(os.path.join(self.DATASET_DIR, self.test_csvs[0]))
        self.df_calc_test = pd.read_csv(os.path.join(self.DATASET_DIR, self.test_csvs[1]))

        self.df_combined_test = pd.concat([self.df_mass_test, self.df_calc_test], ignore_index=True)
        self.df_combined_test['real_path'] = self.df_combined_test['image file path'].apply(self.get_real_path)
        self.df_combined_test = self.df_combined_test.dropna(subset=['real_path'])
        self.df_combined_test['local_path'] = self.df_combined_test['real_path']

        self.df_combined_test['target'] = self.df_combined_test['pathology'].apply(
            lambda x: 1 if 'MALIGNANT' in str(x).upper() else 0
        )

        self.df_combined_test = self.df_combined_test[['patient_id', 'local_path', 'target']]

    def split_by_clients_with_ratio(self, mode="balanced", client_type="train"):
        """
        Split le dataset (train uniquement) en plusieurs clients selon des ratios de classes.
        Split patient-wise (pas de leakage).

        mode:
            - "balanced" : tailles de clients ~ égales, ratios approximatifs
            - "strict"   : ratios exacts, tailles de clients variables
        """

        assert self.n_clients == len(self.class_ratio_per_client), \
            "Le nombre de clients doit correspondre au nombre de ratios fournis."

        for r in self.class_ratio_per_client:
            assert len(r) == 2, "Chaque ratio doit contenir 2 valeurs (2 classes)."
            assert abs(sum(r) - 1.0) < 1e-6, "Les ratios doivent sommer à 1."

        # -----------------------------
        # Classe dominante par patient
        # -----------------------------
        if client_type == "train":
            patient_stats = (
                self.df_combined_train
                .groupby('patient_id')['target']
                .mean()
                .reset_index()
            )
        elif client_type == "test":
            patient_stats = (
                self.df_combined_test
                .groupby('patient_id')['target']
                .mean()
                .reset_index()
            )
        else:
            raise ValueError("client_type must be 'train' or 'test'")

        patient_stats['dominant_class'] = patient_stats['target'].round().astype(int)

        class0_patients = patient_stats[
            patient_stats['dominant_class'] == 0
        ]['patient_id'].tolist()

        class1_patients = patient_stats[
            patient_stats['dominant_class'] == 1
        ]['patient_id'].tolist()


        np.random.seed(self.seed)
        np.random.shuffle(class0_patients)
        np.random.shuffle(class1_patients)

        client_patients = {i: [] for i in range(self.n_clients)}

        # =============================
        # MODE BALANCED (actuel)
        # =============================
        if mode == "balanced":
            total_patients = len(patient_stats)
            if self.client_ratio is None :
                patients_per_client = [total_patients // self.n_clients] * self.n_clients
            else :
                patients_per_client = [
                    int(total_patients * r) for r in self.client_ratio
                ]

            for client_id in range(self.n_clients):
                r0, r1 = self.class_ratio_per_client[client_id]

                n0 = int(patients_per_client[client_id] * r0)
                n1 = int(patients_per_client[client_id] * r1)

                selected_0 = class0_patients[:n0]
                selected_1 = class1_patients[:n1]

                class0_patients = class0_patients[n0:]
                class1_patients = class1_patients[n1:]

                client_patients[client_id].extend(selected_0 + selected_1)

            # Patients restants
            remaining_patients = class0_patients + class1_patients
            np.random.shuffle(remaining_patients)

            for i, pid in enumerate(remaining_patients):
                client_patients[i % self.n_clients].append(pid)

        # =============================
        # MODE STRICT (ratios exacts)
        # =============================
        elif mode == "strict":
            if self.client_ratio is None :
                max_size = [min(
                    len(class0_patients) // self.n_clients,
                    len(class1_patients) // self.n_clients
                ) * 2] * self.n_clients
            else :
                max_size = min(
                    len(class0_patients) // self.n_clients,
                    len(class1_patients) // self.n_clients
                ) * 2
                total_client = max_size * self.n_clients
                max_size = [int(total_client * r) for r in self.client_ratio]

            for client_id in range(self.n_clients):
                r0, r1 = self.class_ratio_per_client[client_id]

                n0 = int(max_size[client_id] * r0)
                n1 = int(max_size[client_id] * r1)

                selected_0 = class0_patients[:n0]
                selected_1 = class1_patients[:n1]

                class0_patients = class0_patients[n0:]
                class1_patients = class1_patients[n1:]

                client_patients[client_id].extend(selected_0 + selected_1)

        else:
            raise ValueError("mode must be 'balanced' or 'strict'")

        # -----------------------------
        # Création des DataFrames
        # -----------------------------
        client_dfs = {}
        if client_type == "train":
            for client_id, patients in client_patients.items():
                client_dfs[f"client_{client_id}"] = (
                    self.df_combined_train[
                        self.df_combined_train['patient_id'].isin(patients)
                    ]
                    .reset_index(drop=True)
                )
        elif client_type == "test":
            for client_id, patients in client_patients.items():
                client_dfs[f"client_{client_id}"] = (
                    self.df_combined_test[
                        self.df_combined_test['patient_id'].isin(patients)
                    ]
                    .reset_index(drop=True)
                )

        return client_dfs
    
    def manage_ratios(self, ratios):
        pass

    def print_dataset_distribution(self, client_dfs):
        for client_name, df in client_dfs.items():
            total = len(df)
            if total == 0:
                print(f"{client_name}: No data available.")
                continue
            count_class0 = len(df[df['target'] == 0])
            count_class1 = len(df[df['target'] == 1])
            ratio_class0 = count_class0 / total * 100
            ratio_class1 = count_class1 / total * 100
            print(f"{client_name}: Total={total}, Class 0={count_class0} ({ratio_class0:.2f}%), Class 1={count_class1} ({ratio_class1:.2f}%)\n")

    def print_images_samples(self, 
                            samples : list[str] = ['benign', 'malignant'], 
                            n_samples : list[int] =[1, 1]
                        ):
        import matplotlib.pyplot as plt

        if len(samples) != len(n_samples):
            print("Error: 'samples' and 'n_samples' must have the same length.")
            return
        
        images = self.take_one_image_by_class()
        plt.figure(figsize=(10, 5))
        for i, class_name in enumerate(samples):
            if class_name in images:
                img_path = images[class_name]
                img = plt.imread(img_path)
                plt.subplot(1, len(samples), i + 1)
                plt.imshow(img, cmap='gray')
                plt.title(f"{class_name.capitalize()}")
                plt.axis('off')
        plt.show()

    def print_patient_count(self):
        n_patients = self.df_combined_train['patient_id'].nunique()
        n_patients_benign = len(self.df_combined_train[self.df_combined_train['target'] == 0]['patient_id'].unique())
        n_patients_malignant = len(self.df_combined_train[self.df_combined_train['target'] == 1]['patient_id'].unique())


        print(f"Nombre total de patients (train): {n_patients}\n")
        print(f" - Benign: {n_patients_benign}\n")
        print(f" - Malignant: {n_patients_malignant}\n")

    def print_patient_by_client_count(self, client_dfs):
        print("Nombre de patients par client:\n")
        for client_name, df in client_dfs.items():
            n_patients = df['patient_id'].nunique()
            print(f"{client_name}: {n_patients} patients")

    def get_dataset_centralized(self):
        return self.df_combined_train, self.df_combined_test
        
    def take_one_image_by_class(self):
        sample_images = {}
        for class_name in self.appelation.keys():
            class_value = self.appelation[class_name]
            df_class = self.df_combined_train[self.df_combined_train['target'] == class_value]
            if not df_class.empty:
                sample_images[class_name] = df_class.iloc[0]['image file path']
        return sample_images

    def __str__(self):
        M, B = 0, 0
        for target in self.df_combined_train['target'] :
            if target==1: M+=1
            else: B+=1
        string = f"Train Dataset: Total={len(self.df_combined_train)} (Malignant={M} ({M/len(self.df_combined_train)*100:.2f}%), Benign={B} ({B/len(self.df_combined_train)*100:.2f}%))"
        M, B = 0, 0
        for target in self.df_combined_test['target'] :
            if target==1: M+=1
            else: B+=1
        string += f"\nTest Dataset: Total={len(self.df_combined_test)} (Malignant={M} ({M/len(self.df_combined_test)*100:.2f}%), Benign={B} ({B/len(self.df_combined_test)*100:.2f}%))"
        return string

# Print image samples
