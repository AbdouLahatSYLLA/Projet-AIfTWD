
# import libraries

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




#-------------------------------------------------------------------------------------------------------#

# Data Cleaning and Preprocessing

#Load all datasets csv
df_mass_test = pd.read_csv('dataset/mass_case_description_test_set.csv')
df_mass_train = pd.read_csv('dataset/mass_case_description_train_set.csv')
df_calc_train = pd.read_csv('dataset/calc_case_description_train_set.csv')
df_calc_test = pd.read_csv('dataset/calc_case_description_test_set.csv')

# Merge the four DataFrames
df_combined = pd.concat([df_mass_test, df_mass_train,df_calc_train,df_calc_test], ignore_index=True)

# Display the first few rows of the combined DataFrame and its information
df_combined.head()
print(df_combined.info())

# use binary label for the type of disease
df_combined['target'] = df_combined['pathology'].apply(
    lambda x: 1 if 'MALIGNANT' in str(x).upper() else 0
)


df_combined = df_combined[['patient_id', 'image file path', 'cropped image file path', 'target', 'abnormality id']]

#create a copy
df_combined_copy = df_combined.copy()

print(f"DataFrame Maître créé. Nombre total d'images: {len(df_combined)}")
print(f"Distribution des étiquettes:\n{df_combined['target'].value_counts()}")

#Dataframe partition using "patient id"

# identify dominant pathology before spliting into groups of patients
patients = df_combined.groupby('patient_id')['target'].agg(['mean', 'count']).reset_index()

patients['is_malignant_dominant'] = patients['mean'] > 0.5

malignant_patients = patients[patients['is_malignant_dominant']]['patient_id'].tolist()
benign_patients = patients[~patients['is_malignant_dominant']]['patient_id'].tolist()

print(f"Patients Malignant Dominant: {len(malignant_patients)}")
print(f"Patients Benign Dominant: {len(benign_patients)}")

#NON-IID repartition
SEED_VALUE = 42
np.random.seed(SEED_VALUE)

#Client-1 : Majority Malignant (70% vs 30%)
N1_M = int(0.7 * len(malignant_patients))
N1_B = int(0.3 * len(benign_patients))
client1_M = np.random.choice(malignant_patients, N1_M, replace=False)
client1_B = np.random.choice(benign_patients, N1_B, replace=False)
client1_patients = np.concatenate([client1_M, client1_B])

# Update remaining lists
malignant_patients_rest = [s for s in malignant_patients if s not in client1_M]
benign_patients_rest = [s for s in benign_patients if s not in client1_B]


#Client-2 : Majority Benign (70% vs 30%)
N2_M = int(0.3 * len(malignant_patients_rest))
N2_B = int(0.7 * len(benign_patients_rest))
client2_M = np.random.choice(malignant_patients_rest, N2_M, replace=False)
client2_B = np.random.choice(benign_patients_rest, N2_B, replace=False)
client2_patients = np.concatenate([client2_M, client2_B])

# 3. Client 3 (Le reste)
#Client-3 : rest of the patients
all_patients = patients['patient_id'].tolist()
used_patients = np.concatenate([client1_patients, client2_patients])
client3_patients = [s for s in all_patients if s not in used_patients]

# Final dataframe
client_dataframes = {}
all_client_patients = {'client1': client1_patients, 'client2': client2_patients, 'client3': client3_patients}

print("\n--- Vérification de la Distribution Non-IID ---")
for name, patients_list in all_client_patients.items():
    df = df_combined[df_combined['patient_id'].isin(patients_list)]
    client_dataframes[name] = df

    malin_count = df['target'].sum()
    total_count = len(df)

    print(f"\nClient: {name}")
    print(f"Total images: {total_count}")
    print(f"Distribution Malignant (1): {malin_count / total_count * 100:.2f}%")

