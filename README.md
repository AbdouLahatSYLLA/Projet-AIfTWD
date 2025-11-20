# Projet-AIfTWD
Implementation of a functional minimal federated setup on Cancer Identification using medical imaging data 

dataset BreaKHis_v1 need to be downloaded and added to the folder "Partie1/dataset/" next to the csv files

link to download the dataset : https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset?resource=download

first results :

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3568 entries, 0 to 3567
Data columns (total 17 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   patient_id               3568 non-null   object 
 1   breast_density           1696 non-null   float64
 2   left or right breast     3568 non-null   object 
 3   image view               3568 non-null   object 
 4   abnormality id           3568 non-null   int64  
 5   abnormality type         3568 non-null   object 
 6   mass shape               1692 non-null   object 
 7   mass margins             1636 non-null   object 
 8   assessment               3568 non-null   int64  
 9   pathology                3568 non-null   object 
 10  subtlety                 3568 non-null   int64  
 11  image file path          3568 non-null   object 
 12  cropped image file path  3568 non-null   object 
 13  ROI mask file path       3568 non-null   object 
 14  breast density           1872 non-null   float64
 15  calc type                1848 non-null   object 
 16  calc distribution        1433 non-null   object 
dtypes: float64(2), int64(3), object(12)
memory usage: 474.0+ KB
None
DataFrame Maître créé. Nombre total d'images: 3568
Distribution des étiquettes:
target
0    2111
1    1457
Name: count, dtype: int64
Patients Malignant Dominant: 710
Patients Benign Dominant: 856

--- Vérification de la Distribution Non-IID ---

Client: client1
Total images: 1622
Distribution Malignant (1): 61.22%

Client: client2
Total images: 1187
Distribution Malignant (1): 13.48%

Client: client3
Total images: 759
Distribution Malignant (1): 40.05%

Process finished with exit code 0
```
