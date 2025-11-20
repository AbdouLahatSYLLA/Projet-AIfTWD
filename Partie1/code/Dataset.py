import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Define standard transformations (Resize and Normalize)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CBISDataset(Dataset):
    def __init__(self, df, transform=None):
        # Initialize dataset with dataframe and transforms
        self.df = df
        self.transform = transform

    def __len__(self):
        # Return total number of samples
        return len(self.df)

    def __getitem__(self, idx):
        # Get image path and label from dataframe
        img_path = self.df.iloc[idx]['image file path']
        label = self.df.iloc[idx]['target']

        # Open image and convert to RGB
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return black image on error
            image = Image.new('RGB', (224, 224))

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)