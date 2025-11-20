from torchvision import models
import torch.nn as nn

def get_model():
    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Replace last layer (Fully Connected)
    # Original input features
    num_ftrs = model.fc.in_features
    # Output = 2 classes (Benign, Malignant)
    model.fc = nn.Linear(num_ftrs, 2)

    return model