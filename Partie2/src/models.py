import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import BasicBlock, Bottleneck
from opacus.validators import ModuleValidator

def safe_basic_block_forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out + identity
    out = self.relu(out)
    return out


def safe_bottleneck_forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv3(out)
    out = self.bn3(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out + identity
    out = self.relu(out)
    return out


# On applique le patch immédiatement
BasicBlock.forward = safe_basic_block_forward
Bottleneck.forward = safe_bottleneck_forward


def get_model(model_name='resnet18', num_classes=2, use_dp=False, device='cpu'):
    # 1. Chargement de l'architecture (qui utilise maintenant nos blocs patchés)
    if model_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1')
    elif model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
    elif model_name == 'resnext50':
        model = models.resnext50_32x4d(weights='IMAGENET1K_V1')
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    # Adaptation de la dernière couche
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Désactivation des ReLU inplace (Double sécurité)
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    # Adaptation Opacus si nécessaire
    if use_dp:
        model = ModuleValidator.fix(model)
        # On revérifie les ReLU après le passage du Validator
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    return model.to(device)