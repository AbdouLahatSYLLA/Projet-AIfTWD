import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from opacus import PrivacyEngine


class Trainer:
    def __init__(self, model, device, dp_settings=None):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        # Optimiseur (SGD est souvent préféré pour DP, Adam pour la perf standard)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.dp_settings = dp_settings

    def train(self, train_loader, epochs, lr, mode='standard', mu=0.0, global_params=None):
        """
        Entraîne le modèle et retourne l'historique (loss, accuracy).
        """
        # Mise à jour du Learning Rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Gestion Differential Privacy
        if mode == 'dp' and self.dp_settings:
            privacy_engine = PrivacyEngine()
            self.model, self.optimizer, train_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=train_loader,
                noise_multiplier=self.dp_settings['noise'],
                max_grad_norm=self.dp_settings['clip'],
            )

        # Historique pour les courbes
        history = {'loss': [], 'accuracy': []}

        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            # Barre de progression si centralisé (plusieurs epochs), sinon silencieux pour fédéré
            iterator = train_loader
            if epochs > 1:
                iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for images, labels in iterator:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Terme Proximal (FedProx)
                if mode == 'fedprox' and global_params is not None:
                    prox_term = 0.0
                    for param, global_param in zip(self.model.parameters(), global_params):
                        prox_term += (param - torch.as_tensor(global_param).to(self.device)).norm(2)
                    loss += (mu / 2) * prox_term

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total

            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)

            if epochs > 1:
                print(f"   Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        # Si on est en DP, on 'nettoie' le modèle pour qu'il redevienne compatible PyTorch standard
        if mode == 'dp':
            self.model = self.model._module

        # En mode Fédéré (1 epoch), on retourne la dernière valeur
        if epochs == 1:
            return history['loss'][-1], history['accuracy'][-1]

        # En mode Centralisé, on retourne tout l'historique
        return history

    def evaluate(self, test_loader):
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return loss / len(test_loader), correct / total