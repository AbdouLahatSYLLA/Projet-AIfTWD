import torch
import torch.nn as nn
from opacus import PrivacyEngine
from tqdm.auto import tqdm


class Trainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader, test_loader, epochs, lr, mode='standard', mu=0.0, dp_settings=None):
        # Optimiseur
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        # --- AJOUT SCHEDULER ---
        # Divise le LR par 10 toutes les 10 époques pour stabiliser la fin d'entraînement
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        # --- Setup Privacy (Si activé) ---
        privacy_engine = None
        if mode == 'dp' and dp_settings:
            privacy_engine = PrivacyEngine()
            # Opacus ne supporte pas toujours bien les schedulers, on fait attention
            self.model, optimizer, train_loader = privacy_engine.make_private(
                module=self.model, optimizer=optimizer, data_loader=train_loader,
                noise_multiplier=dp_settings.get('noise', 1.0),
                max_grad_norm=dp_settings.get('clip', 1.2),
            )

        # --- Setup FedProx ---
        global_params = None
        if mode == 'fedprox' and mu > 0:
            global_params = [p.clone().detach() for p in self.model.parameters()]

        epoch_loss = 0.0

        for epoch in range(epochs):
            self.model.train()
            batch_losses = []
            disable_tqdm = (epochs == 1)

            # On affiche le LR courant dans la barre
            current_lr = optimizer.param_groups[0]['lr']
            desc = f"Ep {epoch + 1} [LR={current_lr:.1e}]"

            pbar = tqdm(train_loader, desc=desc, unit="batch", disable=disable_tqdm)

            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # FedProx Term
                if mode == 'fedprox' and mu > 0:
                    prox_term = 0.0
                    for w, w_t in zip(self.model.parameters(), global_params):
                        prox_term += (w - w_t).norm(2) ** 2
                    loss += (mu / 2) * prox_term

                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
                if not disable_tqdm:
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            if mode == 'standard':
                avg_loss, acc = self.evaluate(test_loader)
                print(f"Loss : {avg_loss:.4f} | Accuracy : {acc*100:.2f}%")

            # Mise à jour du LR à la fin de l'époque (sauf en mode DP où c'est parfois géré autrement)
            if mode != 'dp':
                scheduler.step()

            if batch_losses:
                epoch_loss = sum(batch_losses) / len(batch_losses)

        results = {"loss": epoch_loss}
        if privacy_engine:
            results["epsilon"] = privacy_engine.get_epsilon(delta=1e-5)
        torch.cuda.empty_cache()
        return results

    def evaluate(self, val_loader):
        self.model.eval()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total if total > 0 else 0.0
        avg_loss = loss / len(val_loader) if len(val_loader) > 0 else 0.0
        return avg_loss, acc