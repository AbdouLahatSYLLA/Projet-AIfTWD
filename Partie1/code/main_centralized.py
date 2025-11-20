import torch
from torch.utils.data import DataLoader, random_split
import sys
import os

# Correction of paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

# Import prepared data
try:
    from pre_train import df_combined

    print("Data imported successfully.")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

from Dataset import CBISDataset, data_transforms
from ResNet18 import get_model
from train import train, test


def run_centralized_baseline():
    print("--- Starting Centralized Baseline ---")

    # Configure Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Training on: {device}")

    # Prepare full dataset
    full_dataset = CBISDataset(df_combined, transform=data_transforms)

    # Split Train/Test (80/20)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_set, test_set = random_split(full_dataset, [train_size, test_size],
                                       generator=torch.Generator().manual_seed(42))

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

    # Load Model
    model = get_model()

    # Start Training
    print(f"Training on {len(train_set)} images. Testing on {len(test_set)} images.")
    train(model, train_loader, epochs=5, device=device)

    # Final Evaluation
    print("Evaluating final model...")
    loss, accuracy = test(model, test_loader, device=device)
    print(f"CENTRALIZED BASELINE RESULT -> Accuracy: {accuracy * 100:.2f}%")

    # Save model
    torch.save(model.state_dict(), "baseline_centralized.pth")
    print("Model saved as 'baseline_centralized.pth'")


if __name__ == "__main__":
    run_centralized_baseline()