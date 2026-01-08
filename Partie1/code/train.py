import torch.nn as nn
import torch


def train(net, trainloader, epochs=1, device='cpu'):
    """Train the network on the training set."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    # Set model to training mode
    net.train()
    net.to(device)

    # List to store history for saving
    history = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            outputs = net(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate simple accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total

        print(f"Epoch {epoch + 1}: Loss {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

        # Add current epoch stats
        history.append({"epoch": epoch + 1, "loss": epoch_loss, "accuracy": epoch_acc})

    return history


def test(net, testloader, device='cpu'):
    """Evaluate the network on the test set."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0

    # Set model to evaluation mode
    net.eval()
    net.to(device)

    # No gradient calculation needed
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return loss / len(testloader), accuracy