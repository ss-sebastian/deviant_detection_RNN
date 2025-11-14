import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from main import GRUDeviantDetector  # Import the model from main.py

# Hyperparameters
batch_size = 64
epochs = 10
learning_rate = 0.000001 # test diff. lr

# Load the tensors
input_tensor = torch.load("/users/seb/Desktop/bcbl/msc_thesis/input_tensor.pt")
labels_tensor = torch.load("/users/seb/Desktop/bcbl/msc_thesis/labels_tensor.pt")

labels_tensor = torch.where(labels_tensor == 4, torch.tensor(0), labels_tensor)
labels_tensor = torch.where(labels_tensor == 5, torch.tensor(1), labels_tensor)
labels_tensor = torch.where(labels_tensor == 6, torch.tensor(2), labels_tensor)

# Ensure the tensors are on the correct device (MPS or CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
input_tensor = input_tensor.to(device).float()
labels_tensor = labels_tensor.to(device).float()

# Convert input_tensor and labels_tensor into a TensorDataset
dataset = TensorDataset(input_tensor, labels_tensor)

# Create DataLoader for batch processing
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
model = GRUDeviantDetector(input_size=1, hidden_size=64, num_layers=2, num_classes=3).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Optionally: Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode

    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    # Iterate over the DataLoader for batch processing
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f"Loss for batch {batch_idx + 1}: {loss.item():.4f}")

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        print(f"Predicted: {predicted[:30]}")

        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

        # Print the loss and accuracy every 10 batches
        if (batch_idx + 1) % 10 == 0:
            accuracy = 100 * correct_preds / total_preds
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

    # Print the average loss and accuracy for the epoch
    avg_loss = running_loss / len(train_loader)
    avg_accuracy = 100 * correct_preds / total_preds
    print(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.2f}%\n")
    
    # Step the scheduler
    scheduler.step()

# Optionally: Save the trained model
torch.save(model.state_dict(), '/users/seb/Desktop/bcbl/msc_thesis/gru_deviant_detector.pth')


# unbalanced data class, use focal loss or add more data to it