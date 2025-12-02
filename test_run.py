### test_run.pu
### trying out for a good lr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from main import GRUDeviantDetector 

# --------------------------
# Hyperparameters (global)
# --------------------------
batch_size = 64
epochs = 15              # fewer epochs since we sweep LRs
sigma = 0.01             
learning_rates = [1e-4, 3e-4, 1e-3, 3e-3]

# Optional: for reproducibility
torch.manual_seed(42)

# --------------------------
# Device
# --------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# --------------------------
# Load data once
# --------------------------
input_tensor = torch.load("/users/seb/Desktop/bcbl/msc_thesis/input_tensor.pt")
labels_tensor = torch.load("/users/seb/Desktop/bcbl/msc_thesis/labels_tensor.pt")

# Remap labels: 4,5,6 -> 0,1,2
labels_tensor = torch.where(labels_tensor == 4, torch.tensor(0), labels_tensor)
labels_tensor = torch.where(labels_tensor == 5, torch.tensor(1), labels_tensor)
labels_tensor = torch.where(labels_tensor == 6, torch.tensor(2), labels_tensor)

labels_tensor = labels_tensor.long()  # CrossEntropyLoss expects Long

print("Class counts:", torch.unique(labels_tensor, return_counts=True))

dataset = TensorDataset(input_tensor.float(), labels_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --------------------------
# Helper functions
# --------------------------
def frequencies_to_onehot(x):
    """
    Convert raw frequency values {1455, 1500, 1600} into one-hot vectors.

    Input: x shape can be (N,), (N, T) or (N, T, 1) containing 1455/1500/1600.
    Output: same shape but last dim = 3 (one-hot).
    """
    x = x.float()

    # If last dim is 1, squeeze it (e.g. (batch, seq, 1) -> (batch, seq))
    if x.dim() >= 2 and x.size(-1) == 1:
        x = x.squeeze(-1)

    # Add feature dimension at the end
    x_exp = x.unsqueeze(-1)  # (..., 1)

    freq_vals = torch.tensor([1455.0, 1500.0, 1600.0], device=x.device)
    one_hot = (x_exp == freq_vals).float()  # broadcast -> (..., 3)

    return one_hot


def add_gaussian_noise(x, sigma):
    """Add Gaussian noise ~ N(0, sigma^2) to x, if sigma > 0."""
    if sigma is None or sigma == 0.0:
        return x
    noise = torch.randn_like(x) * sigma
    return x + noise

# --------------------------
# Training over multiple LRs
# --------------------------
history = {}   # {lr: {"loss": [..], "acc": [..]}}

for lr in learning_rates:
    print("=" * 70)
    print(f"Training with learning rate = {lr}")
    print("=" * 70)

    # Fresh model per LR
    model = GRUDeviantDetector(
        input_size=3,   # one-hot size
        hidden_size=64,
        num_layers=2,
        num_classes=3
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history[lr] = {"loss": [], "acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 1) freq -> one-hot
            one_hot_inputs = frequencies_to_onehot(inputs)

            # 2) optional noise
            noisy_inputs = add_gaussian_noise(one_hot_inputs, sigma)

            # 3) forward
            optimizer.zero_grad()
            outputs = model(noisy_inputs)

            # 4) loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # 5) backward + update
            loss.backward()
            optimizer.step()

            # 6) metrics
            _, predicted = torch.max(outputs, dim=1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                batch_acc = 100.0 * correct_preds / total_preds
                print(
                    f"[LR={lr:g}] Epoch [{epoch+1}/{epochs}], "
                    f"Batch [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%"
                )

        avg_loss = running_loss / len(train_loader)
        avg_acc = 100.0 * correct_preds / total_preds
        history[lr]["loss"].append(avg_loss)
        history[lr]["acc"].append(avg_acc)

        print(f"[LR={lr:g}] Epoch [{epoch+1}/{epochs}] - "
              f"Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2f}%\n")

    # Save model for this LR
    lr_str = str(lr).replace(".", "p").replace("-", "m")  # e.g. 0.001 -> '0p001'
    model_path = f"/users/seb/Desktop/bcbl/msc_thesis/gru_deviant_lr_{lr_str}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model for LR={lr} to: {model_path}")

# --------------------------
# Summary + plots
# --------------------------
print("\n=== Summary over learning rates ===")
for lr in learning_rates:
    final_loss = history[lr]["loss"][-1]
    final_acc = history[lr]["acc"][-1]
    print(f"LR={lr:g} -> final train loss={final_loss:.4f}, "
          f"final train acc={final_acc:.2f}%")

epochs_range = range(1, epochs + 1)

# 1) Accuracy curves
plt.figure(figsize=(8, 5))
for lr in learning_rates:
    plt.plot(epochs_range, history[lr]["acc"], marker="o", label=f"LR={lr:g}")
plt.xlabel("Epoch")
plt.ylabel("Training accuracy (%)")
plt.title("Training accuracy vs epoch for different learning rates")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("/users/seb/Desktop/bcbl/msc_thesis/lr_accuracy_curves.png", dpi=150)
plt.show()

# 2) Loss curves
plt.figure(figsize=(8, 5))
for lr in learning_rates:
    plt.plot(epochs_range, history[lr]["loss"], marker="o", label=f"LR={lr:g}")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("Training loss vs epoch for different learning rates")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("/users/seb/Desktop/bcbl/msc_thesis/lr_loss_curves.png", dpi=150)
plt.show()
