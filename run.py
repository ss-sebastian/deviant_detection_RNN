# run.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from main import GRUDeviantDetector  # your model class


# --------------------------
# Hyperparameters
# --------------------------
batch_size = 64
max_epochs = 50          # upper bound (we have early-stopping)
learning_rate = 0.002
sigma = 0.01            
val_ratio = 0.2          # ???not sure if we need this
patience = 5             # early-stopping patience

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
# Load data
# --------------------------
input_tensor = torch.load("/users/seb/Desktop/bcbl/msc_thesis/input_tensor.pt")
labels_tensor = torch.load("/users/seb/Desktop/bcbl/msc_thesis/labels_tensor.pt")

# Remap labels: 4,5,6 -> 0,1,2
labels_tensor = torch.where(labels_tensor == 4, torch.tensor(0), labels_tensor)
labels_tensor = torch.where(labels_tensor == 5, torch.tensor(1), labels_tensor)
labels_tensor = torch.where(labels_tensor == 6, torch.tensor(2), labels_tensor)

labels_tensor = labels_tensor.long()  # CrossEntropyLoss expects long

print("Class counts:", torch.unique(labels_tensor, return_counts=True))

dataset = TensorDataset(input_tensor.float(), labels_tensor)

# Train/validation split
n_total = len(dataset)
n_val = int(val_ratio * n_total)
n_train = n_total - n_val
train_dataset, val_dataset = random_split(
    dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# --------------------------
# frequency transfrom and adding noises
# --------------------------
def frequencies_to_onehot(x: torch.Tensor) -> torch.Tensor:
    """
    Convert raw frequency values {1455, 1500, 1600} into one-hot vectors.

    Input shape:  (N,), (N, T) or (N, T, 1) with values 1455/1500/1600
    Output shape: same but last dimension = 3 (one-hot).
    """
    x = x.float()

    # If last dim is 1, e.g. (batch, seq, 1) -> (batch, seq)
    if x.dim() >= 2 and x.size(-1) == 1:
        x = x.squeeze(-1)

    # Add feature dimension at the end: (..., 1)
    x_exp = x.unsqueeze(-1)

    freq_vals = torch.tensor([1455.0, 1500.0, 1600.0], device=x.device)
    one_hot = (x_exp == freq_vals).float()  # broadcast to (..., 3)

    return one_hot


def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian noise ~ N(0, sigma^2) to x if sigma > 0."""
    if sigma is None or sigma == 0.0:
        return x
    noise = torch.randn_like(x) * sigma
    return x + noise


# --------------------------
# Model, loss, optimiser
# --------------------------
model = GRUDeviantDetector(
    input_size=3,      # one-hot size
    hidden_size=64,
    num_layers=2,
    num_classes=3
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# --------------------------
# Training loop with early stopping
# --------------------------
best_val_loss = float("inf")
best_state_dict = None
epochs_no_improve = 0

train_loss_hist = []
val_loss_hist = []
train_acc_hist = []
val_acc_hist = []

for epoch in range(1, max_epochs + 1):
    # --------- Train ---------
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        one_hot_inputs = frequencies_to_onehot(inputs)
        noisy_inputs = add_gaussian_noise(one_hot_inputs, sigma)

        optimizer.zero_grad()
        outputs = model(noisy_inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100.0 * train_correct / train_total

    # --------- Validation ---------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            one_hot_inputs = frequencies_to_onehot(inputs)
            noisy_inputs = add_gaussian_noise(one_hot_inputs, sigma)

            outputs = model(noisy_inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100.0 * val_correct / val_total

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%"
    )

    # --------- Early stopping check ---------
    if avg_val_loss < best_val_loss - 1e-4:  # small tolerance
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # store a CPU copy of best weights
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

# --------------------------
# Load best weights & save
# --------------------------
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
    model.to(device)
    print(f"Loaded best model with validation loss = {best_val_loss:.4f}")

save_path = "/users/seb/Desktop/bcbl/msc_thesis/gru_deviant_lr_0p002_earlystop.pth"
torch.save(model.state_dict(), save_path)
print(f"Saved best model to: {save_path}")
# --------------------------
# Plots
# --------------------------
epochs_range = range(1, len(train_loss_hist) + 1)

# 1) Loss curves
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_loss_hist, marker="o", label="Train loss")
plt.plot(epochs_range, val_loss_hist, marker="o", label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
loss_fig_path = "/users/seb/Desktop/bcbl/msc_thesis/loss_curves_lr_0p002.png"
plt.savefig(loss_fig_path, dpi=150)
print(f"Saved loss curves to: {loss_fig_path}")
plt.show()

# 2) Accuracy curves
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_acc_hist, marker="o", label="Train accuracy")
plt.plot(epochs_range, val_acc_hist, marker="o", label="Val accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and validation accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
acc_fig_path = "/users/seb/Desktop/bcbl/msc_thesis/accuracy_curves_lr_0p002.png"
plt.savefig(acc_fig_path, dpi=150)
print(f"Saved accuracy curves to: {acc_fig_path}")
plt.show()