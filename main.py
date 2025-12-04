import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Updated parameters
input_size = 3               
sequence_length = 8 # 8 sounds in a sequence
hidden_size = 64            
num_layers = 2               
num_classes = 3              
learning_rate = 0.0000001

class GRUDeviantDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUDeviantDetector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_hidden=False):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, hn = self.gru(x, h0)  # out: (batch_size, seq_len, hidden_size) # hidden layer
        final_hidden = hn[-1]

        logits = self.fc(final_hidden)

        if return_hidden:            
            # out: (batch, seq_len, hidden_size)
            return logits, out
        else:
            return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply softmax to get probabilities (logits -> probabilities)
        inputs = F.softmax(inputs, dim=1)
        
        # Get the probabilities for the correct class (from targets)
        targets = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        # Calculate the focal loss
        p_t = (inputs * targets).sum(dim=1)  # p_t is the probability of the true class
        loss = -self.alpha * (1 - p_t)**self.gamma * torch.log(p_t + 1e-8)  # add small epsilon to avoid log(0)

        # Reduce the loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Instantiate model
model = GRUDeviantDetector(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# add the noises
# vector thing