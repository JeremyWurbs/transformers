import torch.nn as nn


class MLP(nn.Module):
    """Single hidden layer MLP"""

    def __init__(self, input_dim, output_dim=None, hidden_dim=None, dropout=0.):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * input_dim
        if output_dim is None:
            output_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
