import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    """Per-step MLP that maps a single (obs + force) feature vector to binary logits.

    StackCube convention:
      - input_dim = 48 (state obs) + 3 (unit force fx,fy,fz in [-1,1]) = 51
      - num_classes = 2: index 0 = safe (neg), index 1 = critical/crash (pos)
    """

    def __init__(self, input_dim: int = 51, hidden: int = 256, hidden_layer: int = 1, num_classes: int = 2):
        super().__init__()
        hidden_layers = [nn.Linear(hidden, hidden), nn.ReLU()] * hidden_layer
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        # x: (B, input_dim) -> logits: (B, num_classes)
        return self.net(x)
