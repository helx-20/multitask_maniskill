import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    """Per-step MLP with residual blocks and optional projection for mismatched dims.

    Supports two construction styles:
      - pass `hiddens` as a list of hidden widths (e.g. [512,512,512,512,512])
      - or pass scalar `hidden` and `hidden_layer` to create `[hidden]*hidden_layer`.

    The network maps `(B, input_dim)` -> `(B, num_classes)` logits.
    Residual connections project the skip path when input/output dims differ.
    """

    def __init__(
        self,
        input_dim: int = 51,
        hidden: int = 512,
        hidden_layer: int = 5,
        num_classes: int = 2,
    ):
        super().__init__()
        hiddens = [hidden] * hidden_layer

        # initial projection from input to first hidden
        self.in_net = nn.Sequential(nn.Linear(input_dim, hiddens[0]), nn.ReLU())

        # residual blocks: each block maps hiddens[i] -> hiddens[i+1]
        self.blocks = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        for i in range(len(hiddens) - 1):
            in_dim = hiddens[i]
            out_dim = hiddens[i + 1]
            self.blocks.append(nn.Linear(in_dim, out_dim))
            if in_dim != out_dim:
                self.skip_projs.append(nn.Linear(in_dim, out_dim))
            else:
                self.skip_projs.append(None)
            self.relu_layers.append(nn.ReLU())

        # final classifier head
        self.out_net = nn.Linear(hiddens[-1], num_classes)

    def forward(self, x):
        # x: (B, input_dim) -> logits: (B, num_classes)
        x = self.in_net(x)
        # apply residual blocks
        for lin, skip_proj, relu in zip(self.blocks, self.skip_projs, self.relu_layers):
            y = lin(x)
            if skip_proj is None:
                skip = x
            else:
                skip = skip_proj(x)
            x = relu(y + skip)
        x = self.out_net(x)
        return x
