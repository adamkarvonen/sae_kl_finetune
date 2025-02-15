import torch
import torch.nn as nn


class LinearAdapter(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int):
        super().__init__()
        self.d_in = d_in
        self.hidden_dim = hidden_dim

        self.down_proj = nn.Linear(d_in, hidden_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, d_in, bias=False)

        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x):
        return self.up_proj(self.down_proj(x))
