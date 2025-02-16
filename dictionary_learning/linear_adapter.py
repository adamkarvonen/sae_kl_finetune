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
        orig_dtype = x.dtype
        x_fp32 = x.to(dtype=self.down_proj.weight.dtype)

        adapter_out = self.up_proj(self.down_proj(x_fp32))

        adapter_out = adapter_out.to(orig_dtype)
        return adapter_out


class MLPAdapter(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int):
        super().__init__()
        self.d_in = d_in
        self.hidden_dim = hidden_dim

        self.down_proj = nn.Linear(d_in, hidden_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(hidden_dim, d_in)

        nn.init.zeros_(self.up_proj.weight)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        orig_dtype = x.dtype
        x_fp32 = x.to(dtype=self.down_proj.weight.dtype)
        x_fp32 = self.down_proj(x_fp32)
        x_fp32 = self.activation(x_fp32)
        adapter_out = self.up_proj(x_fp32)

        adapter_out = adapter_out.to(orig_dtype)
        return adapter_out
