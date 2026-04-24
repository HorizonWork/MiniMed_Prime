"""Pseudocode task heads."""
from __future__ import annotations

from torch import nn


class BinaryValidityHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, hidden):
        return self.proj(hidden)
