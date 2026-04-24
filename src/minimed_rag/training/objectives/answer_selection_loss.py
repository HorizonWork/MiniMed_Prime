"""Pseudocode answer selection loss."""

from __future__ import annotations

from torch import nn


class AnswerSelectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        return self.loss(logits.squeeze(-1), labels.float())
