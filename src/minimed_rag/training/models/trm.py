"""Pseudocode Tiny Recursive Reasoner model."""

from __future__ import annotations

import torch
from torch import nn


class TinyRecursiveReasoner(nn.Module):
    def __init__(
        self,
        num_entities,
        num_relations,
        num_task_types,
        num_metapaths,
        hidden_dim,
        recursion_steps,
    ):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, hidden_dim)
        self.relation_emb = nn.Embedding(num_relations, hidden_dim)
        self.task_emb = nn.Embedding(num_task_types, hidden_dim)
        self.metapath_emb = nn.Embedding(num_metapaths, hidden_dim)
        self.input_proj = nn.Linear(hidden_dim * 3 + 1, hidden_dim)
        self.recursive_cell = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.answer_head = nn.Linear(hidden_dim, 1)
        self.recursion_steps = recursion_steps

    def encode_path(self, entity_path, relation_path, edge_confidences):
        step_states = []
        for i in range(relation_path.shape[1]):
            subject = self.entity_emb(entity_path[:, i])
            predicate = self.relation_emb(relation_path[:, i])
            obj = self.entity_emb(entity_path[:, i + 1])
            confidence = edge_confidences[:, i].unsqueeze(-1)
            step_input = torch.cat([subject, predicate, obj, confidence], dim=-1)
            step_states.append(self.input_proj(step_input))
        return torch.stack(step_states, dim=0).mean(dim=0)

    def forward(self, entity_path, relation_path, edge_confidences, task_type_id, metapath_id):
        z = self.encode_path(entity_path, relation_path, edge_confidences)
        z = z + self.task_emb(task_type_id) + self.metapath_emb(metapath_id)
        for _ in range(self.recursion_steps):
            z = z + self.recursive_cell(z)
        return self.answer_head(z)
