"""Pseudocode TRM dataset."""
from __future__ import annotations

import torch


class TRMDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, graph_version: str, training_repo):
        self.examples = training_repo.load_examples(split=split, graph_version=graph_version)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {"question_tokens": torch.tensor(ex["question_tokens"]), "option_tokens": torch.tensor(ex["option_tokens"]), "entity_path": torch.tensor(ex["entity_path"]), "relation_path": torch.tensor(ex["relation_path"]), "edge_confidences": torch.tensor(ex["edge_confidences"]), "task_type_id": torch.tensor(ex["task_type_id"]), "metapath_id": torch.tensor(ex["metapath_id"]), "label": torch.tensor(ex["label"])}
