"""Pseudocode TRM training loop."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from biomed_kg.training.datasets.collator import trm_collator
from biomed_kg.training.datasets.trm_dataset import TRMDataset
from biomed_kg.training.models.trm import TinyRecursiveReasoner


class TRMTrainer:
    def __init__(self, config, graph_version: str, training_repo, entity_vocab, relation_vocab, task_type_vocab, metapath_vocab, experiment_tracker, checkpointing):
        self.config = config
        self.graph_version = graph_version
        self.training_repo = training_repo
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.task_type_vocab = task_type_vocab
        self.metapath_vocab = metapath_vocab
        self.experiment_tracker = experiment_tracker
        self.checkpointing = checkpointing

    def train(self):
        train_loader = DataLoader(TRMDataset("train", self.graph_version, self.training_repo), batch_size=self.config.batch_size, shuffle=True, collate_fn=trm_collator)
        val_loader = DataLoader(TRMDataset("val", self.graph_version, self.training_repo), batch_size=self.config.batch_size, shuffle=False, collate_fn=trm_collator)
        model = TinyRecursiveReasoner(self.entity_vocab.size, self.relation_vocab.size, self.task_type_vocab.size, self.metapath_vocab.size, self.config.hidden_dim, self.config.recursion_steps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        for epoch in range(self.config.num_epochs):
            model.train()
            for batch in train_loader:
                logits = model(batch["entity_path"], batch["relation_path"], batch["edge_confidences"], batch["task_type_id"], batch["metapath_id"])
                loss = loss_fn(logits.squeeze(-1), batch["label"].float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            metrics = self.evaluate(model, val_loader)
            self.experiment_tracker.log({"epoch": epoch, "val_auc": metrics.auc, "val_f1": metrics.f1, "val_path_accuracy": metrics.path_accuracy})
            self.checkpointing.save_if_best(model, metrics)
        return model

    def evaluate(self, model, val_loader):
        raise NotImplementedError
