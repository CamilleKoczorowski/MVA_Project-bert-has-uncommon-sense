#!/usr/bin/env python
"""
extension3/contrastive_training.py — NT-Xent contrastive fine-tuning
=====================================================================
Fine-tunes a CWE model with Supervised NT-Xent loss on sense-labeled data.

Key design choices:
  - Batches contain multiple instances of the same lemma (required for contrastive loss)
  - Positives = same sense label, Negatives = different sense (same lemma or other)
  - No fixed margin threshold (unlike standard contrastive loss)
  - Temperature τ controls sharpness of the similarity distribution
  - Projection head (MLP) used during training, discarded at evaluation
"""

import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm


# ---------------------------------------------------------------------------
# NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
# ---------------------------------------------------------------------------
class SupervisedNTXentLoss(nn.Module):
    """
    Supervised NT-Xent loss (SupCon-style).
    For each anchor, all instances with the same sense are positives,
    everything else in the batch is negative.

    L = -1/|P(i)| * sum_{p in P(i)} log( exp(sim(z_i, z_p)/τ) / sum_{a!=i} exp(sim(z_i, z_a)/τ) )

    No fixed threshold needed — the softmax naturally handles the geometry.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (batch_size, dim) — L2-normalized projected embeddings
            labels: list of sense labels (strings), length batch_size
        Returns:
            scalar loss
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Cosine similarity matrix (embeddings are already L2-normalized)
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Build label mask: mask[i,j] = 1 if labels[i] == labels[j]
        label_ids = []
        label_to_id = {}
        for lbl in labels:
            if lbl not in label_to_id:
                label_to_id[lbl] = len(label_to_id)
            label_ids.append(label_to_id[lbl])
        label_ids = torch.tensor(label_ids, device=device)
        positive_mask = (label_ids.unsqueeze(0) == label_ids.unsqueeze(1)).float()

        # Remove self-similarity from masks
        self_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - self_mask

        # Remove self from denominator
        logits_mask = 1.0 - self_mask

        # For numerical stability
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Denominator: sum over all non-self entries
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Average over positive pairs
        num_positives = positive_mask.sum(dim=1)

        # Only compute loss for anchors that have at least one positive
        valid = num_positives > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (num_positives + 1e-12)
        loss = -mean_log_prob_pos[valid].mean()

        return loss


# ---------------------------------------------------------------------------
# Class-weighted NT-Xent variant
# ---------------------------------------------------------------------------
class ClassWeightedNTXentLoss(SupervisedNTXentLoss):
    """
    NT-Xent with per-anchor weighting by inverse sense frequency.
    Rare senses contribute more to the total loss.
    """

    def __init__(self, temperature=0.07, sense_weights=None):
        super().__init__(temperature)
        self.sense_weights = sense_weights or {}

    def forward(self, embeddings, labels):
        device = embeddings.device
        batch_size = embeddings.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        label_ids = []
        label_to_id = {}
        for lbl in labels:
            if lbl not in label_to_id:
                label_to_id[lbl] = len(label_to_id)
            label_ids.append(label_to_id[lbl])
        label_ids = torch.tensor(label_ids, device=device)
        positive_mask = (label_ids.unsqueeze(0) == label_ids.unsqueeze(1)).float()

        self_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - self_mask
        logits_mask = 1.0 - self_mask

        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        num_positives = positive_mask.sum(dim=1)
        valid = num_positives > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (num_positives + 1e-12)

        # Apply per-anchor weights
        weights = torch.tensor(
            [self.sense_weights.get(lbl, 1.0) for lbl in labels],
            device=device, dtype=torch.float,
        )
        weighted_loss = -(mean_log_prob_pos * weights)[valid].mean()

        return weighted_loss


# ---------------------------------------------------------------------------
# Projection head (discarded at evaluation time)
# ---------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    """Two-layer MLP projection head (SimCLR-style)."""

    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# Contrastive batch sampler: ensures each batch has multiple senses per lemma
# ---------------------------------------------------------------------------
class ContrastiveBatchSampler(Sampler):
    """
    Samples batches for contrastive learning.
    Each batch contains instances from N_lemmas lemmas,
    with K_per_lemma instances per lemma.
    Ensures each lemma has at least 2 instances (for positive pairs).
    """

    def __init__(
        self,
        labels,
        lemma_from_label_fn,
        n_lemmas_per_batch=8,
        k_per_lemma=8,
        oversample_indices=None,
    ):
        self.labels = labels
        self.lemma_fn = lemma_from_label_fn

        # Group indices by lemma
        self.lemma_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            lemma = self.lemma_fn(label)
            self.lemma_to_indices[lemma].append(idx)

        # Apply oversampling if provided
        if oversample_indices:
            for idx, factor in oversample_indices:
                label = labels[idx]
                lemma = self.lemma_fn(label)
                for _ in range(factor - 1):  # -1 because original is already there
                    self.lemma_to_indices[lemma].append(idx)

        # Only keep lemmas with at least 2 instances
        self.valid_lemmas = [
            lemma for lemma, indices in self.lemma_to_indices.items()
            if len(indices) >= 2
        ]

        self.n_lemmas = n_lemmas_per_batch
        self.k_per_lemma = k_per_lemma

    def __iter__(self):
        random.shuffle(self.valid_lemmas)
        for start in range(0, len(self.valid_lemmas), self.n_lemmas):
            batch_lemmas = self.valid_lemmas[start:start + self.n_lemmas]
            batch_indices = []
            for lemma in batch_lemmas:
                indices = self.lemma_to_indices[lemma]
                if len(indices) >= self.k_per_lemma:
                    sampled = random.sample(indices, self.k_per_lemma)
                else:
                    # With replacement if not enough
                    sampled = random.choices(indices, k=self.k_per_lemma)
                batch_indices.extend(sampled)
            yield batch_indices

    def __len__(self):
        return len(self.valid_lemmas) // self.n_lemmas


# ---------------------------------------------------------------------------
# Contrastive fine-tuning model
# ---------------------------------------------------------------------------
class ContrastiveFineTuningModel(nn.Module):
    """
    Wraps a HuggingFace transformer with a projection head for contrastive training.
    During training: transformer -> extract target token embedding -> project -> NT-Xent
    During evaluation: only the transformer weights are used (projection head discarded).
    """

    def __init__(self, transformer_model, hidden_dim=768, proj_hidden=256, proj_out=128):
        super().__init__()
        self.transformer = transformer_model
        self.projection = ProjectionHead(hidden_dim, proj_hidden, proj_out)

    def forward(self, input_ids, attention_mask, target_positions):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            target_positions: (batch,) — index of the target token in each sequence
        Returns:
            projected: (batch, proj_out) — L2-normalized projected embeddings
            raw_embeddings: (batch, hidden_dim) — raw transformer embeddings (for SMOTE)
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state: (batch, seq_len, hidden_dim)
        hidden_states = outputs.last_hidden_state

        # Extract target token embeddings
        batch_size = hidden_states.shape[0]
        raw_embeddings = hidden_states[torch.arange(batch_size), target_positions]

        # Project and normalize
        projected = self.projection(raw_embeddings)

        return projected, raw_embeddings


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_contrastive(
    model,
    dataloader,
    loss_fn,
    optimizer,
    num_epochs,
    device,
    smote_data=None,
    log_every=10,
    save_path=None,
):
    """
    Main contrastive training loop.

    Args:
        model: ContrastiveFineTuningModel
        dataloader: yields batches of (input_ids, attention_mask, target_positions, labels)
        loss_fn: SupervisedNTXentLoss or ClassWeightedNTXentLoss
        optimizer: torch optimizer
        num_epochs: number of training epochs
        device: torch device
        smote_data: optional dict {'synthetic': {sense: tensor}, ...} for embedding augmentation
        log_every: print loss every N steps
        save_path: path to save the transformer weights after training

    Returns:
        loss_history: list of (epoch, step, loss_value)
    """
    model.train()
    model.to(device)
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_positions = batch["target_positions"].to(device)
            labels = batch["labels"]  # list of strings

            projected, raw_embeddings = model(input_ids, attention_mask, target_positions)

            # Optionally add SMOTE synthetic embeddings
            if smote_data and "synthetic" in smote_data:
                all_projected = [projected]
                all_labels = list(labels)
                for sense, syn_embeds in smote_data["synthetic"].items():
                    # Project the synthetic embeddings through the head
                    syn_proj = model.projection(syn_embeds.to(device))
                    all_projected.append(syn_proj)
                    all_labels.extend([sense] * syn_proj.shape[0])
                projected = torch.cat(all_projected, dim=0)
                labels = all_labels

            loss = loss_fn(projected, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            num_batches += 1
            loss_history.append((epoch, step, loss_val))

            if step % log_every == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Step {step}, Loss: {loss_val:.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs} — Avg loss: {avg_loss:.4f}")

    # Save transformer weights (without projection head)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.transformer.state_dict(), save_path)
        print(f"[train] Saved transformer weights to {save_path}")

    return loss_history
