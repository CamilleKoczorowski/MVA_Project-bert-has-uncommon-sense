#!/usr/bin/env python
"""
extension3/run_extension3.py — Main script for Extension 3 (FIXED)
===================================================================
Contrastive NT-Xent fine-tuning with augmentation for rare word senses.

FIXES vs previous version:
  - Paraphrases are now loaded and added as real training instances
  - SMOTE embeddings are recalculated each epoch with current model weights
  - All augmentation branches verified to actually modify training data

Usage: see --help or the notebook for examples.
"""

import argparse
import os
import sys
import json
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bssp.common.config import Config
from bssp.common.reading import read_dataset_cached
from bssp.common.analysis import dataset_stats
from bssp.clres.dataset_reader import ClresConlluReader, lemma_from_label as clres_lemma
from bssp.semcor.dataset_reader import SemcorReader, lemma_from_label as semcor_lemma

from extension3.augmentation import (
    compute_sense_frequencies,
    identify_rare_senses,
    load_paraphrases,
)
from extension3.contrastive_training import (
    SupervisedNTXentLoss,
    ClassWeightedNTXentLoss,
    ContrastiveFineTuningModel,
    ContrastiveBatchSampler,
)
from extension3.evaluation import (
    split_train_for_contrastive,
    evaluate_with_breakdown,
)

CORPUS_CONFIG = {
    "clres": {
        "reader": ClresConlluReader,
        "train_path": "data/pdep/pdep_train.conllu",
        "test_path": "data/pdep/pdep_test.conllu",
        "lemma_fn": clres_lemma,
    },
    "semcor": {
        "reader": SemcorReader,
        "train_path": None,
        "test_path": None,
        "lemma_fn": semcor_lemma,
    },
}

MODEL_HIDDEN_DIMS = {
    "bert-base-cased": 768, "roberta-base": 768, "distilbert-base-cased": 768,
    "distilroberta-base": 768, "albert-base-v2": 768, "xlnet-base-cased": 768, "gpt2": 768,
}
MODEL_NUM_LAYERS = {
    "bert-base-cased": 12, "roberta-base": 12, "distilbert-base-cased": 6,
    "distilroberta-base": 6, "albert-base-v2": 12, "xlnet-base-cased": 12, "gpt2": 12,
}


def get_last_layer(model_name):
    return MODEL_NUM_LAYERS[model_name] - 1


def get_results_dir(corpus, model, augmentation, loss, pretrained=False):
    config_name = f"{augmentation}_{loss}"
    if pretrained:
        config_name = f"streusle_then_{config_name}"
    return os.path.join(f"results/{corpus}_ext3", model, config_name)


def get_weights_path(results_dir):
    return os.path.join(results_dir, "contrastive_weights.pt")


# ---------------------------------------------------------------------------
# Contrastive Dataset — NOW with paraphrase support
# ---------------------------------------------------------------------------
class ContrastiveDataset(torch.utils.data.Dataset):
    """
    Dataset for contrastive training.
    Can include both original instances AND paraphrase-augmented instances.
    """

    def __init__(self, instances, ft_indices, tokenizer, max_length=128,
                 paraphrases_df=None, lemma_from_label_fn=None):
        """
        Args:
            instances: full training dataset
            ft_indices: indices of D_ft instances
            tokenizer: HuggingFace tokenizer
            paraphrases_df: DataFrame with paraphrase data (optional)
            lemma_from_label_fn: function to extract lemma from label
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.labels = []
        self.texts = []
        self.target_positions = []

        # Add original instances
        for idx in ft_indices:
            inst = instances[idx]
            self.labels.append(inst["label"].label)
            tokens = [t.text for t in inst["text"].tokens]
            self.texts.append(" ".join(tokens))
            self.target_positions.append(inst["label_span"].span_start)

        n_original = len(self.labels)

        # Add paraphrased instances if provided
        n_paraphrases = 0
        if paraphrases_df is not None and len(paraphrases_df) > 0:
            # Only add paraphrases for instances that are in D_ft
            ft_idx_set = set(ft_indices)
            for _, row in paraphrases_df.iterrows():
                inst_idx = int(row["instance_idx"])
                if inst_idx in ft_idx_set:
                    self.labels.append(row["sense_label"])
                    self.texts.append(row["paraphrase"])
                    # Approximate target position: find the lemma in the paraphrase
                    para_tokens = row["paraphrase"].split()
                    lemma = row["lemma"].lower()
                    target_pos = 0
                    for j, tok in enumerate(para_tokens):
                        if tok.lower() == lemma or lemma in tok.lower():
                            target_pos = j
                            break
                    self.target_positions.append(min(target_pos, max_length - 1))
                    n_paraphrases += 1

        print(f"[dataset] {n_original} original + {n_paraphrases} paraphrase = {len(self.labels)} total instances")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target_pos = min(self.target_positions[idx], self.max_length - 1)
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "target_positions": torch.tensor(target_pos, dtype=torch.long),
            "labels": self.labels[idx],
        }


def collate_contrastive(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "target_positions": torch.stack([b["target_positions"] for b in batch]),
        "labels": [b["labels"] for b in batch],
    }


# ---------------------------------------------------------------------------
# SMOTE: compute fresh synthetic embeddings from current model
# ---------------------------------------------------------------------------
def compute_smote_for_epoch(model, dataset, ft_indices, train_dataset, rare_senses, device):
    """
    Compute SMOTE synthetic embeddings using current model weights.
    Called at each epoch so embeddings stay up-to-date.
    """
    model.eval()
    embeddings_by_sense = defaultdict(list)

    with torch.no_grad():
        # Only process original instances (not paraphrases)
        n_original = len(ft_indices)
        for i in range(min(n_original, len(dataset))):
            label = dataset.labels[i]
            if label in rare_senses:
                item = dataset[i]
                _, raw_emb = model(
                    item["input_ids"].unsqueeze(0).to(device),
                    item["attention_mask"].unsqueeze(0).to(device),
                    item["target_positions"].unsqueeze(0).to(device),
                )
                embeddings_by_sense[label].append(raw_emb.squeeze(0).cpu())

    # Stack and generate synthetic
    embeddings_by_sense = {
        s: torch.stack(embs) for s, embs in embeddings_by_sense.items()
        if len(embs) >= 2
    }

    from extension3.augmentation import smote_embeddings
    synthetic, parents = smote_embeddings(embeddings_by_sense)

    model.train()
    return synthetic, parents


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(args, model_name):
    corpus_cfg = CORPUS_CONFIG[args.corpus]
    lemma_fn = corpus_cfg["lemma_fn"]
    last_layer = get_last_layer(model_name)
    hidden_dim = MODEL_HIDDEN_DIMS[model_name]

    results_dir = get_results_dir(
        args.corpus, model_name, args.augmentation, args.loss,
        pretrained=args.pretrained_weights is not None,
    )
    os.makedirs(results_dir, exist_ok=True)

    config = vars(args).copy()
    config["model"] = model_name
    config["last_layer"] = last_layer
    with open(os.path.join(results_dir, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  Loss: {args.loss}, Augmentation: {args.augmentation}")
    print(f"  FT instances: {args.ft_instances}, Seed: {args.seed}")
    if args.pretrained_weights:
        print(f"  STREUSLE pretrained: {args.pretrained_weights}")
    print(f"  Results: {results_dir}")
    print(f"{'='*60}")

    # --- Load data ---
    cfg = Config(
        args.corpus, embedding_model=model_name, override_weights_path=None,
        metric="cosine", top_n=50, query_n=1, bert_layers=[last_layer],
    )

    print("[data] Loading datasets...")
    train_dataset = read_dataset_cached(
        cfg, corpus_cfg["reader"], "train", corpus_cfg["train_path"], with_embeddings=True,
    )
    test_dataset = read_dataset_cached(
        cfg, corpus_cfg["reader"], "test", corpus_cfg["test_path"], with_embeddings=False,
    )

    dataset_stats("train", train_dataset, f"{args.corpus}_stats", lemma_fn)
    dataset_stats("test", test_dataset, f"{args.corpus}_stats", lemma_fn)

    # --- Split ---
    ft_indices, eval_indices, ft_senses = split_train_for_contrastive(
        train_dataset, args.ft_instances, lemma_fn,
        rare_threshold=args.rare_threshold, seed=args.seed,
    )
    with open(os.path.join(results_dir, "ft_indices.json"), "w") as f:
        json.dump(ft_indices, f)

    sense_freq, lemma_freq, sense_proportion = compute_sense_frequencies(train_dataset, lemma_fn)
    rare_senses = identify_rare_senses(sense_proportion, args.rare_threshold)
    print(f"[data] Rare senses (r < {args.rare_threshold}): {len(rare_senses)}")

    # --- Training ---
    weights_path = get_weights_path(results_dir)

    if not args.eval_only:
        print("\n[train] Starting contrastive fine-tuning...")
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformer = AutoModel.from_pretrained(model_name)

        if args.pretrained_weights and os.path.isfile(args.pretrained_weights):
            print(f"[train] Loading STREUSLE pretrained weights: {args.pretrained_weights}")
            state_dict = torch.load(args.pretrained_weights, map_location="cpu")
            transformer.load_state_dict(state_dict, strict=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = ContrastiveFineTuningModel(
            transformer, hidden_dim=hidden_dim, proj_hidden=256, proj_out=128,
        )

        # --- Load paraphrases if needed ---
        paraphrases_df = None
        if args.augmentation in ("paraphrase", "paraphrase+smote"):
            para_path = f"results/{args.corpus}_ext3/paraphrases/paraphrases.tsv"
            if os.path.isfile(para_path):
                paraphrases_df = load_paraphrases(para_path)
                print(f"[augment] Loaded {len(paraphrases_df)} valid paraphrases from {para_path}")
            else:
                print(f"[augment] WARNING: paraphrases file not found at {para_path}")
                print(f"[augment] Falling back to no paraphrase augmentation")

        # --- Build dataset WITH paraphrases ---
        dataset = ContrastiveDataset(
            train_dataset, ft_indices, tokenizer,
            paraphrases_df=paraphrases_df,
            lemma_from_label_fn=lemma_fn,
        )

        # --- Build sampler ---
        # Use ALL labels (original + paraphrases) for the sampler
        oversample_idx = None
        if args.augmentation == "oversample":
            # For oversampling, duplicate rare-sense indices in the sampler
            from extension3.augmentation import oversample_indices as compute_oversample
            from extension3.augmentation import apply_augmentation
            aug_data = apply_augmentation(
                "oversample", train_dataset, rare_senses, sense_proportion, lemma_fn,
            )
            ft_set = set(ft_indices)
            oversample_idx = []
            for idx, factor in aug_data["indices"]:
                if idx in ft_set:
                    local_idx = ft_indices.index(idx)
                    oversample_idx.append((local_idx, factor))

        sampler = ContrastiveBatchSampler(
            dataset.labels, lemma_fn,
            n_lemmas_per_batch=args.batch_lemmas,
            k_per_lemma=args.batch_k,
            oversample_indices=oversample_idx,
        )

        # --- Build loss ---
        if args.loss == "nt-xent":
            loss_fn = SupervisedNTXentLoss(temperature=args.temperature)
        elif args.loss == "class-weighted-nt-xent":
            sense_weights = {label: 1.0 / (r + 0.01) for label, r in sense_proportion.items()}
            loss_fn = ClassWeightedNTXentLoss(
                temperature=args.temperature, sense_weights=sense_weights,
            )
        else:
            raise ValueError(f"Unknown loss: {args.loss}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        use_smote = args.augmentation in ("smote", "paraphrase+smote")

        # --- Training loop ---
        print(f"[train] Device: {device}")
        print(f"[train] Epochs: {args.epochs}, LR: {args.lr}, Temperature: {args.temperature}")
        print(f"[train] Dataset size: {len(dataset)} (includes paraphrases: {paraphrases_df is not None})")
        print(f"[train] SMOTE: {use_smote}")

        model.train()
        model.to(device)
        loss_history = []

        for epoch in range(args.epochs):
            # Recompute SMOTE each epoch with current weights
            smote_synthetic = None
            if use_smote:
                smote_synthetic, smote_parents = compute_smote_for_epoch(
                    model, dataset, ft_indices, train_dataset, rare_senses, device,
                )
                if epoch == 0:
                    # Save parents from first epoch for visualization
                    parents_ser = {
                        s: [(int(a), int(b), float(c)) for a, b, c in p]
                        for s, p in smote_parents.items()
                    }
                    with open(os.path.join(results_dir, "smote_parents.json"), "w") as f:
                        json.dump(parents_ser, f, indent=2)

            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for indices in sampler:
                batch_items = [dataset[i] for i in indices if i < len(dataset)]
                if not batch_items:
                    continue
                batch = collate_contrastive(batch_items)

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_positions = batch["target_positions"].to(device)
                labels = batch["labels"]

                projected, raw_embeddings = model(input_ids, attention_mask, target_positions)

                # Inject SMOTE synthetic points into the batch
                if smote_synthetic:
                    all_proj = [projected]
                    all_labels = list(labels)
                    for sense, syn_emb in smote_synthetic.items():
                        syn_proj = model.projection(syn_emb.to(device))
                        all_proj.append(syn_proj)
                        all_labels.extend([sense] * syn_proj.shape[0])
                    projected = torch.cat(all_proj, dim=0)
                    labels = all_labels

                loss = loss_fn(projected, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            loss_history.append({"epoch": epoch + 1, "avg_loss": round(avg_loss, 6)})
            print(f"  Epoch {epoch+1}/{args.epochs} — Avg loss: {avg_loss:.4f}")

        # Save transformer weights (no projection head)
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        torch.save(model.transformer.state_dict(), weights_path)
        print(f"[train] Saved weights to {weights_path}")

        with open(os.path.join(results_dir, "loss_history.json"), "w") as f:
            json.dump(loss_history, f, indent=2)

    # --- Evaluation ---
    if not args.no_eval:
        print("\n[eval] Running similarity ranking evaluation...")
        eval_cfg = Config(
            args.corpus, embedding_model=model_name,
            override_weights_path=weights_path if os.path.isfile(weights_path) else None,
            metric="cosine", top_n=50, query_n=1, bert_layers=[last_layer],
        )

        results = evaluate_with_breakdown(
            eval_cfg, args.corpus, train_dataset, test_dataset,
            eval_indices, ft_senses, lemma_fn, results_dir,
        )

        print(f"\n[eval] Results saved to {results_dir}/map_results.json")
        if "global" in results:
            print("\n--- Global MAP@50 ---")
            for k, v in results["global"].items():
                if v is not None and not str(k).endswith("(count)"):
                    print(f"  {k}: {v}")
        if "unseen" in results:
            print("\n--- Unseen queries MAP@50 ---")
            for k, v in results["unseen"].items():
                if v is not None and not str(k).endswith("(count)"):
                    print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extension 3 — Contrastive fine-tuning with augmentation for rare WSD",
    )
    parser.add_argument("--corpus", default="clres", choices=list(CORPUS_CONFIG.keys()))
    parser.add_argument("--models", nargs="+", default=["bert-base-cased"],
                        choices=list(MODEL_HIDDEN_DIMS.keys()))
    parser.add_argument("--loss", default="nt-xent",
                        choices=["nt-xent", "class-weighted-nt-xent"])
    parser.add_argument("--augmentation", default="none",
                        choices=["none", "oversample", "paraphrase", "smote", "paraphrase+smote"])
    parser.add_argument("--ft-instances", type=int, default=500)
    parser.add_argument("--pretrained-weights", default=None)
    parser.add_argument("--rare-threshold", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-lemmas", type=int, default=8)
    parser.add_argument("--batch-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--visualize-only", action="store_true")
    parser.add_argument("--no-eval", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.visualize_only:
        from extension3.visualize_results import main as viz_main
        sys.argv = ["", "--corpus", args.corpus]
        viz_main()
        return

    for model_name in args.models:
        run_experiment(args, model_name)

    print("\n[viz] Generating visualizations...")
    try:
        from extension3.visualize_results import (
            build_comparison_table, print_comparison_table,
            generate_latex_table, plot_seen_unseen,
        )
        rdir = f"results/{args.corpus}_ext3"
        df = build_comparison_table(rdir)
        print_comparison_table(df, rdir)
        generate_latex_table(df, rdir)
        plot_seen_unseen(df, rdir)
    except Exception as e:
        print(f"[viz] Warning: visualization failed: {e}")

    print("\n[done] All experiments complete.")


if __name__ == "__main__":
    main()
