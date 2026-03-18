#!/usr/bin/env python
"""
extension3/run_extension3.py — Main script for Extension 3
============================================================
Fine-tunes CWE models with contrastive NT-Xent loss and augmentation
strategies targeting rare word senses, then evaluates with the paper's
similarity ranking protocol (MAP@50).

Usage examples:
---------------
# 1) NT-Xent only, no augmentation, BERT on PDEP
python extension3/run_extension3.py \\
    --corpus clres --models bert-base-cased \\
    --loss nt-xent --augmentation none --ft-instances 500

# 2) NT-Xent + paraphrase augmentation
python extension3/run_extension3.py \\
    --corpus clres --models bert-base-cased roberta-base \\
    --loss nt-xent --augmentation paraphrase --ft-instances 500

# 3) NT-Xent + SMOTE
python extension3/run_extension3.py \\
    --corpus clres --models bert-base-cased \\
    --loss nt-xent --augmentation smote --ft-instances 500

# 4) Two-stage: STREUSLE init -> NT-Xent + paraphrase
python extension3/run_extension3.py \\
    --corpus clres --models bert-base-cased \\
    --loss nt-xent --augmentation paraphrase --ft-instances 500 \\
    --pretrained-weights models/bert-base-cased_500.pt

# 5) Evaluate only (skip training, use existing weights)
python extension3/run_extension3.py \\
    --corpus clres --models bert-base-cased \\
    --loss nt-xent --augmentation none --ft-instances 500 \\
    --eval-only

# 6) Visualize all results
python extension3/run_extension3.py --corpus clres --visualize-only

Arguments:
----------
--corpus            Dataset: clres (PDEP), semcor (default: clres)
--models            Model(s) to fine-tune and evaluate (default: bert-base-cased)
--loss              Loss function: nt-xent, class-weighted-nt-xent (default: nt-xent)
--augmentation      Strategy: none, oversample, paraphrase, smote, paraphrase+smote
--ft-instances      Number of fine-tuning instances from PDEP train (default: 500)
--pretrained-weights  Path to STREUSLE-pretrained weights (two-stage fine-tuning)
--rare-threshold    Proportional frequency threshold for rare senses (default: 0.25)
--temperature       NT-Xent temperature τ (default: 0.07)
--epochs            Training epochs (default: 20)
--lr                Learning rate (default: 2e-5)
--batch-lemmas      Lemmas per batch (default: 8)
--batch-k           Instances per lemma per batch (default: 8)
--seed              Random seed (default: 42)
--eval-only         Skip training, only evaluate with existing weights
--visualize-only    Only generate plots from existing results
--no-eval           Skip evaluation (training only)

Results go to:
    results/{corpus}_ext3/{model}/{augmentation}_{loss}/
"""

import argparse
import os
import sys
import json
import random
from collections import Counter, defaultdict

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bssp.common.config import Config
from bssp.common.reading import read_dataset_cached, make_indexer, make_embedder
from bssp.common.analysis import dataset_stats
from bssp.clres.dataset_reader import ClresConlluReader, lemma_from_label as clres_lemma
from bssp.semcor.dataset_reader import SemcorReader, lemma_from_label as semcor_lemma

from extension3.augmentation import (
    compute_sense_frequencies,
    identify_rare_senses,
    apply_augmentation,
    load_paraphrases,
)
from extension3.contrastive_training import (
    SupervisedNTXentLoss,
    ClassWeightedNTXentLoss,
    ContrastiveFineTuningModel,
    ContrastiveBatchSampler,
    train_contrastive,
)
from extension3.evaluation import (
    split_train_for_contrastive,
    evaluate_with_breakdown,
)


# ---------------------------------------------------------------------------
# Corpus configs
# ---------------------------------------------------------------------------
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
    "bert-base-cased": 768,
    "roberta-base": 768,
    "distilbert-base-cased": 768,
    "distilroberta-base": 768,
    "albert-base-v2": 768,
    "xlnet-base-cased": 768,
    "gpt2": 768,
}

MODEL_NUM_LAYERS = {
    "bert-base-cased": 12,
    "roberta-base": 12,
    "distilbert-base-cased": 6,
    "distilroberta-base": 6,
    "albert-base-v2": 12,
    "xlnet-base-cased": 12,
    "gpt2": 12,
}


def get_last_layer(model_name):
    """Return the 0-indexed last layer for a model."""
    return MODEL_NUM_LAYERS[model_name] - 1


# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------
def get_results_dir(corpus, model, augmentation, loss, pretrained=False):
    """Build the results directory path from experiment config."""
    config_name = f"{augmentation}_{loss}"
    if pretrained:
        config_name = f"streusle_then_{config_name}"
    return os.path.join(f"results/{corpus}_ext3", model, config_name)


def get_weights_path(results_dir):
    """Path to save/load contrastive fine-tuned weights."""
    return os.path.join(results_dir, "contrastive_weights.pt")


# ---------------------------------------------------------------------------
# Build contrastive dataset from D_ft
# ---------------------------------------------------------------------------
class ContrastiveDataset(torch.utils.data.Dataset):
    """
    Wraps D_ft instances for contrastive training.
    Each item returns tokenized input + target position + sense label.
    """

    def __init__(self, instances, ft_indices, tokenizer, max_length=128):
        self.instances = [instances[i] for i in ft_indices]
        self.ft_indices = ft_indices
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.labels = []
        self.texts = []
        self.target_positions = []

        for inst in self.instances:
            self.labels.append(inst["label"].label)
            tokens = [t.text for t in inst["text"].tokens]
            self.texts.append(" ".join(tokens))
            self.target_positions.append(inst["label_span"].span_start)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Map original target position to tokenized position
        # Approximate: find the subword token closest to the original position
        target_pos = min(self.target_positions[idx], self.max_length - 1)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "target_positions": torch.tensor(target_pos, dtype=torch.long),
            "labels": self.labels[idx],
        }


def collate_contrastive(batch):
    """Custom collate that handles string labels."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "target_positions": torch.stack([b["target_positions"] for b in batch]),
        "labels": [b["labels"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_experiment(args, model_name):
    """Run a single experiment for one model."""
    corpus_cfg = CORPUS_CONFIG[args.corpus]
    lemma_fn = corpus_cfg["lemma_fn"]
    last_layer = get_last_layer(model_name)
    hidden_dim = MODEL_HIDDEN_DIMS[model_name]

    results_dir = get_results_dir(
        args.corpus, model_name, args.augmentation, args.loss,
        pretrained=args.pretrained_weights is not None,
    )
    os.makedirs(results_dir, exist_ok=True)

    # Save experiment config
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
        args.corpus,
        embedding_model=model_name,
        override_weights_path=None,  # will be set after training
        metric="cosine",
        top_n=50,
        query_n=1,
        bert_layers=[last_layer],
    )

    print("[data] Loading datasets...")
    train_dataset = read_dataset_cached(
        cfg, corpus_cfg["reader"], "train", corpus_cfg["train_path"], with_embeddings=True,
    )
    test_dataset = read_dataset_cached(
        cfg, corpus_cfg["reader"], "test", corpus_cfg["test_path"], with_embeddings=False,
    )

    # Write stats
    dataset_stats("train", train_dataset, f"{args.corpus}_stats", lemma_fn)
    dataset_stats("test", test_dataset, f"{args.corpus}_stats", lemma_fn)

    # --- Split D_ft / D_eval ---
    ft_indices, eval_indices, ft_senses = split_train_for_contrastive(
        train_dataset, args.ft_instances, lemma_fn,
        rare_threshold=args.rare_threshold, seed=args.seed,
    )

    # Save split indices for reproducibility
    with open(os.path.join(results_dir, "ft_indices.json"), "w") as f:
        json.dump(ft_indices, f)

    # --- Compute sense frequencies ---
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

        # Load pretrained weights if two-stage
        if args.pretrained_weights and os.path.isfile(args.pretrained_weights):
            print(f"[train] Loading STREUSLE pretrained weights: {args.pretrained_weights}")
            state_dict = torch.load(args.pretrained_weights, map_location="cpu")
            transformer.load_state_dict(state_dict, strict=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build contrastive model
        model = ContrastiveFineTuningModel(
            transformer, hidden_dim=hidden_dim,
            proj_hidden=256, proj_out=128,
        )

        # Build dataset and sampler
        ft_labels = [train_dataset[i]["label"].label for i in ft_indices]

        # Apply augmentation to get oversampling indices if needed
        oversample_idx = None
        if args.augmentation == "oversample":
            aug_data = apply_augmentation(
                "oversample", train_dataset, rare_senses, sense_proportion, lemma_fn,
            )
            # Remap to D_ft indices
            ft_set = set(ft_indices)
            oversample_idx = [
                (ft_indices.index(idx), factor)
                for idx, factor in aug_data["indices"]
                if idx in ft_set
            ]

        sampler = ContrastiveBatchSampler(
            ft_labels, lemma_fn,
            n_lemmas_per_batch=args.batch_lemmas,
            k_per_lemma=args.batch_k,
            oversample_indices=oversample_idx,
        )

        # Build dataloader
        dataset = ContrastiveDataset(train_dataset, ft_indices, tokenizer)

        def batch_from_sampler(sampler, dataset):
            """Yield batches from the sampler."""
            for indices in sampler:
                batch = [dataset[i] for i in indices if i < len(dataset)]
                if batch:
                    yield collate_contrastive(batch)

        # Build loss
        if args.loss == "nt-xent":
            loss_fn = SupervisedNTXentLoss(temperature=args.temperature)
        elif args.loss == "class-weighted-nt-xent":
            # Weights = 1 / (r + epsilon) for each sense
            sense_weights = {
                label: 1.0 / (r + 0.01)
                for label, r in sense_proportion.items()
            }
            loss_fn = ClassWeightedNTXentLoss(
                temperature=args.temperature, sense_weights=sense_weights,
            )
        else:
            raise ValueError(f"Unknown loss: {args.loss}")

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # Handle SMOTE augmentation
        smote_data = None
        if args.augmentation in ("smote", "paraphrase+smote"):
            print("[train] Computing SMOTE embeddings for rare senses...")
            # Extract embeddings for rare senses in D_ft
            model.eval()
            model.to(device)
            embeddings_by_sense = defaultdict(list)
            with torch.no_grad():
                for i, idx in enumerate(ft_indices):
                    label = train_dataset[idx]["label"].label
                    if label in rare_senses:
                        item = dataset[i]
                        _, raw_emb = model(
                            item["input_ids"].unsqueeze(0).to(device),
                            item["attention_mask"].unsqueeze(0).to(device),
                            item["target_positions"].unsqueeze(0).to(device),
                        )
                        embeddings_by_sense[label].append(raw_emb.squeeze(0).cpu())

            # Stack into tensors
            embeddings_by_sense = {
                s: torch.stack(embs) for s, embs in embeddings_by_sense.items()
                if len(embs) >= 2
            }

            from extension3.augmentation import smote_embeddings
            synthetic, parents = smote_embeddings(embeddings_by_sense)
            smote_data = {"synthetic": synthetic, "parents": parents}

            # Save SMOTE parents for visualization
            parents_serializable = {
                s: [(int(a), int(b), float(c)) for a, b, c in p]
                for s, p in parents.items()
            }
            with open(os.path.join(results_dir, "smote_parents.json"), "w") as f:
                json.dump(parents_serializable, f, indent=2)

            model.train()

        # Training loop
        print(f"[train] Device: {device}")
        print(f"[train] Epochs: {args.epochs}, LR: {args.lr}, Temperature: {args.temperature}")

        # Simple epoch-based training with the sampler
        model.train()
        model.to(device)
        loss_history = []

        for epoch in range(args.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in batch_from_sampler(sampler, dataset):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_positions = batch["target_positions"].to(device)
                labels = batch["labels"]

                projected, raw_embeddings = model(input_ids, attention_mask, target_positions)

                # Add SMOTE synthetic points if available
                if smote_data and "synthetic" in smote_data:
                    all_proj = [projected]
                    all_labels = list(labels)
                    for sense, syn_emb in smote_data["synthetic"].items():
                        syn_proj = model.projection(syn_emb.to(device))
                        all_proj.append(syn_proj)
                        all_labels.extend([sense] * syn_proj.shape[0])
                    projected = torch.cat(all_proj, dim=0)
                    labels = all_labels

                loss = loss_fn(projected, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            loss_history.append({"epoch": epoch + 1, "avg_loss": round(avg_loss, 6)})
            print(f"  Epoch {epoch+1}/{args.epochs} — Avg loss: {avg_loss:.4f}")

        # Save weights (transformer only, no projection head)
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        torch.save(model.transformer.state_dict(), weights_path)
        print(f"[train] Saved weights to {weights_path}")

        # Save loss history
        with open(os.path.join(results_dir, "loss_history.json"), "w") as f:
            json.dump(loss_history, f, indent=2)

    # --- Evaluation ---
    if not args.no_eval:
        print("\n[eval] Running similarity ranking evaluation...")

        # Build config pointing to our fine-tuned weights
        eval_cfg = Config(
            args.corpus,
            embedding_model=model_name,
            override_weights_path=weights_path if os.path.isfile(weights_path) else None,
            metric="cosine",
            top_n=50,
            query_n=1,
            bert_layers=[last_layer],
        )

        results = evaluate_with_breakdown(
            eval_cfg, args.corpus,
            train_dataset, test_dataset,
            eval_indices, ft_senses,
            lemma_fn, results_dir,
        )

        print(f"\n[eval] Results saved to {results_dir}/map_results.json")

        # Print summary
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--corpus", default="clres", choices=list(CORPUS_CONFIG.keys()),
                        help="Dataset (default: clres = PDEP)")
    parser.add_argument("--models", nargs="+", default=["bert-base-cased"],
                        choices=list(MODEL_HIDDEN_DIMS.keys()),
                        help="Model(s) to evaluate (default: bert-base-cased)")
    parser.add_argument("--loss", default="nt-xent",
                        choices=["nt-xent", "class-weighted-nt-xent"],
                        help="Loss function (default: nt-xent)")
    parser.add_argument("--augmentation", default="none",
                        choices=["none", "oversample", "paraphrase", "smote", "paraphrase+smote"],
                        help="Augmentation strategy (default: none)")
    parser.add_argument("--ft-instances", type=int, default=500,
                        help="Number of fine-tuning instances (default: 500)")
    parser.add_argument("--pretrained-weights", default=None,
                        help="Path to STREUSLE weights for two-stage fine-tuning")
    parser.add_argument("--rare-threshold", type=float, default=0.25,
                        help="Proportional frequency threshold for rare senses (default: 0.25)")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="NT-Xent temperature τ (default: 0.07)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch-lemmas", type=int, default=8,
                        help="Lemmas per batch (default: 8)")
    parser.add_argument("--batch-k", type=int, default=8,
                        help="Instances per lemma per batch (default: 8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate")
    parser.add_argument("--visualize-only", action="store_true",
                        help="Only generate plots from existing results")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation (training only)")

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Visualize-only mode
    if args.visualize_only:
        from extension3.visualize_results import main as viz_main
        sys.argv = ["", "--corpus", args.corpus]
        viz_main()
        return

    # Run experiments
    for model_name in args.models:
        run_experiment(args, model_name)

    # Generate visualizations at the end
    print("\n[viz] Generating visualizations...")
    try:
        from extension3.visualize_results import (
            build_comparison_table,
            print_comparison_table,
            generate_latex_table,
            plot_seen_unseen,
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
