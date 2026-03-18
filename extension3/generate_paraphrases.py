#!/usr/bin/env python
"""
extension3/generate_paraphrases.py — Offline paraphrase generation
===================================================================
Generates meaning-preserving paraphrases for rare-sense instances
using T5 and saves them to a TSV file for inspection and later use.

Usage:
    python extension3/generate_paraphrases.py \\
        --corpus clres \\
        --rare-threshold 0.25 \\
        --model-name Vamsi/T5_Paraphrase_Paws \\
        --output results/clres_ext3/paraphrases/paraphrases.tsv

Arguments:
    --corpus         Dataset to use: clres (PDEP), semcor, ontonotes
    --rare-threshold Senses with r < threshold are considered rare (default: 0.25)
    --model-name     HuggingFace T5 model for paraphrasing (default: Vamsi/T5_Paraphrase_Paws)
    --output         Path to save the paraphrases TSV
    --seed           Random seed (default: 42)

Output TSV columns:
    instance_idx, lemma, sense_label, proportion_r,
    original_sentence, paraphrase, target_token, lemma_preserved
"""

import argparse
import os
import sys

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bssp.common.config import Config
from bssp.common.reading import read_dataset_cached
from bssp.clres.dataset_reader import ClresConlluReader, lemma_from_label as clres_lemma
from bssp.semcor.dataset_reader import SemcorReader, lemma_from_label as semcor_lemma

from extension3.augmentation import (
    compute_sense_frequencies,
    identify_rare_senses,
    generate_paraphrases_offline,
)


CORPUS_CONFIG = {
    "clres": {
        "reader": ClresConlluReader,
        "train_path": "data/pdep/pdep_train.conllu",
        "lemma_fn": clres_lemma,
    },
    "semcor": {
        "reader": SemcorReader,
        "train_path": None,
        "lemma_fn": semcor_lemma,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate paraphrases for rare-sense instances (offline)",
    )
    parser.add_argument("--corpus", required=True, choices=list(CORPUS_CONFIG.keys()))
    parser.add_argument("--rare-threshold", type=float, default=0.25)
    parser.add_argument("--model-name", default="Vamsi/T5_Paraphrase_Paws")
    parser.add_argument("--output", default=None, help="Output TSV path (auto-generated if not set)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/{args.corpus}_ext3/paraphrases/paraphrases.tsv"

    config = CORPUS_CONFIG[args.corpus]
    lemma_fn = config["lemma_fn"]

    # Load dataset with a dummy config (no embeddings needed here)
    dummy_cfg = Config(
        args.corpus,
        embedding_model="bert-base-cased",
        metric="cosine",
        top_n=50,
        query_n=1,
        bert_layers=[11],
    )

    print(f"[main] Loading {args.corpus} training data...")
    train_dataset = read_dataset_cached(
        dummy_cfg, config["reader"], "train", config["train_path"], with_embeddings=False,
    )

    # Compute sense frequencies
    sense_freq, lemma_freq, sense_proportion = compute_sense_frequencies(train_dataset, lemma_fn)
    rare_senses = identify_rare_senses(sense_proportion, threshold=args.rare_threshold)

    print(f"[main] Total senses: {len(sense_freq)}")
    print(f"[main] Rare senses (r < {args.rare_threshold}): {len(rare_senses)}")
    total_rare = sum(1 for inst in train_dataset if inst["label"].label in rare_senses)
    print(f"[main] Rare-sense instances: {total_rare}")

    # Generate paraphrases
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_df = generate_paraphrases_offline(
        dataset=train_dataset,
        rare_senses=rare_senses,
        sense_proportion=sense_proportion,
        lemma_from_label_fn=lemma_fn,
        output_path=args.output,
        model_name=args.model_name,
        device=device,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"PARAPHRASE GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Corpus: {args.corpus}")
    print(f"Rare threshold: {args.rare_threshold}")
    print(f"Valid paraphrases: {len(valid_df)}")
    print(f"Output: {args.output}")
    print(f"\nSample paraphrases:")
    for _, row in valid_df.head(5).iterrows():
        print(f"  [{row['lemma']}] sense={row['sense_label']} r={row['proportion_r']}")
        print(f"    Original:   {row['original_sentence'][:80]}...")
        print(f"    Paraphrase: {row['paraphrase'][:80]}...")
        print()


if __name__ == "__main__":
    main()
