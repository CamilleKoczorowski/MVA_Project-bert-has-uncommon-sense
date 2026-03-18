#!/usr/bin/env python
"""
extension3/evaluation.py — Evaluation pipeline for Extension 3
===============================================================
Re-uses the original paper's similarity ranking evaluation (MAP@50),
with additional seen/unseen breakdown for generalization analysis.

Key addition: after contrastive fine-tuning on D_ft (500 instances),
we evaluate on D_eval = D \ D_ft, splitting queries into:
  - Q_seen:   queries whose (lemma, sense) appeared in D_ft
  - Q_unseen: queries whose (lemma, sense) did NOT appear in D_ft
"""

import os
import json
import csv
from copy import copy
from collections import defaultdict

import torch
import pandas as pd
from tqdm import tqdm

from bssp.common import paths
from bssp.common.config import Config
from bssp.common.pickle import pickle_read
from bssp.common.reading import read_dataset_cached, make_indexer, make_embedder
from bssp.common.analysis import metrics_at_k, dataset_stats
from bssp.common.nearest_neighbor_models import (
    NearestNeighborRetriever,
    NearestNeighborPredictor,
    RandomRetriever,
)
from bssp.common.util import batch_queries, format_sentence


# ---------------------------------------------------------------------------
# Data splitting: D_ft / D_eval
# ---------------------------------------------------------------------------
def split_train_for_contrastive(
    dataset,
    num_ft_instances,
    lemma_from_label_fn,
    rare_threshold=0.25,
    seed=42,
):
    """
    Split the training dataset into:
      - D_ft:   instances for contrastive fine-tuning (stratified by sense)
      - D_eval: remaining instances for evaluation database

    Stratified sampling ensures rare senses are represented in D_ft.

    Returns:
        ft_indices: list of int (indices into dataset for fine-tuning)
        eval_indices: list of int (indices for evaluation database)
        ft_senses: set of (lemma, sense_label) tuples seen in D_ft
    """
    import random
    random.seed(seed)

    from collections import Counter
    sense_freq = Counter()
    lemma_freq = Counter()
    for inst in dataset:
        label = inst["label"].label
        lemma = lemma_from_label_fn(label)
        sense_freq[label] += 1
        lemma_freq[lemma] += 1

    sense_proportion = {}
    for label, count in sense_freq.items():
        lemma = lemma_from_label_fn(label)
        sense_proportion[label] = count / lemma_freq[lemma]

    # Group indices by sense label
    sense_to_indices = defaultdict(list)
    for i, inst in enumerate(dataset):
        sense_to_indices[inst["label"].label].append(i)

    # Prioritize rare senses in sampling
    rare_senses = {s for s, r in sense_proportion.items() if r < rare_threshold}
    common_senses = set(sense_proportion.keys()) - rare_senses

    ft_indices = []
    remaining_budget = num_ft_instances

    # First, sample from rare senses (at least 2 per sense if possible)
    rare_list = sorted(rare_senses)
    random.shuffle(rare_list)
    for sense in rare_list:
        if remaining_budget <= 0:
            break
        indices = sense_to_indices[sense]
        n_sample = min(2, len(indices), remaining_budget)
        sampled = random.sample(indices, n_sample)
        ft_indices.extend(sampled)
        remaining_budget -= n_sample

    # Fill remaining budget from common senses
    common_list = sorted(common_senses)
    random.shuffle(common_list)
    for sense in common_list:
        if remaining_budget <= 0:
            break
        indices = sense_to_indices[sense]
        n_sample = min(3, len(indices), remaining_budget)
        sampled = random.sample(indices, n_sample)
        ft_indices.extend(sampled)
        remaining_budget -= n_sample

    ft_set = set(ft_indices)
    eval_indices = [i for i in range(len(dataset)) if i not in ft_set]

    # Record which (lemma, sense) pairs are in D_ft
    ft_senses = set()
    for i in ft_indices:
        label = dataset[i]["label"].label
        lemma = lemma_from_label_fn(label)
        ft_senses.add((lemma, label))

    print(f"[split] D_ft: {len(ft_indices)} instances, D_eval: {len(eval_indices)} instances")
    print(f"[split] Unique (lemma, sense) in D_ft: {len(ft_senses)}")
    print(f"[split] Rare senses in D_ft: {len([s for s in ft_senses if sense_proportion.get(s[1], 1) < rare_threshold])}")

    return ft_indices, eval_indices, ft_senses


# ---------------------------------------------------------------------------
# Evaluation with seen/unseen breakdown
# ---------------------------------------------------------------------------
def evaluate_with_breakdown(
    cfg,
    corpus_name,
    train_dataset,
    test_dataset,
    eval_indices,
    ft_senses,
    lemma_from_label_fn,
    results_dir,
):
    """
    Run the standard similarity ranking evaluation on D_eval,
    with additional breakdown into Q_seen and Q_unseen.

    Args:
        cfg: Config for the model
        corpus_name: 'clres', 'semcor', etc.
        train_dataset: full training dataset
        test_dataset: test/dev dataset (Q)
        eval_indices: indices of D_eval within train_dataset
        ft_senses: set of (lemma, sense) seen during fine-tuning
        lemma_from_label_fn: function to extract lemma from label
        results_dir: where to save results

    Returns:
        results: dict with MAP scores per bucket, globally and by seen/unseen
    """
    # Build D_eval subset
    eval_dataset = [train_dataset[i] for i in eval_indices]

    # Classify queries as seen/unseen
    q_seen_indices = []
    q_unseen_indices = []
    for i, inst in enumerate(test_dataset):
        label = inst["label"].label
        lemma = lemma_from_label_fn(label)
        if (lemma, label) in ft_senses:
            q_seen_indices.append(i)
        else:
            q_unseen_indices.append(i)

    print(f"[eval] Q total: {len(test_dataset)}, Q_seen: {len(q_seen_indices)}, Q_unseen: {len(q_unseen_indices)}")

    # Save the split info
    split_info = {
        "num_eval_database": len(eval_dataset),
        "num_queries_total": len(test_dataset),
        "num_queries_seen": len(q_seen_indices),
        "num_queries_unseen": len(q_unseen_indices),
        "ft_senses": [list(s) for s in ft_senses],
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    # Run evaluation on the full Q (standard comparison)
    # This uses the existing infrastructure from the paper
    results = {
        "global": _run_ranking_and_metrics(cfg, eval_dataset, test_dataset, lemma_from_label_fn, corpus_name),
    }

    # Run on Q_seen only
    if q_seen_indices:
        q_seen = [test_dataset[i] for i in q_seen_indices]
        results["seen"] = _run_ranking_and_metrics(cfg, eval_dataset, q_seen, lemma_from_label_fn, corpus_name)

    # Run on Q_unseen only
    if q_unseen_indices:
        q_unseen = [test_dataset[i] for i in q_unseen_indices]
        results["unseen"] = _run_ranking_and_metrics(cfg, eval_dataset, q_unseen, lemma_from_label_fn, corpus_name)

    # Save results
    with open(os.path.join(results_dir, "map_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def _run_ranking_and_metrics(cfg, database, queries, lemma_fn, corpus_name):
    """
    Run nearest-neighbor ranking and compute MAP@50 per bucket.
    Returns dict of {bucket_label: MAP_score}.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    indexer = make_indexer(cfg)
    vocab, embedder = make_embedder(cfg)

    from allennlp.data import Vocabulary
    label_vocab = Vocabulary.from_instances(database)
    label_vocab.extend_from_instances(queries)
    for ns in ["tokens"]:
        try:
            del label_vocab._token_to_index[ns]
        except KeyError:
            pass
        try:
            del label_vocab._index_to_token[ns]
        except KeyError:
            pass
    vocab.extend_from_vocab(label_vocab)

    model = (
        NearestNeighborRetriever(
            vocab=vocab,
            embedder=embedder,
            target_dataset=database,
            distance_metric=cfg.metric,
            device=device,
            top_n=cfg.top_n,
            same_lemma=True,
        )
        .eval()
        .to(device)
    )

    # Get label frequencies from database
    from collections import Counter
    label_freqs = Counter(inst["label"].label for inst in database)
    lemma_freqs = Counter(lemma_fn(inst["label"].label) for inst in database)

    # Filter queries to those with at least 5 occurrences in database
    valid_queries = [q for q in queries if label_freqs[q["label"].label] >= 5]

    if not valid_queries:
        return {"note": "No valid queries (all senses have < 5 instances in D_eval)"}

    # Run predictions
    from bssp.clres.dataset_reader import ClresConlluReader
    dummy_reader = ClresConlluReader(split="train", token_indexers={"tokens": indexer})
    predictor = NearestNeighborPredictor(model=model, dataset_reader=dummy_reader)

    batches = batch_queries(valid_queries, cfg.query_n)

    all_precisions = []  # list of average_precision per query

    freq_buckets = [(5, 500), (500, int(1e9))]
    rarity_buckets = [(0.0, 0.25), (0.25, 1.0)]
    bucket_precisions = {
        (mf, xf, mr, xr): []
        for mf, xf in freq_buckets
        for mr, xr in rarity_buckets
    }

    with torch.no_grad():
        for batch in tqdm(batches, desc="Evaluating"):
            ds = predictor.predict_batch_instance(batch)
            d = ds[0]

            label = batch[0]["label"].label
            lemma = lemma_fn(label)
            label_freq = label_freqs[label]
            lemma_total = lemma_freqs[lemma]
            r = label_freq / lemma_total if lemma_total > 0 else 0

            results = d[f"top_{cfg.top_n}"]
            results += [None for _ in range(cfg.top_n - len(results))]

            # Compute precision at each k
            hits = 0
            precisions_at_k = []
            for k, result in enumerate(results, 1):
                if result is not None:
                    idx, dist = result
                    if database[idx]["label"].label == label:
                        hits += 1
                precisions_at_k.append(hits / k)

            avg_precision = sum(precisions_at_k) / len(precisions_at_k) if precisions_at_k else 0
            all_precisions.append(avg_precision)

            # Assign to bucket
            for mf, xf in freq_buckets:
                for mr, xr in rarity_buckets:
                    if mf <= lemma_total < xf and mr <= r < xr:
                        bucket_precisions[(mf, xf, mr, xr)].append(avg_precision)

    bucket_labels = {
        (5, 500, 0.0, 0.25): "ℓ<500, r<0.25",
        (5, 500, 0.25, 1.0): "ℓ<500, r≥0.25",
        (500, int(1e9), 0.0, 0.25): "ℓ≥500, r<0.25",
        (500, int(1e9), 0.25, 1.0): "ℓ≥500, r≥0.25",
    }

    result = {
        "MAP@50_global": round(sum(all_precisions) / max(len(all_precisions), 1) * 100, 2),
        "num_queries": len(all_precisions),
    }
    for bkey, blabel in bucket_labels.items():
        precs = bucket_precisions[bkey]
        if precs:
            result[blabel] = round(sum(precs) / len(precs) * 100, 2)
            result[blabel + " (count)"] = len(precs)
        else:
            result[blabel] = None
            result[blabel + " (count)"] = 0

    return result
