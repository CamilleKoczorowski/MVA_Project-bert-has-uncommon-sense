#!/usr/bin/env python
"""
extension3/evaluation.py — Evaluation pipeline for Extension 3
===============================================================
Re-uses the original paper's similarity ranking evaluation (MAP@50)
by calling main.py trial + summarize as subprocesses. This avoids
the span_embeddings issue since the paper's pipeline correctly handles
embedding computation with override weights.

Additionally computes a seen/unseen breakdown from the predictions TSV.
"""

import os
import json
import subprocess
import sys
from collections import defaultdict, Counter

import pandas as pd


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

    Returns:
        ft_indices, eval_indices, ft_senses
    """
    import random
    random.seed(seed)

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

    sense_to_indices = defaultdict(list)
    for i, inst in enumerate(dataset):
        sense_to_indices[inst["label"].label].append(i)

    rare_senses = {s for s, r in sense_proportion.items() if r < rare_threshold}
    common_senses = set(sense_proportion.keys()) - rare_senses

    ft_indices = []
    remaining_budget = num_ft_instances

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
# Evaluation using the original paper's pipeline (subprocess)
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
    Run evaluation using the original paper's main.py trial + summarize,
    then compute seen/unseen breakdown from the predictions TSV.
    """
    os.makedirs(results_dir, exist_ok=True)

    weights_path = cfg.override_weights_path
    model_name = cfg.embedding_model
    last_layer = cfg.bert_layers[0] if cfg.bert_layers else 11

    # --- Step 1: Run the paper's trial command ---
    print("[eval] Running paper's evaluation pipeline with fine-tuned weights...")

    cmd_trial = [
        sys.executable, "main.py", "trial",
        "--embedding-model", model_name,
        "--metric", "cosine",
        "--query-n", "1",
        "--bert-layer", str(last_layer),
        corpus_name,
    ]
    if weights_path and os.path.isfile(weights_path):
        cmd_trial.extend(["--override-weights", weights_path])

    print(f"[eval] Command: {' '.join(cmd_trial)}")
    result = subprocess.run(cmd_trial, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Trial command failed with return code {result.returncode}")

    # --- Step 2: Run summarize ---
    cmd_summarize = [
        sys.executable, "main.py", "summarize",
        "--embedding-model", model_name,
        "--metric", "cosine",
        "--query-n", "1",
        "--bert-layer", str(last_layer),
        corpus_name,
    ]
    if weights_path and os.path.isfile(weights_path):
        cmd_summarize.extend(["--override-weights", weights_path])

    print(f"[eval] Command: {' '.join(cmd_summarize)}")
    result_sum = subprocess.run(cmd_summarize, capture_output=False)

    # --- Step 3: Find and read predictions TSV ---
    from bssp.common import paths as bssp_paths
    predictions_path = bssp_paths.predictions_tsv_path(cfg)

    if not os.path.isfile(predictions_path):
        # Try to find it in cache
        print(f"[eval] Predictions not at {predictions_path}, searching cache/...")
        found = False
        for dirpath, dirnames, filenames in os.walk("cache"):
            for fn in filenames:
                if fn.endswith(".tsv") and model_name.replace("-", "") in fn.replace("-", ""):
                    predictions_path = os.path.join(dirpath, fn)
                    print(f"[eval] Found: {predictions_path}")
                    found = True
                    break
            if found:
                break

        if not found:
            print("[eval] ERROR: could not find predictions TSV")
            return {"error": "predictions file not found"}

    print(f"[eval] Reading predictions from {predictions_path}")
    df = pd.read_csv(predictions_path, sep="\t", on_bad_lines="skip")

    # --- Step 4: Compute MAP with seen/unseen split ---
    seen_mask = []
    unseen_mask = []
    for _, row in df.iterrows():
        label = str(row.get("label", ""))
        lemma = str(row.get("lemma", ""))
        if (lemma, label) in ft_senses:
            seen_mask.append(True)
            unseen_mask.append(False)
        else:
            seen_mask.append(False)
            unseen_mask.append(True)

    df["is_seen"] = seen_mask
    df["is_unseen"] = unseen_mask

    n_seen = sum(seen_mask)
    n_unseen = sum(unseen_mask)
    print(f"[eval] Q total: {len(df)}, Q_seen: {n_seen}, Q_unseen: {n_unseen}")

    results = {}
    for subset_name, mask_col in [("global", None), ("seen", "is_seen"), ("unseen", "is_unseen")]:
        if mask_col is not None:
            sub_df = df[df[mask_col]].copy()
        else:
            sub_df = df.copy()

        if sub_df.empty:
            results[subset_name] = {"note": f"No {subset_name} queries"}
            continue

        results[subset_name] = _compute_map_from_predictions(sub_df, cfg.top_n)

    # Save
    split_info = {
        "num_queries_total": len(df),
        "num_queries_seen": n_seen,
        "num_queries_unseen": n_unseen,
        "ft_senses_count": len(ft_senses),
    }
    with open(os.path.join(results_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    with open(os.path.join(results_dir, "map_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"[eval] Results saved to {results_dir}/map_results.json")
    return results


def _compute_map_from_predictions(df, top_n=50):
    """
    Compute MAP@50 per bucket from a predictions DataFrame.
    Columns expected: label, lemma, label_freq_in_train, label_1..label_50.
    """
    freq_buckets = [(5, 500), (500, int(1e9))]
    rarity_buckets = [(0.0, 0.25), (0.25, 1.0)]

    bucket_labels = {
        (5, 500, 0.0, 0.25): "ℓ<500, r<0.25",
        (5, 500, 0.25, 1.0): "ℓ<500, r≥0.25",
        (500, int(1e9), 0.0, 0.25): "ℓ≥500, r<0.25",
        (500, int(1e9), 0.25, 1.0): "ℓ≥500, r≥0.25",
    }

    # Load lemma and label frequencies from stats cache
    lemma_freqs = {}
    label_freqs = {}

    for stats_dir_name in ["clres_stats", "semcor_stats", "ontonotes_stats"]:
        lemma_path = f"cache/{stats_dir_name}/train_lemma_freq.tsv"
        label_path = f"cache/{stats_dir_name}/train_label_freq.tsv"
        if os.path.isfile(lemma_path):
            with open(lemma_path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        lemma_freqs[parts[0]] = int(parts[1])
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        label_freqs[parts[0]] = int(parts[1])
            break

    bucket_precisions = {bkey: [] for bkey in bucket_labels}
    all_precisions = []

    for _, row in df.iterrows():
        label = str(row["label"])
        lemma = str(row["lemma"])
        freq_label = label_freqs.get(label, int(row.get("label_freq_in_train", 0)))
        freq_lemma = lemma_freqs.get(lemma, 0)

        if freq_lemma == 0:
            continue

        r = freq_label / freq_lemma

        # Average precision for this query
        hits = 0
        precisions_at_k = []
        for k in range(1, top_n + 1):
            col = f"label_{k}"
            if col in df.columns and pd.notna(row.get(col)):
                if str(row[col]) == label:
                    hits += 1
            precisions_at_k.append(hits / k)

        avg_precision = sum(precisions_at_k) / len(precisions_at_k) if precisions_at_k else 0.0
        all_precisions.append(avg_precision)

        for bkey in bucket_labels:
            min_f, max_f, min_r, max_r = bkey
            if min_f <= freq_lemma < max_f and min_r <= r < max_r:
                bucket_precisions[bkey].append(avg_precision)

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
