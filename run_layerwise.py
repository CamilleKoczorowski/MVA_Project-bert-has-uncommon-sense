#!/usr/bin/env python
"""
Extension 1 — Layer-wise MAP Analysis
======================================
Evaluates MAP_bucket at every layer ℓ ∈ {0, …, L-1} for each selected model,
optionally with inoculation fine-tuning at different instance counts.

Usage examples
--------------
# 1) Quick validation: BERT on PDEP, no fine-tuning, all layers
python run_layerwise.py --corpus clres --models bert-base-cased

# 2) Three models, no fine-tuning
python run_layerwise.py --corpus clres --models bert-base-cased roberta-base distilbert-base-cased

# 3) With fine-tuning levels
python run_layerwise.py --corpus clres --models bert-base-cased roberta-base --ft-levels 0 100 250 500

# 4) Generate plots and tables from existing results
python run_layerwise.py --corpus clres --models bert-base-cased roberta-base distilbert-base-cased --plot-only

Results go to:  results/<corpus>_layerwise/<model>/layer_<L>/...
Summary TSVs:   results/<corpus>_layerwise/summary/
Figures:        results/<corpus>_layerwise/figures/
"""

import argparse
import os
import sys
import csv
import json
from copy import copy
from collections import defaultdict

import torch
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import from the existing codebase
# ---------------------------------------------------------------------------
from bssp.common.config import Config
from bssp.common import paths
from bssp.common.pickle import pickle_read
from bssp.common.reading import read_dataset_cached, make_indexer, make_embedder
from bssp.common.analysis import metrics_at_k, dataset_stats
from bssp.common.nearest_neighbor_models import (
    NearestNeighborRetriever,
    NearestNeighborPredictor,
    RandomRetriever,
)
from bssp.common.util import batch_queries, format_sentence

import bssp.clres.dataset_reader
import bssp.ontonotes.dataset_reader
import bssp.semcor.dataset_reader
import bssp.fews.dataset_reader
from bssp.clres.dataset_reader import ClresConlluReader, lemma_from_label as clres_lemma
from bssp.ontonotes.dataset_reader import OntonotesReader
from bssp.semcor.dataset_reader import SemcorReader
from bssp.fews.dataset_reader import FewsReader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Number of transformer layers per model (0-indexed: layers 0 .. NUM_LAYERS-1)
MODEL_NUM_LAYERS = {
    "bert-base-cased": 12,
    "bert-base-uncased": 12,
    "roberta-base": 12,
    "distilbert-base-cased": 6,
    "distilroberta-base": 6,
    "albert-base-v2": 12,
    "xlnet-base-cased": 12,
    "gpt2": 12,
}

# Buckets matching the paper's Table 1
FREQ_BUCKETS = [(5, 500), (500, int(1e9))]
RARITY_BUCKETS = [(0.0, 0.25), (0.25, 1.0)]

BUCKET_LABELS = {
    (5, 500, 0.0, 0.25): "ℓ<500, r<0.25",
    (5, 500, 0.25, 1.0): "ℓ<500, r≥0.25",
    (500, int(1e9), 0.0, 0.25): "ℓ≥500, r<0.25",
    (500, int(1e9), 0.25, 1.0): "ℓ≥500, r≥0.25",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_lemma_f(corpus_name):
    """Return the lemma extraction function for a given corpus."""
    if corpus_name == "clres":
        return bssp.clres.dataset_reader.lemma_from_label
    elif corpus_name == "ontonotes":
        return bssp.ontonotes.dataset_reader.lemma_from_label
    elif corpus_name == "semcor":
        return bssp.semcor.dataset_reader.lemma_from_label
    elif corpus_name == "fews":
        return bssp.fews.dataset_reader.lemma_from_label
    else:
        raise ValueError(f"Unknown corpus: {corpus_name}")


def get_reader_and_paths(corpus_name):
    """Return (reader_class, train_path, test_path) for a corpus."""
    if corpus_name == "clres":
        return ClresConlluReader, "data/pdep/pdep_train.conllu", "data/pdep/pdep_test.conllu"
    elif corpus_name == "semcor":
        return SemcorReader, None, None
    elif corpus_name == "ontonotes":
        return (
            OntonotesReader,
            "data/conll-formatted-ontonotes-5.0/data/train",
            "data/conll-formatted-ontonotes-5.0/data/test",
        )
    else:
        raise ValueError(f"Unknown corpus: {corpus_name}")


def results_dir(corpus_name):
    return f"results/{corpus_name}_layerwise"


def model_layer_dir(corpus_name, model_name, layer, ft_insts):
    ft_tag = f"ft{ft_insts}" if ft_insts > 0 else "base"
    return os.path.join(results_dir(corpus_name), model_name, ft_tag, f"layer_{layer}")


def summary_dir(corpus_name):
    return os.path.join(results_dir(corpus_name), "summary")


def figures_dir(corpus_name):
    return os.path.join(results_dir(corpus_name), "figures")


def weights_path(model_name, ft_insts):
    """Path to fine-tuned weights (matches the existing convention)."""
    return f"models/{model_name}_{ft_insts}.pt"


def finetune_if_needed(model_name, ft_insts):
    """Run fine-tuning if weights file does not exist. Reuses existing main.py logic."""
    wpath = weights_path(model_name, ft_insts)
    if os.path.isfile(wpath):
        print(f"[finetune] Weights already exist: {wpath}")
        return wpath

    os.makedirs("models", exist_ok=True)
    print(f"[finetune] Fine-tuning {model_name} on {ft_insts} STREUSLE instances...")

    from allennlp.data.data_loaders import SimpleDataLoader
    from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
    from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
    from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
    from allennlp.training import GradientDescentTrainer
    from allennlp.training.optimizers import HuggingfaceAdamWOptimizer
    from allennlp.data import Vocabulary
    from bssp.fine_tuning.models import StreusleFineTuningModel
    from bssp.fine_tuning.streusle import StreusleJsonReader

    num_each = ft_insts // 3
    indexer = PretrainedTransformerMismatchedIndexer(model_name)
    reader = StreusleJsonReader(
        tokenizer=None,
        token_indexers={"tokens": indexer},
        max_n=num_each,
        max_v=num_each,
        max_p=num_each,
    )
    instances = list(reader.read("data/streusle/train/streusle.ud_train.json"))
    required = (ft_insts // 3) * 3
    if len(instances) < required:
        raise RuntimeError(f"Requested {required} instances, got only {len(instances)}")

    vocab = Vocabulary.from_instances(instances)
    loader = SimpleDataLoader(instances, batch_size=8, vocab=vocab)

    token_embedder = PretrainedTransformerMismatchedEmbedder(model_name, train_parameters=True)
    embedder = BasicTextFieldEmbedder({"tokens": token_embedder})
    model = StreusleFineTuningModel(vocab, embedder)
    model.to("cpu")

    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = HuggingfaceAdamWOptimizer(parameters, lr=2e-5)
    trainer = GradientDescentTrainer(
        model=model,
        data_loader=loader,
        validation_data_loader=loader,
        num_epochs=40,
        patience=5,
        optimizer=optimizer,
        run_sanity_checks=False,
    )
    trainer.train()
    transformer_model = model.embedder._token_embedders["tokens"]._matched_embedder.transformer_model
    torch.save(transformer_model.state_dict(), wpath)
    print(f"[finetune] Saved weights to {wpath}")
    return wpath


# ---------------------------------------------------------------------------
# Core: run a single (model, layer, ft_insts) trial
# ---------------------------------------------------------------------------
def run_single_trial(corpus_name, model_name, layer, ft_insts, top_n=50, query_n=1):
    """
    Run CWE similarity ranking for one model at one layer, optionally with
    fine-tuned weights.  Returns the path to the predictions TSV.
    """
    override = weights_path(model_name, ft_insts) if ft_insts > 0 else None
    cfg = Config(
        corpus_name,
        embedding_model=model_name,
        override_weights_path=override,
        metric="cosine",
        top_n=top_n,
        query_n=query_n,
        bert_layers=[layer],
    )

    # Check if predictions already exist
    predictions_path = paths.predictions_tsv_path(cfg)
    if os.path.isfile(predictions_path):
        print(f"[trial] Predictions exist, skipping: {predictions_path}")
    else:
        _run_predict(cfg, corpus_name)

    return cfg, predictions_path


def _run_predict(cfg, corpus_name):
    """Predict step: read data, build model, rank neighbours, write TSV."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    reader_cls, train_path, test_path = get_reader_and_paths(corpus_name)
    train_dataset = read_dataset_cached(cfg, reader_cls, "train", train_path, with_embeddings=True)
    test_dataset = read_dataset_cached(cfg, reader_cls, "test", test_path, with_embeddings=False)

    # For ontonotes, merge dev+test
    if corpus_name == "ontonotes":
        dev_path = "data/conll-formatted-ontonotes-5.0/data/development"
        dev_dataset = read_dataset_cached(cfg, reader_cls, "dev", dev_path, with_embeddings=False)
        test_dataset = dev_dataset + test_dataset

    lemma_f = get_lemma_f(corpus_name)
    train_labels, _ = dataset_stats("train", train_dataset, f"{corpus_name}_stats", lemma_f)
    dataset_stats("test", test_dataset, f"{corpus_name}_stats", lemma_f)

    indexer = make_indexer(cfg)
    vocab, embedder = make_embedder(cfg)

    from allennlp.data import Vocabulary as AllenVocab
    label_vocab = AllenVocab.from_instances(train_dataset)
    label_vocab.extend_from_instances(test_dataset)
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
            target_dataset=train_dataset,
            distance_metric=cfg.metric,
            device=device,
            top_n=cfg.top_n,
            same_lemma=True,
        )
        .eval()
        .to(device)
    )
    dummy_reader = reader_cls(split="train", token_indexers={"tokens": indexer})
    predictor = NearestNeighborPredictor(model=model, dataset_reader=dummy_reader)

    instances = [i for i in test_dataset if train_labels[i["label"].label] >= 5]
    batches = batch_queries(instances, cfg.query_n)

    predictions_path = paths.predictions_tsv_path(cfg)
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)

    with open(predictions_path, "wt") as f, torch.no_grad():
        tsv_writer = csv.writer(f, delimiter="\t")
        header = ["sentence", "label", "lemma", "label_freq_in_train"]
        header += [f"label_{i+1}" for i in range(cfg.top_n)]
        header += [f"lemma_{i+1}" for i in range(cfg.top_n)]
        header += [f"sentence_{i+1}" for i in range(cfg.top_n)]
        header += [f"distance_{i+1}" for i in range(cfg.top_n)]
        tsv_writer.writerow(header)

        for batch in tqdm(batches, desc=f"Predicting {cfg.embedding_model} layer={cfg.bert_layers}"):
            ds = predictor.predict_batch_instance(batch)
            d = ds[0]
            sentences = [[t.text for t in i["text"].tokens] for i in batch]
            spans = [i["label_span"] for i in batch]
            sentences = [
                format_sentence(sentence, span.span_start, span.span_end)
                for sentence, span in zip(sentences, spans)
            ]
            label = batch[0]["label"].label
            lemma_f_local = get_lemma_f(corpus_name)
            lemma = lemma_f_local(label)
            label_freq_in_train = train_labels[label]

            row = [" || ".join(sentences), label, lemma, label_freq_in_train]
            results = d[f"top_{cfg.top_n}"]
            results += [None for _ in range(cfg.top_n - len(results))]

            labels, lemmas, sents, distances = [], [], [], []
            for result in results:
                if result is None:
                    distances.append(88888888)
                    labels.append("")
                    lemmas.append("")
                    sents.append("")
                else:
                    index, distance = result
                    distances.append(distance)
                    instance = train_dataset[index]
                    labels.append(instance["label"].label)
                    lemmas.append(lemma_f_local(labels[-1]))
                    span = instance["label_span"]
                    sents.append(
                        format_sentence(
                            [t.text for t in instance["text"].tokens],
                            span.span_start,
                            span.span_end,
                        )
                    )

            row += labels + lemmas + sents + distances
            tsv_writer.writerow(row)

    print(f"[trial] Wrote predictions to {predictions_path}")


# ---------------------------------------------------------------------------
# Summarize: compute MAP per bucket for one trial
# ---------------------------------------------------------------------------
def summarize_trial(cfg, corpus_name):
    """
    Compute MAP per bucket for a single (model, layer, ft) configuration.
    Returns a dict: {bucket_key: {"map": float, "count": int}}.
    """
    label_freqs, lemma_freqs = _read_freq_files(corpus_name)
    predictions_path = paths.predictions_tsv_path(cfg)
    df = pd.read_csv(predictions_path, sep="\t", on_bad_lines="skip")
    lemma_f = get_lemma_f(corpus_name)

    results = {}
    for min_freq, max_freq in FREQ_BUCKETS:
        for min_r, max_r in RARITY_BUCKETS:
            metrics_at_k(
                cfg, df, label_freqs, lemma_freqs, lemma_f,
                min_train_freq=min_freq, max_train_freq=max_freq,
                min_rarity=min_r, max_rarity=max_r,
            )
            # Read back the precision pickle to compute MAP
            prec_path = paths.bucketed_metric_at_k_path(
                cfg, min_freq, max_freq, min_r, max_r, "prec"
            )
            prec = pickle_read(prec_path)
            count_path = paths.bucketed_metric_at_k_path(
                cfg, min_freq, max_freq, min_r, max_r, "count"
            )
            count = pickle_read(count_path)

            if prec is not None:
                map_val = sum(v["label"] for v in prec.values()) / len(prec)
            else:
                map_val = None

            bucket_key = (min_freq, max_freq, min_r, max_r)
            results[bucket_key] = {"map": map_val, "count": count}

    return results


def _read_freq_files(corpus_name):
    readf = lambda f: {k: int(v) for k, v in map(lambda l: l.strip().split("\t"), f)}
    with open(paths.freq_tsv_path(f"{corpus_name}_stats", "train", "label"), "r") as f:
        label_freqs = readf(f)
    with open(paths.freq_tsv_path(f"{corpus_name}_stats", "train", "lemma"), "r") as f:
        lemma_freqs = readf(f)
    return label_freqs, lemma_freqs


# ---------------------------------------------------------------------------
# Aggregate all trials into a summary TSV
# ---------------------------------------------------------------------------
def aggregate_results(corpus_name, all_results):
    """
    all_results: list of dicts with keys
        model, layer, num_layers, relative_depth, ft_insts,
        and one key per bucket_label -> MAP value.
    Writes a summary TSV and returns the DataFrame.
    """
    sdir = summary_dir(corpus_name)
    os.makedirs(sdir, exist_ok=True)

    rows = []
    for r in all_results:
        row = {
            "model": r["model"],
            "layer": r["layer"],
            "num_layers": r["num_layers"],
            "relative_depth": r["relative_depth"],
            "ft_insts": r["ft_insts"],
        }
        for bkey, blabel in BUCKET_LABELS.items():
            bucket_data = r.get(bkey)
            if bucket_data and bucket_data["map"] is not None:
                row[blabel] = round(bucket_data["map"] * 100, 2)  # as percentage
                row[blabel + " (count)"] = bucket_data["count"]
            else:
                row[blabel] = None
                row[blabel + " (count)"] = 0
        rows.append(row)

    new_df = pd.DataFrame(rows)

    # Merge with existing summary if present (so parallel runs accumulate)
    tsv_path = os.path.join(sdir, "layerwise_summary.tsv")
    if os.path.isfile(tsv_path):
        existing_df = pd.read_csv(tsv_path, sep="\t")
        # Remove rows that will be replaced (same model + layer + ft_insts)
        merge_keys = ["model", "layer", "ft_insts"]
        new_keys = set(new_df[merge_keys].apply(tuple, axis=1))
        mask = ~existing_df[merge_keys].apply(tuple, axis=1).isin(new_keys)
        existing_df = existing_df[mask]
        df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"[summary] Merged {len(new_df)} new rows with {len(existing_df)} existing rows.")
    else:
        df = new_df

    # Sort for readability
    df = df.sort_values(["model", "ft_insts", "layer"]).reset_index(drop=True)
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"[summary] Wrote {tsv_path} ({len(df)} total rows)")

    # Also save as JSON for convenience
    json_path = os.path.join(sdir, "layerwise_summary.json")
    df.to_json(json_path, orient="records", indent=2)
    print(f"[summary] Wrote {json_path}")

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def generate_plots(corpus_name):
    """Generate seaborn figures from the summary TSV."""
    from layerwise_plots import plot_all
    plot_all(corpus_name)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extension 1 — Layer-wise MAP Analysis for CWE Similarity Ranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--corpus",
        required=True,
        choices=["clres", "ontonotes", "semcor", "fews"],
        help="Corpus to evaluate on (clres = PDEP).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bert-base-cased"],
        choices=list(MODEL_NUM_LAYERS.keys()),
        help="Model(s) to evaluate. Default: bert-base-cased.",
    )
    parser.add_argument(
        "--ft-levels",
        nargs="+",
        type=int,
        default=[0],
        help="Fine-tuning instance counts. 0 = base model (no FT). Default: [0].",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of nearest neighbours to retrieve. Default: 50.",
    )
    parser.add_argument(
        "--query-n",
        type=int,
        default=1,
        help="Number of query sentences to average-pool. Default: 1.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip trials, only regenerate plots from existing summary TSV.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Run trials but skip plot generation.",
    )

    args = parser.parse_args()

    # ---- Plot-only mode ----
    if args.plot_only:
        print("[mode] Plot-only: regenerating figures from existing summary.")
        generate_plots(args.corpus)
        return

    # ---- Run trials ----
    all_results = []

    for model_name in args.models:
        num_layers = MODEL_NUM_LAYERS[model_name]
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}  ({num_layers} layers)")
        print(f"{'='*60}")

        for ft_insts in args.ft_levels:
            # Fine-tune if needed
            if ft_insts > 0:
                finetune_if_needed(model_name, ft_insts)

            for layer in range(num_layers):
                relative_depth = layer / (num_layers - 1) if num_layers > 1 else 0.0
                ft_tag = f"ft={ft_insts}" if ft_insts > 0 else "base"
                print(f"\n--- {model_name} | {ft_tag} | layer {layer}/{num_layers-1} (d={relative_depth:.3f}) ---")

                cfg, _ = run_single_trial(
                    args.corpus, model_name, layer, ft_insts,
                    top_n=args.top_n, query_n=args.query_n,
                )
                bucket_results = summarize_trial(cfg, args.corpus)

                all_results.append({
                    "model": model_name,
                    "layer": layer,
                    "num_layers": num_layers,
                    "relative_depth": round(relative_depth, 4),
                    "ft_insts": ft_insts,
                    **{bkey: bdata for bkey, bdata in bucket_results.items()},
                })

    # ---- Aggregate ----
    df = aggregate_results(args.corpus, all_results)
    print(f"\n[done] Summary table ({len(df)} rows):")
    print(df.to_string(index=False))

    # ---- Plots ----
    if not args.no_plot:
        generate_plots(args.corpus)


if __name__ == "__main__":
    main()
