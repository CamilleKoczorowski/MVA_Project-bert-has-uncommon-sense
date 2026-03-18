#!/usr/bin/env python
"""
extension3/augmentation.py — Augmentation strategies for rare word senses
=========================================================================
Three strategies targeting senses with proportional frequency r < threshold:
  1. Random oversampling:  duplicate existing instances
  2. Paraphrase (T5):     generate meaning-preserving paraphrases (offline)
  3. Embedding SMOTE:      interpolate between same-sense embeddings

Paraphrases are generated offline and saved to TSV for inspection.
SMOTE operates in embedding space during training.
"""

import os
import json
import random
from collections import defaultdict, Counter

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Sense frequency analysis
# ---------------------------------------------------------------------------
def compute_sense_frequencies(dataset, lemma_from_label_fn):
    """
    Compute proportional frequency r for each sense label.
    Returns:
        sense_freq: dict {label: count}
        lemma_freq: dict {lemma: count}
        sense_proportion: dict {label: r} where r = count(label) / count(lemma)
    """
    sense_freq = Counter()
    lemma_freq = Counter()

    for instance in dataset:
        label = instance["label"].label
        lemma = lemma_from_label_fn(label)
        sense_freq[label] += 1
        lemma_freq[lemma] += 1

    sense_proportion = {}
    for label, count in sense_freq.items():
        lemma = lemma_from_label_fn(label)
        sense_proportion[label] = count / lemma_freq[lemma]

    return sense_freq, lemma_freq, sense_proportion


def identify_rare_senses(sense_proportion, threshold=0.25):
    """Return the set of sense labels with proportional frequency r < threshold."""
    return {label for label, r in sense_proportion.items() if r < threshold}


def get_rare_sense_instances(dataset, rare_senses):
    """Return indices of instances whose sense is in rare_senses."""
    return [i for i, inst in enumerate(dataset) if inst["label"].label in rare_senses]


# ---------------------------------------------------------------------------
# Strategy 1: Random oversampling
# ---------------------------------------------------------------------------
def oversample_indices(dataset, rare_senses, sense_proportion, lemma_from_label_fn):
    """
    Compute oversampling factors for rare senses.
    More rare senses get higher duplication factors.

    Returns:
        List of (index, duplication_count) for rare-sense instances.
    """
    results = []
    for i, inst in enumerate(dataset):
        label = inst["label"].label
        if label in rare_senses:
            r = sense_proportion[label]
            # More rare -> more copies. r < 0.10 -> factor 5, 0.10 <= r < 0.25 -> factor 3
            if r < 0.10:
                factor = 5
            else:
                factor = 3
            results.append((i, factor))
    return results


# ---------------------------------------------------------------------------
# Strategy 2: Paraphrase generation (offline with T5)
# ---------------------------------------------------------------------------
class ParaphraseGenerator:
    """
    Generate meaning-preserving paraphrases using T5.
    The target lemma must be preserved in the output.
    """

    def __init__(self, model_name="Vamsi/T5_Paraphrase_Paws", device=None, max_length=128):
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[paraphrase] Loading {model_name} on {self.device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length

    def generate(self, sentence, target_lemma, num_paraphrases=3, num_beams=10, num_return=6):
        """
        Generate paraphrases of `sentence` that preserve `target_lemma`.
        Returns up to `num_paraphrases` valid paraphrases.
        """
        input_text = f"paraphrase: {sentence} </s>"
        encoding = self.tokenizer.encode_plus(
            input_text, return_tensors="pt", max_length=self.max_length, truncation=True
        )
        input_ids = encoding["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=self.max_length,
                num_beams=num_beams,
                num_return_sequences=num_return,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        candidates = []
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            # Verify target lemma is preserved (case-insensitive)
            if target_lemma.lower() in decoded.lower() and decoded.strip() != sentence.strip():
                candidates.append(decoded)

        return candidates[:num_paraphrases]


def generate_paraphrases_offline(
    dataset,
    rare_senses,
    sense_proportion,
    lemma_from_label_fn,
    output_path,
    model_name="Vamsi/T5_Paraphrase_Paws",
    device=None,
):
    """
    Generate paraphrases for all rare-sense instances and save to TSV.
    Adaptive: r < 0.10 -> 4 paraphrases, 0.10 <= r < 0.25 -> 2 paraphrases.

    Columns: lemma, sense_label, original_sentence, paraphrase, target_span, lemma_preserved
    """
    generator = ParaphraseGenerator(model_name=model_name, device=device)
    rows = []

    for i, inst in enumerate(tqdm(dataset, desc="Generating paraphrases")):
        label = inst["label"].label
        if label not in rare_senses:
            continue

        r = sense_proportion[label]
        num_para = 4 if r < 0.10 else 2

        lemma = lemma_from_label_fn(label)
        tokens = [t.text for t in inst["text"].tokens]
        sentence = " ".join(tokens)

        # Get the target token
        span = inst["label_span"]
        target_token = tokens[span.span_start]

        paraphrases = generator.generate(sentence, lemma, num_paraphrases=num_para)

        for para in paraphrases:
            lemma_preserved = lemma.lower() in para.lower()
            rows.append({
                "instance_idx": i,
                "lemma": lemma,
                "sense_label": label,
                "proportion_r": round(r, 4),
                "original_sentence": sentence,
                "paraphrase": para,
                "target_token": target_token,
                "lemma_preserved": lemma_preserved,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"[paraphrase] Saved {len(df)} paraphrases to {output_path}")
    print(f"[paraphrase] Lemma preservation rate: {df['lemma_preserved'].mean():.1%}")

    # Filter to only valid paraphrases
    valid = df[df["lemma_preserved"]].copy()
    print(f"[paraphrase] Valid paraphrases: {len(valid)} / {len(df)}")
    return valid


def load_paraphrases(path):
    """Load pre-generated paraphrases from TSV."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Paraphrases not found: {path}. Run generate_paraphrases.py first.")
    df = pd.read_csv(path, sep="\t")
    # Only keep valid paraphrases
    return df[df["lemma_preserved"] == True].copy()


# ---------------------------------------------------------------------------
# Strategy 3: Embedding-space SMOTE
# ---------------------------------------------------------------------------
def smote_embeddings(embeddings_by_sense, num_synthetic_per_sense=None, alpha_range=(0.0, 1.0)):
    """
    Generate synthetic embeddings by interpolating between same-sense pairs.

    Args:
        embeddings_by_sense: dict {sense_label: tensor of shape (N, D)}
        num_synthetic_per_sense: dict {sense_label: int} or None (auto)
        alpha_range: range for interpolation coefficient

    Returns:
        synthetic: dict {sense_label: tensor of shape (M, D)}
        parents: dict {sense_label: list of (idx_i, idx_j, alpha)}
    """
    synthetic = {}
    parents = {}

    for sense, embeds in embeddings_by_sense.items():
        n = embeds.shape[0]
        if n < 2:
            continue

        if num_synthetic_per_sense and sense in num_synthetic_per_sense:
            num_syn = num_synthetic_per_sense[sense]
        else:
            # Default: generate enough to roughly double the sense count
            num_syn = max(n, 5)

        syn_list = []
        parent_list = []

        for _ in range(num_syn):
            i, j = random.sample(range(n), 2)
            alpha = random.uniform(*alpha_range)
            h_new = alpha * embeds[i] + (1 - alpha) * embeds[j]
            syn_list.append(h_new)
            parent_list.append((i, j, round(alpha, 4)))

        synthetic[sense] = torch.stack(syn_list)
        parents[sense] = parent_list

    return synthetic, parents


# ---------------------------------------------------------------------------
# Unified augmentation interface
# ---------------------------------------------------------------------------
def apply_augmentation(
    strategy,
    dataset,
    rare_senses,
    sense_proportion,
    lemma_from_label_fn,
    paraphrases_path=None,
    embeddings_by_sense=None,
):
    """
    Apply the chosen augmentation strategy.

    Returns a dict with strategy-specific data:
      - 'none': empty dict
      - 'oversample': {'indices': [(idx, factor), ...]}
      - 'paraphrase': {'paraphrases_df': DataFrame}
      - 'smote': {'synthetic': {sense: tensor}, 'parents': {sense: [...]}}
      - 'paraphrase+smote': both paraphrase and smote data
    """
    result = {"strategy": strategy}

    if strategy == "none":
        return result

    elif strategy == "oversample":
        indices = oversample_indices(dataset, rare_senses, sense_proportion, lemma_from_label_fn)
        result["indices"] = indices
        return result

    elif strategy == "paraphrase":
        if paraphrases_path is None:
            raise ValueError("paraphrases_path required for paraphrase strategy")
        df = load_paraphrases(paraphrases_path)
        result["paraphrases_df"] = df
        return result

    elif strategy == "smote":
        if embeddings_by_sense is None:
            raise ValueError("embeddings_by_sense required for smote strategy")
        synthetic, parents = smote_embeddings(embeddings_by_sense)
        result["synthetic"] = synthetic
        result["parents"] = parents
        return result

    elif strategy == "paraphrase+smote":
        if paraphrases_path is None:
            raise ValueError("paraphrases_path required for paraphrase+smote strategy")
        if embeddings_by_sense is None:
            raise ValueError("embeddings_by_sense required for paraphrase+smote strategy")
        df = load_paraphrases(paraphrases_path)
        synthetic, parents = smote_embeddings(embeddings_by_sense)
        result["paraphrases_df"] = df
        result["synthetic"] = synthetic
        result["parents"] = parents
        return result

    else:
        raise ValueError(f"Unknown augmentation strategy: {strategy}")
