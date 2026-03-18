#!/usr/bin/env python
"""
extension3/visualize_results.py — Visualization for Extension 3
================================================================
Generates:
  1. Comparison table (LaTeX + terminal) of MAP@50 across all configurations
  2. Seen/unseen breakdown chart
  3. SMOTE scatter plots (before/after, per lemma)
  4. Paraphrase quality summary

Usage:
    python extension3/visualize_results.py --corpus clres --results-dir results/clres_ext3
"""

import argparse
import os
import json
import glob

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def setup_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })


# ---------------------------------------------------------------------------
# 1. Comparison table across all experiments
# ---------------------------------------------------------------------------
def build_comparison_table(results_dir):
    """
    Scan all experiment subdirectories and build a comparison DataFrame.
    Each experiment has a map_results.json file.
    """
    rows = []
    for exp_dir in sorted(glob.glob(os.path.join(results_dir, "*", "*"))):
        json_path = os.path.join(exp_dir, "map_results.json")
        if not os.path.isfile(json_path):
            continue

        with open(json_path) as f:
            data = json.load(f)

        # Parse experiment name from directory structure
        parts = exp_dir.replace(results_dir, "").strip("/").split("/")
        if len(parts) >= 2:
            model_name = parts[0]
            config_name = parts[1]
        else:
            model_name = parts[0] if parts else "unknown"
            config_name = "unknown"

        # Global results
        if "global" in data:
            row = {
                "Model": model_name,
                "Config": config_name,
                "Subset": "All",
            }
            row.update({k: v for k, v in data["global"].items() if not k.endswith("(count)")})
            rows.append(row)

        # Seen results
        if "seen" in data and isinstance(data["seen"], dict):
            row = {
                "Model": model_name,
                "Config": config_name,
                "Subset": "Seen",
            }
            row.update({k: v for k, v in data["seen"].items() if not k.endswith("(count)")})
            rows.append(row)

        # Unseen results
        if "unseen" in data and isinstance(data["unseen"], dict):
            row = {
                "Model": model_name,
                "Config": config_name,
                "Subset": "Unseen",
            }
            row.update({k: v for k, v in data["unseen"].items() if not k.endswith("(count)")})
            rows.append(row)

    if not rows:
        print("[viz] No results found.")
        return None

    df = pd.DataFrame(rows)
    return df


def print_comparison_table(df, results_dir):
    """Print and save the comparison table."""
    if df is None or df.empty:
        return

    # Terminal display
    print("\n" + "=" * 80)
    print("EXTENSION 3 — RESULTS COMPARISON")
    print("=" * 80)

    # Show "All" subset first
    all_df = df[df["Subset"] == "All"].drop(columns=["Subset"])
    print("\n--- Global (all queries) ---")
    print(all_df.to_string(index=False))

    # Show seen/unseen breakdown
    for subset in ["Seen", "Unseen"]:
        sub_df = df[df["Subset"] == subset].drop(columns=["Subset"])
        if not sub_df.empty:
            print(f"\n--- {subset} queries only ---")
            print(sub_df.to_string(index=False))

    # Save CSV
    out_path = os.path.join(results_dir, "comparison_table.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[viz] Saved {out_path}")


def generate_latex_table(df, results_dir):
    """Generate LaTeX table for the paper."""
    if df is None or df.empty:
        return

    all_df = df[df["Subset"] == "All"].drop(columns=["Subset"]).copy()
    bucket_cols = [c for c in all_df.columns if c.startswith("ℓ")]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Extension 3 — MAP@50 comparison across fine-tuning and augmentation strategies}")
    lines.append(r"\label{tab:ext3_results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{ll" + "c" * len(bucket_cols) + r"}")
    lines.append(r"\toprule")

    header = r"Model & Config"
    for col in bucket_cols:
        header += " & " + col.replace("≥", r"$\geq$").replace("<", r"$<$").replace("ℓ", r"$\ell$")
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for _, row in all_df.iterrows():
        line = f"{row['Model']} & {row['Config']}"
        for col in bucket_cols:
            val = row.get(col)
            if val is not None and not pd.isna(val):
                line += f" & {val:.1f}"
            else:
                line += r" & --"
        line += r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    out_path = os.path.join(results_dir, "ext3_table.tex")
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"[viz] Saved {out_path}")


# ---------------------------------------------------------------------------
# 2. Seen vs Unseen bar chart
# ---------------------------------------------------------------------------
def plot_seen_unseen(df, results_dir):
    """Bar chart comparing MAP on seen vs unseen queries."""
    setup_style()

    if df is None:
        return

    # Focus on rare senses bucket
    target_col = "ℓ<500, r<0.25"
    if target_col not in df.columns:
        target_col = "ℓ≥500, r<0.25"
    if target_col not in df.columns:
        print("[viz] No rare-sense bucket found, skipping seen/unseen plot.")
        return

    plot_df = df[df["Subset"].isin(["Seen", "Unseen"])].copy()
    if plot_df.empty:
        return

    plot_df["label"] = plot_df["Model"] + "\n" + plot_df["Config"]

    fig, ax = plt.subplots(figsize=(12, 6))
    configs = plot_df["label"].unique()
    x = np.arange(len(configs))
    width = 0.35

    seen_vals = []
    unseen_vals = []
    for cfg in configs:
        seen_row = plot_df[(plot_df["label"] == cfg) & (plot_df["Subset"] == "Seen")]
        unseen_row = plot_df[(plot_df["label"] == cfg) & (plot_df["Subset"] == "Unseen")]
        seen_vals.append(seen_row[target_col].values[0] if not seen_row.empty and pd.notna(seen_row[target_col].values[0]) else 0)
        unseen_vals.append(unseen_row[target_col].values[0] if not unseen_row.empty and pd.notna(unseen_row[target_col].values[0]) else 0)

    ax.bar(x - width / 2, seen_vals, width, label="Q_seen", color="#2176AE", alpha=0.8)
    ax.bar(x + width / 2, unseen_vals, width, label="Q_unseen", color="#E04040", alpha=0.8)

    ax.set_ylabel("MAP@50 (%)")
    ax.set_title(f"Generalization: Seen vs Unseen queries — {target_col}")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=8, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()

    out = os.path.join(results_dir, "seen_unseen_comparison.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"[viz] Saved {out}")


# ---------------------------------------------------------------------------
# 3. SMOTE scatter plots
# ---------------------------------------------------------------------------
def plot_smote_scatter(embeddings_before, embeddings_after, labels, synthetic_mask, lemma, results_dir):
    """
    2D scatter plot (PCA or t-SNE) showing original and synthetic embeddings.

    Args:
        embeddings_before: (N, D) original embeddings
        embeddings_after: (N+M, D) original + synthetic embeddings
        labels: sense labels for all points
        synthetic_mask: boolean array, True for synthetic points
        lemma: lemma string for the title
        results_dir: where to save
    """
    setup_style()
    from sklearn.decomposition import PCA

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Before (original only)
    pca = PCA(n_components=2)
    coords_before = pca.fit_transform(embeddings_before)

    unique_labels = sorted(set(labels[:len(embeddings_before)]))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {l: c for l, c in zip(unique_labels, colors)}

    for lbl in unique_labels:
        mask = np.array(labels[:len(embeddings_before)]) == lbl
        ax1.scatter(coords_before[mask, 0], coords_before[mask, 1],
                    c=[label_to_color[lbl]], label=lbl, s=30, alpha=0.7)
    ax1.set_title(f"Before SMOTE — '{lemma}'")
    ax1.legend(fontsize=7, loc="best")

    # After (original + synthetic)
    coords_after = pca.transform(embeddings_after)
    original_mask = ~synthetic_mask

    for lbl in unique_labels:
        mask_orig = np.array(labels) == lbl
        mask_orig = mask_orig & original_mask
        mask_syn = np.array(labels) == lbl
        mask_syn = mask_syn & synthetic_mask

        if mask_orig.any():
            ax2.scatter(coords_after[mask_orig, 0], coords_after[mask_orig, 1],
                        c=[label_to_color[lbl]], marker="o", s=30, alpha=0.7, label=f"{lbl} (orig)")
        if mask_syn.any():
            ax2.scatter(coords_after[mask_syn, 0], coords_after[mask_syn, 1],
                        c=[label_to_color[lbl]], marker="*", s=80, alpha=0.9, label=f"{lbl} (SMOTE)")

    ax2.set_title(f"After SMOTE — '{lemma}'")
    ax2.legend(fontsize=7, loc="best")

    fig.suptitle(f"Embedding-space SMOTE for lemma '{lemma}'", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out = os.path.join(results_dir, f"smote_scatter_{lemma}.pdf")
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"[viz] Saved {out}")


# ---------------------------------------------------------------------------
# 4. Paraphrase quality summary
# ---------------------------------------------------------------------------
def summarize_paraphrases(paraphrases_path, results_dir):
    """Print and save a summary of paraphrase quality."""
    if not os.path.isfile(paraphrases_path):
        print(f"[viz] Paraphrases file not found: {paraphrases_path}")
        return

    df = pd.read_csv(paraphrases_path, sep="\t")
    print(f"\n{'='*60}")
    print("PARAPHRASE QUALITY SUMMARY")
    print(f"{'='*60}")
    print(f"Total paraphrases generated: {len(df)}")
    print(f"Lemma preserved: {df['lemma_preserved'].sum()} ({df['lemma_preserved'].mean():.1%})")
    print(f"Unique senses covered: {df['sense_label'].nunique()}")
    print(f"Unique lemmas covered: {df['lemma'].nunique()}")

    # Per-rarity breakdown
    print(f"\nBy rarity:")
    for threshold in [0.10, 0.25]:
        sub = df[df["proportion_r"] < threshold]
        print(f"  r < {threshold}: {len(sub)} paraphrases, "
              f"{sub['lemma_preserved'].mean():.1%} lemma preserved")

    # Save summary
    summary = {
        "total": len(df),
        "lemma_preserved": int(df["lemma_preserved"].sum()),
        "preservation_rate": round(df["lemma_preserved"].mean(), 4),
        "unique_senses": df["sense_label"].nunique(),
        "unique_lemmas": df["lemma"].nunique(),
    }
    out = os.path.join(results_dir, "paraphrase_summary.json")
    os.makedirs(results_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[viz] Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize Extension 3 results")
    parser.add_argument("--corpus", required=True, choices=["clres", "semcor", "ontonotes"])
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = f"results/{args.corpus}_ext3"

    df = build_comparison_table(args.results_dir)
    print_comparison_table(df, args.results_dir)
    generate_latex_table(df, args.results_dir)
    plot_seen_unseen(df, args.results_dir)

    # Paraphrase summary
    para_path = os.path.join(args.results_dir, "paraphrases", "paraphrases.tsv")
    summarize_paraphrases(para_path, args.results_dir)


if __name__ == "__main__":
    main()
