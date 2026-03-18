#!/usr/bin/env python
"""
layerwise_plots.py — Seaborn figures and LaTeX tables for Extension 1
=====================================================================
Called by run_layerwise.py after trials complete, or standalone:
    python layerwise_plots.py --corpus clres
"""

import os
import argparse

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def _setup_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


# Distinct, color-blind-friendly palette
MODEL_PALETTE = {
    "bert-base-cased": "#2176AE",
    "roberta-base": "#E04040",
    "distilbert-base-cased": "#57B8FF",
    "distilroberta-base": "#FF8C69",
    "albert-base-v2": "#6B4C9A",
    "xlnet-base-cased": "#2E8B57",
    "gpt2": "#DAA520",
}

MODEL_MARKERS = {
    "bert-base-cased": "o",
    "roberta-base": "s",
    "distilbert-base-cased": "D",
    "distilroberta-base": "^",
    "albert-base-v2": "P",
    "xlnet-base-cased": "X",
    "gpt2": "v",
}

# Short display names
MODEL_DISPLAY = {
    "bert-base-cased": "BERT",
    "roberta-base": "RoBERTa",
    "distilbert-base-cased": "DistilBERT",
    "distilroberta-base": "DistilRoBERTa",
    "albert-base-v2": "ALBERT",
    "xlnet-base-cased": "XLNet",
    "gpt2": "GPT-2",
}

BUCKET_COLS = [
    "ℓ<500, r<0.25",
    "ℓ<500, r≥0.25",
    "ℓ≥500, r<0.25",
    "ℓ≥500, r≥0.25",
]

BUCKET_TITLES = {
    "ℓ<500, r<0.25": "Rare lemmas, rare senses\n(ℓ < 500, r < 0.25)",
    "ℓ<500, r≥0.25": "Rare lemmas, common senses\n(ℓ < 500, r ≥ 0.25)",
    "ℓ≥500, r<0.25": "Frequent lemmas, rare senses\n(ℓ ≥ 500, r < 0.25)",
    "ℓ≥500, r≥0.25": "Frequent lemmas, common senses\n(ℓ ≥ 500, r ≥ 0.25)",
}

CORPUS_DISPLAY = {
    "clres": "PDEP",
    "ontonotes": "OntoNotes",
    "semcor": "SemCor",
}


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
def _figures_dir(corpus):
    d = f"results/{corpus}_layerwise/figures"
    os.makedirs(d, exist_ok=True)
    return d


def _summary_path(corpus):
    return f"results/{corpus}_layerwise/summary/layerwise_summary.tsv"


def _load_summary(corpus):
    path = _summary_path(corpus)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Summary TSV not found: {path}\nRun trials first with run_layerwise.py")
    return pd.read_csv(path, sep="\t")


# ---------------------------------------------------------------------------
# Figure 1: MAP vs relative depth — one subplot per bucket, lines per model
# (base models only, ft_insts == 0)
# ---------------------------------------------------------------------------
def plot_map_vs_depth(df, corpus):
    _setup_style()
    base = df[df["ft_insts"] == 0].copy()
    if base.empty:
        print("[plot] No base-model results to plot.")
        return

    models = base["model"].unique()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=False)
    corpus_label = CORPUS_DISPLAY.get(corpus, corpus)

    for ax, col in zip(axes, BUCKET_COLS):
        for model in models:
            mdf = base[base["model"] == model].sort_values("relative_depth")
            if col not in mdf.columns or mdf[col].isna().all():
                continue
            ax.plot(
                mdf["relative_depth"],
                mdf[col],
                color=MODEL_PALETTE.get(model, "#333"),
                marker=MODEL_MARKERS.get(model, "o"),
                markersize=5,
                linewidth=1.8,
                label=MODEL_DISPLAY.get(model, model),
                alpha=0.9,
            )
            # Mark the optimum with a star
            best_idx = mdf[col].idxmax()
            if pd.notna(best_idx):
                best_row = mdf.loc[best_idx]
                ax.plot(
                    best_row["relative_depth"],
                    best_row[col],
                    marker="*",
                    markersize=14,
                    color=MODEL_PALETTE.get(model, "#333"),
                    zorder=5,
                )

        ax.set_title(BUCKET_TITLES[col], fontsize=10)
        ax.set_xlabel("Relative depth  d = ℓ / (L−1)")
        ax.set_xlim(-0.03, 1.03)

    axes[0].set_ylabel("MAP@50  (%)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    fig.suptitle(
        f"Layer-wise MAP — {corpus_label} (base models, no fine-tuning)",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()

    out = os.path.join(_figures_dir(corpus), "map_vs_depth_base.pdf")
    fig.savefig(out)
    out_png = out.replace(".pdf", ".png")
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[plot] Saved {out} and {out_png}")


# ---------------------------------------------------------------------------
# Figure 2: MAP vs relative depth — rare senses only (r < 0.25), focus plot
# ---------------------------------------------------------------------------
def plot_rare_senses_focus(df, corpus):
    _setup_style()
    base = df[df["ft_insts"] == 0].copy()
    if base.empty:
        return

    models = base["model"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    corpus_label = CORPUS_DISPLAY.get(corpus, corpus)
    rare_cols = ["ℓ<500, r<0.25", "ℓ≥500, r<0.25"]
    titles = [
        "Rare lemmas × Rare senses\n(ℓ < 500, r < 0.25)",
        "Frequent lemmas × Rare senses\n(ℓ ≥ 500, r < 0.25)",
    ]

    for ax, col, title in zip(axes, rare_cols, titles):
        for model in models:
            mdf = base[base["model"] == model].sort_values("relative_depth")
            if col not in mdf.columns or mdf[col].isna().all():
                continue
            ax.plot(
                mdf["relative_depth"],
                mdf[col],
                color=MODEL_PALETTE.get(model, "#333"),
                marker=MODEL_MARKERS.get(model, "o"),
                markersize=6,
                linewidth=2,
                label=MODEL_DISPLAY.get(model, model),
            )
            # Annotate optimum
            best_idx = mdf[col].idxmax()
            if pd.notna(best_idx):
                best = mdf.loc[best_idx]
                ax.annotate(
                    f"d={best['relative_depth']:.2f}",
                    xy=(best["relative_depth"], best[col]),
                    xytext=(5, 8),
                    textcoords="offset points",
                    fontsize=7,
                    color=MODEL_PALETTE.get(model, "#333"),
                    fontweight="bold",
                )

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Relative depth  d = ℓ / (L−1)")
        ax.set_ylabel("MAP@50  (%)")
        ax.set_xlim(-0.03, 1.03)

    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.suptitle(
        f"Rare senses layer profile — {corpus_label}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    out = os.path.join(_figures_dir(corpus), "rare_senses_focus.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"[plot] Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Optimal layer heatmap (model × bucket)
# ---------------------------------------------------------------------------
def plot_optimal_layer_heatmap(df, corpus):
    _setup_style()
    base = df[df["ft_insts"] == 0].copy()
    if base.empty:
        return

    models = sorted(base["model"].unique(), key=lambda m: list(MODEL_DISPLAY.keys()).index(m) if m in MODEL_DISPLAY else 99)
    data_depth = []
    data_map = []

    for model in models:
        mdf = base[base["model"] == model]
        row_depth = []
        row_map = []
        for col in BUCKET_COLS:
            if col in mdf.columns and not mdf[col].isna().all():
                best_idx = mdf[col].idxmax()
                best = mdf.loc[best_idx]
                row_depth.append(best["relative_depth"])
                row_map.append(best[col])
            else:
                row_depth.append(np.nan)
                row_map.append(np.nan)
        data_depth.append(row_depth)
        data_map.append(row_map)

    labels = [MODEL_DISPLAY.get(m, m) for m in models]
    short_buckets = ["ℓ<500\nr<0.25", "ℓ<500\nr≥0.25", "ℓ≥500\nr<0.25", "ℓ≥500\nr≥0.25"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(3, 0.7 * len(models))))

    # Left: optimal relative depth
    depth_df = pd.DataFrame(data_depth, index=labels, columns=short_buckets)
    sns.heatmap(
        depth_df, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax1,
        vmin=0, vmax=1, linewidths=0.5,
        cbar_kws={"label": "Optimal relative depth d*"},
    )
    ax1.set_title("Optimal layer (relative depth d*)", fontsize=11)
    ax1.set_xlabel("")

    # Right: MAP at optimal layer
    map_df = pd.DataFrame(data_map, index=labels, columns=short_buckets)
    sns.heatmap(
        map_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax2,
        linewidths=0.5,
        cbar_kws={"label": "MAP@50 (%)"},
    )
    ax2.set_title("MAP@50 at optimal layer", fontsize=11)
    ax2.set_xlabel("")

    corpus_label = CORPUS_DISPLAY.get(corpus, corpus)
    fig.suptitle(
        f"Optimal layer analysis — {corpus_label}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(_figures_dir(corpus), "optimal_layer_heatmap.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"[plot] Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4 (optional): effect of fine-tuning across layers
# ---------------------------------------------------------------------------
def plot_ft_effect(df, corpus):
    _setup_style()
    ft_data = df[df["ft_insts"] > 0]
    if ft_data.empty:
        print("[plot] No fine-tuning results; skipping FT effect plot.")
        return

    col = "ℓ<500, r<0.25"  # focus on rare senses
    if col not in df.columns:
        return

    models = df["model"].unique()
    for model in models:
        mdf = df[df["model"] == model].copy()
        ft_levels = sorted(mdf["ft_insts"].unique())
        if len(ft_levels) <= 1:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for ft in ft_levels:
            sub = mdf[mdf["ft_insts"] == ft].sort_values("relative_depth")
            label = "base" if ft == 0 else f"FT {ft}"
            lw = 2.5 if ft == 0 else 1.5
            ls = "-" if ft == 0 else "--"
            ax.plot(
                sub["relative_depth"], sub[col],
                linewidth=lw, linestyle=ls, marker="o", markersize=4,
                label=label, alpha=0.85,
            )

        ax.set_xlabel("Relative depth  d = ℓ / (L−1)")
        ax.set_ylabel("MAP@50  (%)")
        ax.set_title(
            f"{MODEL_DISPLAY.get(model, model)} — Rare senses (ℓ<500, r<0.25)\n"
            f"Effect of inoculation fine-tuning",
            fontsize=11,
        )
        ax.legend(title="# FT instances")
        ax.set_xlim(-0.03, 1.03)
        fig.tight_layout()

        out = os.path.join(_figures_dir(corpus), f"ft_effect_{model}.pdf")
        fig.savefig(out)
        fig.savefig(out.replace(".pdf", ".png"))
        plt.close(fig)
        print(f"[plot] Saved {out}")


# ---------------------------------------------------------------------------
# LaTeX table: optimal layer per model × bucket
# ---------------------------------------------------------------------------
def generate_latex_table(df, corpus):
    base = df[df["ft_insts"] == 0].copy()
    if base.empty:
        return

    models = sorted(base["model"].unique(), key=lambda m: list(MODEL_DISPLAY.keys()).index(m) if m in MODEL_DISPLAY else 99)
    corpus_label = CORPUS_DISPLAY.get(corpus, corpus)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Optimal layer $\ell^*$ and MAP@50 per bucket — " + corpus_label + r"}")
    lines.append(r"\label{tab:layerwise_" + corpus + r"}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "cc" * 4 + r"}")
    lines.append(r"\toprule")

    # Header
    header1 = r"\multirow{2}{*}{Model}"
    for col in BUCKET_COLS:
        header1 += r" & \multicolumn{2}{c}{" + col.replace("≥", r"$\geq$").replace("<", r"$<$").replace("ℓ", r"$\ell$") + "}"
    header1 += r" \\"
    lines.append(header1)
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    sub_header = " " * 20
    for _ in BUCKET_COLS:
        sub_header += r" & $d^*$ & MAP"
    sub_header += r" \\"
    lines.append(sub_header)
    lines.append(r"\midrule")

    for model in models:
        mdf = base[base["model"] == model]
        display = MODEL_DISPLAY.get(model, model)
        row = display
        for col in BUCKET_COLS:
            if col in mdf.columns and not mdf[col].isna().all():
                best_idx = mdf[col].idxmax()
                best = mdf.loc[best_idx]
                row += f" & {best['relative_depth']:.2f} & {best[col]:.1f}"
            else:
                row += r" & -- & --"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    out = os.path.join(_figures_dir(corpus), "layerwise_table.tex")
    with open(out, "w") as f:
        f.write(tex)
    print(f"[latex] Saved {out}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def plot_all(corpus):
    df = _load_summary(corpus)
    plot_map_vs_depth(df, corpus)
    plot_rare_senses_focus(df, corpus)
    plot_optimal_layer_heatmap(df, corpus)
    plot_ft_effect(df, corpus)
    generate_latex_table(df, corpus)
    print(f"\n[plots] All figures saved in {_figures_dir(corpus)}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for layer-wise analysis")
    parser.add_argument("--corpus", required=True, choices=["clres", "ontonotes", "semcor"])
    args = parser.parse_args()
    plot_all(args.corpus)
