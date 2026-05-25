"""
plot_verl.py — local plotting script for veRL training runs.

Usage:
    python3 plot_verl.py [verl_metrics.csv]

Reads verl_metrics.csv (produced by parse_verl_logs.py on HPC) and saves:
    verl_train_metrics.png   — reward, loss, KL vs training step
    verl_eval_cladder.png    — CLaDDer accuracy breakdown vs eval pass
    verl_eval_causci.png     — CauSciBench metric breakdown vs eval pass
"""

import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

CSV_FILE = sys.argv[1] if len(sys.argv) > 1 else "verl_metrics.csv"
df = pd.read_csv(CSV_FILE)

train_df = df[df["step"].notna()].copy()
eval_df  = df[df["eval_pass"].notna()].copy()

# ── helpers ────────────────────────────────────────────────────────────────

def _plot_grid(ax_data: list[tuple], x_col: str, xlabel: str, title: str, out: str) -> None:
    """
    ax_data: list of (ax, col, label) — skips columns absent from the dataframe.
    """
    valid = [(ax, col, lbl) for ax, col, lbl in ax_data if col in eval_df.columns or col in train_df.columns]
    if not valid:
        return
    n   = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]
    for ax, (_, col, lbl) in zip(axes, valid):
        src = train_df if x_col == "step" else eval_df
        if col not in src.columns:
            ax.set_visible(False)
            continue
        sub = src[[x_col, col]].dropna()
        ax.plot(sub[x_col], sub[col], marker="o", markersize=3, linewidth=1.5, label=lbl)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── 1. Training metrics ────────────────────────────────────────────────────

TRAIN_COLS = [
    ("reward/mean",  "Reward / mean"),
    ("actor/loss",   "Actor loss"),
    ("critic/loss",  "Critic loss"),
    ("kl",           "KL divergence"),
]

if not train_df.empty:
    valid = [(col, lbl) for col, lbl in TRAIN_COLS if col in train_df.columns]
    if valid:
        n = len(valid)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        axes = axes[0]
        for ax, (col, lbl) in zip(axes, valid):
            sub = train_df[["step", col]].dropna()
            ax.plot(sub["step"], sub[col], linewidth=1.5)
            ax.set_title(lbl, fontsize=10)
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)
        fig.suptitle("Training Metrics", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig("verl_train_metrics.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved → verl_train_metrics.png")

# ── 2. CLaDDer eval ────────────────────────────────────────────────────────

CLADDER_COLS = [
    ("cladder/overall_acc",             "Overall acc"),
    ("cladder/rung1_acc",               "Rung 1 acc"),
    ("cladder/rung2_acc",               "Rung 2 acc"),
    ("cladder/rung3_acc",               "Rung 3 acc"),
    ("cladder/commonsensical_acc",      "Commonsensical"),
    ("cladder/nonsensical_acc",         "Nonsensical"),
    ("cladder/anti_commonsensical_acc", "Anti-commonsensical"),
]

if not eval_df.empty:
    valid = [(col, lbl) for col, lbl in CLADDER_COLS if col in eval_df.columns]
    if valid:
        n = len(valid)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        axes = axes[0]
        for ax, (col, lbl) in zip(axes, valid):
            sub = eval_df[["eval_pass", col]].dropna()
            ax.plot(sub["eval_pass"], sub[col], marker="o", markersize=3, linewidth=1.5)
            ax.set_title(lbl, fontsize=10)
            ax.set_xlabel("Eval pass")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        fig.suptitle("CLaDDer Eval Metrics", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig("verl_eval_cladder.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved → verl_eval_cladder.png")

# ── 3. CauSciBench eval ────────────────────────────────────────────────────

CAUSCI_COLS = [
    ("causci/method_acc",    "Method sel. acc"),
    ("causci/treatment_acc", "Treatment var acc"),
    ("causci/outcome_acc",   "Outcome var acc"),
    ("causci/control_acc",   "Control var acc"),
    ("causci/effect_acc",    "Effect acc"),
    ("causci/mre",           "MRE"),
]

if not eval_df.empty:
    valid = [(col, lbl) for col, lbl in CAUSCI_COLS if col in eval_df.columns]
    if valid:
        n = len(valid)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        axes = axes[0]
        for ax, (col, lbl) in zip(axes, valid):
            sub = eval_df[["eval_pass", col]].dropna()
            ax.plot(sub["eval_pass"], sub[col], marker="o", markersize=3, linewidth=1.5)
            ax.set_title(lbl, fontsize=10)
            ax.set_xlabel("Eval pass")
            ax.grid(True, alpha=0.3)
        fig.suptitle("CauSciBench Eval Metrics", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig("verl_eval_causci.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved → verl_eval_causci.png")
