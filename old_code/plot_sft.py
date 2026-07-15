"""
plot_sft.py — local plotting script for SFT training runs.

Mirror of plot_verl.py. Reads the eval_log.jsonl written by train_sft_ddp.py
(one JSON object per eval, keyed by global `step`) and saves:
    sft_train_metrics.png   — loss + train accuracy vs training step
    sft_eval_cladder.png    — CLaDDer accuracy breakdown vs step
    sft_eval_causci.png     — CauSciBench metric breakdown vs step

Usage:
    python3 plot_sft.py [eval_log.jsonl]
"""

import json
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else "eval_log.jsonl"
df = pd.DataFrame(json.loads(l) for l in open(LOG_FILE) if l.strip())


def _plot(cols, suptitle, out, ylim=None):
    """cols: list of (column, label) — skips columns absent from the log."""
    valid = [(c, l) for c, l in cols if c in df.columns and df[c].notna().any()]
    if df.empty or not valid:
        return
    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]
    for ax, (col, lbl) in zip(axes, valid):
        sub = df[["step", col]].dropna()
        ax.plot(sub["step"], sub[col], marker="o", markersize=3, linewidth=1.5)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("Step")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── 1. Training metrics ──────────────────────────────────────────────────────
_plot([
    ("train/loss",              "Train loss"),
    ("train/cladder_acc",       "CLaDDer train acc"),
    ("train/causci_method_acc", "CauSci method train acc"),
], "Training Metrics", "sft_train_metrics.png")

# ── 2. CLaDDer eval ──────────────────────────────────────────────────────────
_plot([
    ("cladder/overall_acc",             "Overall acc"),
    ("cladder/rung1_acc",               "Rung 1 acc"),
    ("cladder/rung2_acc",               "Rung 2 acc"),
    ("cladder/rung3_acc",               "Rung 3 acc"),
    ("cladder/step2_acc",               "Query-type acc"),
    ("cladder/commonsensical_acc",      "Commonsensical"),
    ("cladder/nonsensical_acc",         "Nonsensical"),
    ("cladder/anti_commonsensical_acc", "Anti-commonsensical"),
], "CLaDDer Eval Metrics", "sft_eval_cladder.png", ylim=(0, 1))

# ── 3. CauSciBench eval ──────────────────────────────────────────────────────
_plot([
    ("causci/method_acc",    "Method sel. acc"),
    ("causci/treatment_acc", "Treatment var acc"),
    ("causci/outcome_acc",   "Outcome var acc"),
    ("causci/control_acc",   "Control var acc"),
    ("causci/effect_acc",    "Effect acc"),
    ("causci/mre",           "MRE"),
], "CauSciBench Eval Metrics", "sft_eval_causci.png")
