"""plot.py — CauSci synth→{real,qr} SFT plots for one run (SFT_OUT_DIR selects which). Emits two:
  progression.png       — per-metric curves vs training step (from output/sft/metrics.csv)
  bars_before_after.png — base vs final bars per metric (from output/sft/phase_metrics.jsonl)
Metrics: method, treatment, outcome, control (recall), control (F1). Run locally (cluster lacks matplotlib).
"""

import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(os.environ.get("SFT_OUT_DIR") or Path(__file__).resolve().parent / "output")
PLOTS = Path(os.environ.get("SFT_PLOTS_DIR") or (OUT / "plots"))   # override to a shared plots dir
TAG = os.environ.get("SFT_TAG", "")                                # filename suffix to distinguish runs


def _rows():
    f = OUT / "sft" / "metrics.csv"
    if not f.exists():
        print("no output/sft/metrics.csv — nothing to plot")
        return []
    return [{k: float(v) if v not in ("", None) else 0.0 for k, v in r.items()}
            for r in csv.DictReader(open(f))]


# (title, per-split key suffix, combined/overall column)
METRICS = [("method", "method", "method_correctness"), ("treatment", "treatment", "treatment_acc"),
           ("outcome", "outcome", "outcome_acc"), ("control (recall)", "control_recall", "control_recall"),
           ("control (precision)", "control_precision", "control_precision"),
           ("control (acc)", "control_acc", "control_acc"), ("control (F1)", "conf_f1", "confounder_f1")]


def plot_synth2rest():
    rows = _rows()
    if not rows:
        return
    steps = [r["step"] for r in rows]
    # only panels whose metric is actually logged (older runs have recall but not per-step F1)
    present = [m for m in METRICS if any(m[2] in r or f"real/{m[1]}" in r or f"qr/{m[1]}" in r for r in rows)]
    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 5))
    if len(present) == 1:
        axes = [axes]
    for ax, (title, key, comb) in zip(axes, present):
        ax.plot(steps, [r.get(comb, 0.0) for r in rows], "-^", color="black", label="combined")
        for sp, sty in (("real", "-o"), ("qr", "-s")):
            ax.plot(steps, [r.get(f"{sp}/{key}", 0.0) for r in rows], sty, label=sp)
        ax.set_title(title); ax.set_xlabel("step"); ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("CauSci synth→{real,qr} SFT — checkpoint progression (per split)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / f"progression{TAG}.png", dpi=140, bbox_inches="tight")
    print(f"Saved → {PLOTS / 'progression.png'}")


def plot_bars():
    """Base (pre-SFT) vs final (post-SFT) bars per metric, from this run's phase_metrics.jsonl."""
    f = OUT / "sft" / "phase_metrics.jsonl"
    if not f.exists():
        print("no output/sft/phase_metrics.jsonl — skip bars")
        return
    d = {}
    for l in open(f):
        m = json.loads(l); d[m["phase"]] = m
    present = [t for t in METRICS if any(t[2] in d.get(p, {}) for p in ("base", "final"))]  # skip un-graded metrics
    fig, axes = plt.subplots(1, len(present), figsize=(3.6 * len(present), 4.5))
    if len(present) == 1:
        axes = [axes]
    for ax, (title, _, comb) in zip(axes, present):
        vals = [d.get("base", {}).get(comb, 0.0), d.get("final", {}).get(comb, 0.0)]
        b = ax.bar(["base", "final"], vals, color=["#1f77b4", "#ff7f0e"])
        ax.bar_label(b, fmt="%.2f", fontsize=9)
        ax.set_ylim(0, 1); ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("CauSci synth→{real,qr} SFT — before vs after (combined real+qr)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / f"bars_before_after{TAG}.png", dpi=140, bbox_inches="tight")
    print(f"Saved → {PLOTS / 'bars_before_after.png'}")


if __name__ == "__main__":
    plot_synth2rest()
    plot_bars()
