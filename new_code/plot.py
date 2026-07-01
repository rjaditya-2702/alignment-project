"""Plot the CauSciBench test metrics across SFT then RL as one continuous timeline.

Reads two metric CSVs (eval_pass + causci/* columns) — one per phase — and draws one
figure: 6 subplots (one per metric), SFT passes then RL passes laid end to end with a
dashed divider at the SFT→RL handoff.

Run:  python plot.py   (after both phases have written their metrics CSVs)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

METRICS = [
    ("causci/method_acc",    "Method acc"),
    ("causci/treatment_acc", "Treatment acc"),
    ("causci/outcome_acc",   "Outcome acc"),
    ("causci/control_acc",   "Control acc"),
    ("causci/effect_acc",    "Effect acc"),
    ("causci/mre",           "MRE"),
]


def _load(path, start):
    """Read a phase's metrics CSV (or None if absent) and lay it on the shared step axis."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["t"] = range(start, start + len(df))
    return df


def main():
    # Plot whichever phases have run — SFT alone (decide if RL is worth it), RL alone, or both.
    sft = _load(config.SFT_METRICS_CSV, 0)
    rl  = _load(config.RL_METRICS_CSV, len(sft) if sft is not None else 0)
    if sft is None and rl is None:
        raise FileNotFoundError(f"No metrics found at {config.SFT_METRICS_CSV} or {config.RL_METRICS_CSV}")

    divider = len(sft) - 0.5 if (sft is not None and rl is not None) else None

    config.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, label) in zip(axes.flat, METRICS):
        for df, name, color in [(sft, "SFT", "tab:blue"), (rl, "RL", "tab:orange")]:
            if df is not None and col in df.columns:
                ax.plot(df["t"], df[col], marker="o", ms=3, lw=1.5, color=color, label=name)
        if divider is not None:
            ax.axvline(divider, ls="--", color="gray", alpha=0.7)
        ax.set_title(label)
        ax.set_xlabel("eval pass (SFT → RL)")
        if col != "causci/mre":
            ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("CauSciBench test metrics across SFT → RL", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = config.PLOT_DIR / "metrics_sft_rl.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
