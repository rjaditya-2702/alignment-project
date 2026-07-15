"""plot.py — dashboards from the RL log + CauSci metrics.

CLadder (output/plots/cladder_metrics.png), from the veRL log:
  row 0  training curves (reward, policy loss, KL, grad norm) vs step
  row 1  per-step diagnostic (graph_f1, query_acc, estimand_acc, calc_acc, answer_acc, full) vs eval pass
  row 2  per-RUNG breakdown (answer / estimand / graph / full for rung 1/2/3 + overall) + verifier/parse

CauSci (output/plots/causci_metrics.png), from output/causci_metrics.json (written by causci_eval.py):
  per-split bars: method_correctness / treatment / outcome / control_coverage / effect_acc

Run:  python plot.py [verl_training.log]
"""

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "output"
PLOTS = OUT / "plots"
SFT_DIR = OUT / "sft"
_NUM = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_KV = re.compile(rf"([\w/]+):\s*({_NUM})")
_STEP = re.compile(r"\bstep[:\s]+(\d+)", re.IGNORECASE)
_EVAL = re.compile(r"\[verl_eval\]\s+eval_pass:(\d+)\s+(.*)")
_CAUSCI = re.compile(r"\[causci_eval\]\s+eval_pass:(\d+)\s+(.*)")


def parse_log(log):
    train, ev, causci = {}, {}, {}
    for line in open(log, errors="ignore"):
        cm = _CAUSCI.search(line)
        if cm:
            kv = {k: float(v) for k, v in _KV.findall(cm.group(2)) if k != "eval_pass"}
            if kv:
                causci.setdefault(int(cm.group(1)), {}).update(kv)
            continue
        m = _EVAL.search(line)
        if m:
            kv = {k: float(v) for k, v in _KV.findall(m.group(2)) if k not in ("call", "eval_pass")}
            if kv:
                ev.setdefault(int(m.group(1)), {}).update(kv)
            continue
        sm = _STEP.search(line)
        if sm:
            kv = {k: float(v) for k, v in _KV.findall(line) if k.lower() != "step"}
            if kv:
                train.setdefault(int(sm.group(1)), {}).update(kv)
    return train, ev, causci


def _xy(d, key):
    """(xs, ys) for an exact metric key across sorted x."""
    xs = sorted(x for x in d if key in d[x])
    return xs, [d[x][key] for x in xs]


def _first(d, *cands):
    keys = {k for row in d.values() for k in row}
    return next((k for c in cands for k in keys if c.lower() in k.lower()), None)


def _panel(ax, title, series, xlabel, is01=False):
    """series: list of (label, key, dict). Plots each present key as a line."""
    plotted = False
    for label, key, d in series:
        xs, ys = _xy(d, key)
        if xs:
            ax.plot(xs, ys, marker="o", ms=2.5, lw=1.3, label=label)
            plotted = True
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    if is01:
        ax.set_ylim(0, 1)
    if plotted and len(series) > 1:
        ax.legend(fontsize=7)
    if not plotted:
        ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes, color="gray")


def plot_cladder(train, ev, causci=None):
    causci = causci or {}
    fig, ax = plt.subplots(3, 6, figsize=(28, 13))
    # row 0 — training
    for a, (title, cands) in zip(ax[0], [("Reward (mean)", ("critic/rewards/mean", "reward")),
                                         ("Policy loss", ("pg_loss", "actor/loss")),
                                         ("KL", ("kl_loss", "/kl")), ("Grad norm", ("grad_norm",))]):
        k = _first(train, *cands)
        _panel(a, title + (f"\n({k})" if k else ""), [("", k, train)] if k else [], "step")
    # CauSci transfer during RL (from [causci_eval] lines) — the headline transfer signal
    _panel(ax[0][4], "CauSci transfer", [("method", "method_correctness", causci), ("treatment", "treatment_acc", causci),
           ("outcome", "outcome_acc", causci), ("control", "control_coverage", causci)], "eval pass", is01=True)
    ax[0][5].axis("off")
    # row 1 — per-step diagnostic (overall)
    for a, (title, key) in zip(ax[1], [("Graph F1", "graph_f1"), ("Query acc", "query_acc"),
                                       ("Estimand acc", "estimand_acc"), ("Calc acc", "calc_acc"),
                                       ("Answer acc", "answer_acc"), ("Full correct", "full_correct")]):
        _panel(a, title, [("", key, ev)], "eval pass", is01=True)
    # row 2 — per-rung + verifier. STEPKEYS[step] = (overall key, per-rung suffix)
    STEPKEYS = {"answer": ("answer_acc", "answer_acc"), "estimand": ("estimand_acc", "estimand_acc"),
                "graph": ("graph_f1", "graph_f1"), "full": ("full_correct", "full")}
    def series(step):
        ov_key, suf = STEPKEYS[step]
        return [("overall", ov_key, ev)] + [(f"rung{r}", f"r{r}/{suf}", ev) for r in (1, 2, 3)]
    _panel(ax[2][0], "Answer acc by rung", series("answer"), "eval pass", is01=True)
    _panel(ax[2][1], "Estimand acc by rung", series("estimand"), "eval pass", is01=True)
    _panel(ax[2][2], "Graph F1 by rung", series("graph"), "eval pass", is01=True)
    _panel(ax[2][3], "Full correct by rung", series("full"), "eval pass", is01=True)
    _panel(ax[2][4], "Verifier / parse", [("verified", "verified_rate", ev),
           ("parse_fail", "parse_fail_rate", ev)], "eval pass", is01=True)
    ax[2][5].axis("off")

    fig.suptitle("CLadder RLVR — training, per-step diagnostic, per-rung", fontsize=14, fontweight="bold")
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "cladder_metrics.png", dpi=140, bbox_inches="tight")
    print(f"Saved → {PLOTS / 'cladder_metrics.png'}")


def plot_causci():
    f = OUT / "causci_metrics.json"
    if not f.exists():
        print("no output/causci_metrics.json — skip CauSci plot (run causci_eval.py first)")
        return
    rep = json.loads(f.read_text())
    splits = [s for s in ("synth", "real") if s in rep]
    metrics = ["method_correctness", "treatment_acc", "outcome_acc", "control_coverage"]
    if not splits:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.8 / len(splits)
    for i, s in enumerate(splits):
        vals = [rep[s].get(k, 0.0) for k in metrics]
        ax.bar([x + i * w for x in range(len(metrics))], vals, w, label=s)
    ax.set_xticks([x + w * (len(splits) - 1) / 2 for x in range(len(metrics))])
    ax.set_xticklabels(metrics, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1); ax.grid(True, axis="y", alpha=0.3); ax.legend()
    ax.set_title(f"CauSci transfer — {rep.get('tag','')}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "causci_metrics.png", dpi=140, bbox_inches="tight")
    print(f"Saved → {PLOTS / 'causci_metrics.png'}")


def plot_sft():
    """SFT training progress from output/sft/metrics.csv (periodic held-out grader)."""
    f = SFT_DIR / "metrics.csv"
    if not f.exists():
        print("no output/sft/metrics.csv — skip SFT plot")
        return
    rows = list(csv.DictReader(open(f)))
    if not rows:
        return
    # metrics.csv appends Phase A then Phase B, each with step reset to 0 → lay them on a
    # continuous axis (offset Phase B by Phase A's last step) so the curve doesn't jump backward.
    x, off, prev, bnd = [], 0, 0, None
    for r in rows:
        s = int(r["step"])
        if s < prev:
            off += prev; bnd = off        # phase boundary
        x.append(off + s); prev = s

    def line(ax, title, keys, labels):
        for k, lab in zip(keys, labels):
            ys = [float(r[k]) for r in rows if r.get(k) not in (None, "")]
            if ys:
                ax.plot(x[:len(ys)], ys, marker="o", ms=3, lw=1.3, label=lab)
        if bnd:
            ax.axvline(bnd, ls="--", c="gray", lw=1, alpha=0.6)   # Phase A | Phase B
        ax.set_title(title, fontsize=10); ax.set_xlabel("SFT step (Phase A → | → Phase B)"); ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    line(ax[0], "Per-step (overall)", ["graph_f1", "query_acc", "estimand_acc", "answer_acc", "full_correct"],
         ["graph_f1", "query", "estimand", "answer", "full"])
    line(ax[1], "Answer acc by rung", ["answer_acc", "r1/answer_acc", "r2/answer_acc", "r3/answer_acc"],
         ["overall", "rung1", "rung2", "rung3"])
    line(ax[2], "Estimand acc by rung", ["estimand_acc", "r1/estimand_acc", "r2/estimand_acc", "r3/estimand_acc"],
         ["overall", "rung1", "rung2", "rung3"])
    fig.suptitle("SFT — held-out CLadder diagnostic vs step", fontsize=13, fontweight="bold")
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "sft_metrics.png", dpi=140, bbox_inches="tight")
    print(f"Saved → {PLOTS / 'sft_metrics.png'}")


def plot_sft_phases():
    """Base vs after-Phase-A vs after-Phase-B snapshots from output/sft/phase_metrics.jsonl."""
    f = SFT_DIR / "phase_metrics.jsonl"
    if not f.exists():
        print("no output/sft/phase_metrics.jsonl — skip SFT phase plot")
        return
    recs = {json.loads(l)["phase"]: json.loads(l) for l in open(f)}   # last per phase wins
    phases = [p for p in ("base", "A", "B") if p in recs]
    if not phases:
        return
    lbl = {"base": "base", "A": "after Phase A", "B": "after Phase B"}
    panels = [("CLadder (held-out)", ["graph_f1", "query_acc", "estimand_acc", "answer_acc", "full_correct"]),
              ("CauSci transfer", ["causci_method_correctness", "causci_treatment_acc",
                                   "causci_outcome_acc", "causci_confounder_f1"])]
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    w = 0.8 / len(phases)
    for ax, (title, metrics) in zip(axes, panels):
        for i, p in enumerate(phases):
            ax.bar([x + i * w for x in range(len(metrics))], [recs[p].get(k, 0.0) for k in metrics],
                   w, label=lbl[p])
        ax.set_xticks([x + w * (len(phases) - 1) / 2 for x in range(len(metrics))])
        ax.set_xticklabels([k.replace("causci_", "") for k in metrics], rotation=15, fontsize=9)
        ax.set_ylim(0, 1); ax.grid(True, axis="y", alpha=0.3); ax.legend()
        ax.set_title(title, fontsize=12)
    fig.suptitle("SFT — base vs per-phase (CLadder in-dist. left, CauSci transfer right)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "sft_phases.png", dpi=140, bbox_inches="tight")
    print(f"Saved → {PLOTS / 'sft_phases.png'}")


def plot_causci_thinking():
    """Thinking OFF vs ON bars per checkpoint, from output/sft/causci_thinking.jsonl."""
    f = SFT_DIR / "causci_thinking.jsonl"
    if not f.exists():
        print("no output/sft/causci_thinking.jsonl — skip CauSci thinking plot")
        return
    rows = [json.loads(l) for l in open(f)]
    if not rows:
        return
    metrics = ["method_correctness", "treatment_acc", "outcome_acc", "confounder_f1"]
    labels = list(dict.fromkeys(r["label"] for r in rows))          # preserve file order
    data = {(r["label"], r["think"]): r for r in rows}
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    x = range(len(labels)); w = 0.38
    for ax, k in zip(axes, metrics):
        off = [data.get((l, False), {}).get(k, 0.0) for l in labels]
        on = [data.get((l, True), {}).get(k, 0.0) for l in labels]
        b0 = ax.bar([i - w / 2 for i in x], off, w, label="think off")
        b1 = ax.bar([i + w / 2 for i in x], on, w, label="think on")
        ax.bar_label(b0, fmt="%.2f", fontsize=7); ax.bar_label(b1, fmt="%.2f", fontsize=7)
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1); ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
        ax.set_title(k, fontsize=11)
    fig.suptitle("CauSci transfer — thinking OFF vs ON", fontsize=13, fontweight="bold")
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "causci_thinking.png", dpi=140, bbox_inches="tight")
    print(f"Saved → {PLOTS / 'causci_thinking.png'}")


def main():
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT.parent / "verl_training.log"
    if log.exists():
        plot_cladder(*parse_log(log))
    else:
        print(f"log not found: {log} — skip CLadder RL plot")
    plot_sft()
    plot_sft_phases()
    plot_causci()
    plot_causci_thinking()


if __name__ == "__main__":
    main()
