import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.training.tool_calling import library_fn

# Pearl's ladder of causation — derived from query type
RUNG_MAP = {
    "marginal":            1,  "correlation":         1,
    "ate":                 2,  "backadj":             2,  "collider_bias":       2,
    "exp_away":            2,  "ett":                 2,
    "det-counterfactual":  3,  "nde":                 3,  "nie":                 3,
}


def compute_eval_metrics(items):
    """
    items: list of (source, parsed_prediction, ground_truth_dict, csv_path)

    CLaDDer gt:   step2 (query_type), step5 (yes/no label), is_commonsense (optional bool)
    CauSciBench gt: step1 (dict), step2 (method name), step5 (effect float)

    Returns dict of metric_name -> float.
    Commonsense breakdown is skipped when is_commonsense is absent from gt.
    """
    cladder_items = [(p, g)    for (src, p, g, _) in items if src == "cladder"]
    causci_items  = [(p, g, c) for (src, p, g, c) in items if src == "causcibench"]

    metrics = {}

    # ── CLaDDer ────────────────────────────────────────────────────────────────
    if cladder_items:
        by_rung        = {1: [], 2: [], 3: []}
        cs_correct     = []
        non_cs_correct = []
        all_correct    = []

        for parsed, gt in cladder_items:
            pred    = (parsed or {}).get("step5", "").strip().lower()
            true    = str(gt.get("step5", "")).strip().lower()
            correct = int(pred == true)
            all_correct.append(correct)

            rung = RUNG_MAP.get(gt.get("step2", ""), 0)
            if rung in by_rung:
                by_rung[rung].append(correct)

            is_cs = gt.get("is_commonsense")
            if is_cs is not None:
                (cs_correct if is_cs else non_cs_correct).append(correct)

        metrics["cladder/overall_acc"] = sum(all_correct) / len(all_correct)
        for r in [1, 2, 3]:
            if by_rung[r]:
                metrics[f"cladder/rung{r}_acc"] = sum(by_rung[r]) / len(by_rung[r])
        if cs_correct:
            metrics["cladder/commonsense_acc"]     = sum(cs_correct)     / len(cs_correct)
        if non_cs_correct:
            metrics["cladder/non_commonsense_acc"] = sum(non_cs_correct) / len(non_cs_correct)

    # ── CauSciBench ────────────────────────────────────────────────────────────
    if causci_items:
        method_correct    = []
        treatment_correct = []
        outcome_correct   = []
        effect_correct    = []
        mres              = []

        for parsed, gt, csv_path in causci_items:
            if parsed is None:
                method_correct.append(0)
                continue

            pred_method = parsed.get("step2", "").strip().lower()
            gt_method   = (gt.get("step2") or "").strip().lower()
            m_ok        = int(pred_method == gt_method)
            method_correct.append(m_ok)
            if not m_ok:
                continue

            pred_s1 = parsed.get("step1") or {}
            gt_s1   = gt.get("step1") or {}

            t_ok = int(pred_s1.get("treatment", "").strip() == str(gt_s1.get("treatment", "")).strip())
            treatment_correct.append(t_ok)
            o_ok = int(pred_s1.get("outcome", "").strip() == str(gt_s1.get("outcome", "")).strip())
            outcome_correct.append(o_ok)

            if t_ok and o_ok and csv_path:
                parsed["step1"]["csv_path"] = csv_path
                effect = library_fn(parsed)
                ref    = gt.get("step5")
                if ref is not None and ref != 0:
                    mre = abs(effect - ref) / abs(ref)
                    mres.append(mre)
                    effect_correct.append(int(mre <= 0.05))

        metrics["causci/method_acc"]    = sum(method_correct)    / len(method_correct)    if method_correct    else 0.0
        metrics["causci/treatment_acc"] = sum(treatment_correct) / len(treatment_correct) if treatment_correct else 0.0
        metrics["causci/outcome_acc"]   = sum(outcome_correct)   / len(outcome_correct)   if outcome_correct   else 0.0
        metrics["causci/effect_acc"]    = sum(effect_correct)    / len(effect_correct)    if effect_correct    else 0.0
        metrics["causci/mre"]           = sum(mres)              / len(mres)              if mres              else 0.0

    return metrics


def save_eval_plots(history, steps, plot_dir):
    """
    history:  dict of metric_name -> list[float]   (one entry per eval run)
    steps:    list[int]                             (global step at each eval run)
    plot_dir: str | Path
    Saves eval_cladder.png and eval_causci.png to plot_dir.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    cladder_keys = sorted(k for k in history if k.startswith("cladder/"))
    causci_keys  = sorted(k for k in history if k.startswith("causci/"))
    reward_keys  = sorted(k for k in history if k.startswith("reward/"))
    train_keys   = sorted(k for k in history if k.startswith("train/"))

    for keys, suptitle, fname in [
        (train_keys,   "Training Metrics",  "train_metrics.png"),
        (cladder_keys, "CLaDDer Eval",      "eval_cladder.png"),
        (causci_keys,  "CauSciBench Eval",  "eval_causci.png"),
        (reward_keys,  "Training Reward",   "eval_reward.png"),
    ]:
        if not keys:
            continue
        n   = len(keys)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, key in zip(axes, keys):
            vals = history[key]
            ax.plot(steps[:len(vals)], vals, marker="o", markersize=4, linewidth=1.5)
            ax.set_title(key.split("/")[1])
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        fig.tight_layout()
        path = plot_dir / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved → {path}")
