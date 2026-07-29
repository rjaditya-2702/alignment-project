"""causci_eval.py — CauSciBench TRANSFER scoring (potential-outcomes, method-menu; no model, no code).

Variable roles are WITHHELD in the prompt — the model must identify treatment/outcome/confounders
and choose a method from the fixed menu itself (the transfer signal). This module SCORES those
structured specs; it does not generate. During RL, CauSci is a second veRL validation set: veRL
rolls out with the live policy, and reward.py calls `score_causci` on each completion.

  <method>     → Method Correctness (accuracy + macro-F1 vs gold method bucket)
  <variables>  → Variable Selection: treatment/outcome exact + CONFOUNDER-SET F1 (the hard metric)
  effect       → deferred to the estimator tool (needs a vetted per-method library + CSVs); NOT here

`build_user` builds the prompt (data.py bakes it into causci_val.parquet, with CAUSCI_SYSTEM).
  conda run -n alignment python3 causci_eval.py --validate   # oracle sanity (no GPU)
"""

import argparse
import collections
import json
import re
from pathlib import Path

from schema import CAUSCI_USER

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent                                  # causal_alignment/ (csv_path is relative to here)
MERGED = PROJ / "data" / "merged.jsonl"             # canonical CauSciBench (real/synthetic/qr + synth_generated)
BENCH = ("real", "synthetic", "qr")                 # the 3 benchmark splits (difficulty ladder)

METHODS = ["ols", "psm", "iv", "did", "rdd", "frontdoor", "glm"]
MENU_TO_BUCKET = {"ols": "ols", "psm": "ps", "iv": "iv", "did": "did", "rdd": "rdd",
                  "frontdoor": "fd", "glm": "glm"}
REQUIRED_SLOTS = {"iv": ["instrument"], "rdd": ["running_variable", "cutoff"],
                  "did": ["time", "group"], "frontdoor": ["mediator"]}
SLOTS = ["treatment", "outcome", "confounders", "instrument", "running_variable",
         "cutoff", "time", "group", "mediator"]


def _norm_method(m):
    """Bucket a gold method name (CauSciBench standardize_method_name)."""
    if not isinstance(m, str):
        return None
    m = m.lower()
    if any(k in m for k in ("weighting", "ipw", "propensity", "matching", "psm")): return "ps"
    if "front" in m:                                                               return "fd"
    if any(k in m for k in ("discontinuity", "fuzzy", "rdd")):                     return "rdd"
    if any(k in m for k in ("in-difference", "did", "in-diff", "fixed effects", "panel")): return "did"
    if any(k in m for k in ("logistic", "probit", "logit", "glm")):                return "glm"
    if any(k in m for k in ("linear", "means", "ordinary", "rct", "ols", "wls")):  return "ols"
    if any(k in m for k in ("instrument", "encouragement", "2sls", "iv")):         return "iv"
    return None


def _sanit(s):
    return re.sub(r"[.\s\-]+", "_", str(s).strip().lower())


def _controls_list(c):
    if isinstance(c, list):
        return [str(x) for x in c]
    return [x.strip() for x in str(c or "").split(",") if x.strip()]


# ── prompt builder + column source (from the study's CSV header) ────────────

def csv_columns(csv_path):
    """Column names from the study's CSV header (csv_path is relative to causal_alignment/)."""
    import pandas as pd
    try:
        return pd.read_csv(PROJ / csv_path, nrows=0).columns.tolist()
    except Exception:
        return []


def build_user(description, columns, query):
    return CAUSCI_USER.format(description=description, columns="\n".join(columns), question=query)


def load_bench(limit=None):
    """CauSciBench benchmark records from merged.jsonl (real/synthetic/qr)."""
    recs = [json.loads(l) for l in open(MERGED)]
    recs = [r for r in recs if r.get("source") in BENCH]
    return recs[:limit] if limit else recs


# ── parse the method/variables spec ─────────────────────────────────────────

def parse_causci(text):
    if "</think>" in text:
        text = text.split("</think>")[-1]

    def tag(t):
        ms = re.findall(rf"<{t}>(.*?)</{t}>", text, re.DOTALL | re.IGNORECASE)
        return ms[-1].strip() if ms else ""

    mtext = tag("method").lower()
    method = next((k for k in METHODS if re.search(rf"\b{k}\b", mtext)), "")
    vb = tag("variables")

    def slot(name):
        m = re.search(rf"^\s*{name}\s*:\s*(.+)$", vb, re.IGNORECASE | re.MULTILINE)
        v = m.group(1).strip() if m else ""
        return "" if v.upper() in ("NA", "") else v

    out = {s: slot(s) for s in SLOTS}
    out["confounders"] = [x.strip() for x in out["confounders"].split(",") if x.strip()]
    out["method"] = method
    return out


def _match_col(v, cmap):
    s = _sanit(v)
    if not s:
        return None
    if s in cmap:
        return cmap[s]
    for cs, col in cmap.items():
        if cs and (cs in s or s in cs):
            return col
    return None


# ── scoring ─────────────────────────────────────────────────────────────────

def score_causci(rollout, columns, gt, split=None):
    """(scalar, comp) for one CauSci rollout. scalar = mean(method, treatment, outcome, confounder_f1)
    for veRL val logging; comp holds per-field flags for the aggregate [causci_eval] line.
    Decision on method/slot consistency: method correctness = method CHOICE only; whether the
    required slots are filled is a SEPARATE `valid` flag (valid_rate) — because CauSciBench gold
    often doesn't name method-specific columns, so gating method on slots would zero the iv/rdd/did
    ceiling. `valid` matters for the (deferred) effect tool, which needs a runnable spec."""
    p = parse_causci(rollout)
    cmap = {_sanit(c): c for c in columns}
    g1 = gt.get("step1") or {}
    gm = _norm_method(gt.get("step2"))
    pm = MENU_TO_BUCKET.get(p["method"])
    valid = all(p.get(s) for s in REQUIRED_SLOTS.get(p["method"], []))
    method = float(pm is not None and pm == gm)

    treat, out = _match_col(p["treatment"], cmap), _match_col(p["outcome"], cmap)
    t_ok = float(treat is not None and _sanit(treat) == _sanit(g1.get("treatment") or "x@x"))
    o_ok = float(out is not None and _sanit(out) == _sanit(g1.get("outcome") or "x@x"))

    gold_c = {_sanit(c) for c in _controls_list(g1.get("controls"))}
    pred_c = {_sanit(c) for c in (_match_col(x, cmap) for x in p["confounders"]) if c}
    tp = len(pred_c & gold_c)
    ctrl_acc = float(pred_c == gold_c)                              # exact-set match (predicted the gold set exactly)
    ctrl_recall = tp / len(gold_c) if gold_c else 1.0              # |correct| / |gold| (empty gold → satisfied)
    ctrl_prec = tp / len(pred_c) if pred_c else (1.0 if not gold_c else 0.0)  # |correct| / |predicted|
    if not gold_c and not pred_c:
        conf = 1.0
    elif not gold_c or not pred_c:
        conf = 0.0
    else:
        conf = 2 * tp / (len(pred_c) + len(gold_c)) if tp else 0.0   # set-F1 (= 2·overlap/(|p|+|g|))

    comp = {"method": method, "treatment": t_ok, "outcome": o_ok, "confounder_f1": conf,
            "control_recall": ctrl_recall, "control_precision": ctrl_prec, "control_acc": ctrl_acc,
            "gold_method": gm or "none", "pred_method": pm or "none", "valid": float(valid),
            "split": split or "?",
            # raw role picks (for truncation-free treatment/outcome role-confusion matrices)
            "pred_treatment": treat or "", "gold_treatment": g1.get("treatment") or "",
            "pred_outcome": out or "", "gold_outcome": g1.get("outcome") or "",
            "gold_controls": list(_controls_list(g1.get("controls")))}
    return (method + t_ok + o_ok + conf) / 4.0, comp


def compute_causci_metrics(items):
    """items: (_, comp) — CauSci transfer metrics for the [causci_eval] line, incl. method macro-F1."""
    comps = [c for _, c in items if c is not None]
    n = len(items) or 1
    m = {"method_correctness": sum(c["method"] for c in comps) / n,
         "treatment_acc": sum(c["treatment"] for c in comps) / n,
         "outcome_acc": sum(c["outcome"] for c in comps) / n,
         "confounder_f1": sum(c["confounder_f1"] for c in comps) / n,
         "control_recall": sum(c.get("control_recall", 0.0) for c in comps) / n,
         "control_precision": sum(c.get("control_precision", 0.0) for c in comps) / n,
         "control_acc": sum(c.get("control_acc", 0.0) for c in comps) / n,
         "valid_rate": sum(c["valid"] for c in comps) / n, "n": len(items)}
    # method macro-F1 over gold buckets present
    tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
    by_gold = collections.defaultdict(lambda: [0, 0])
    for c in comps:
        g, p = c["gold_method"], c["pred_method"]
        by_gold[g][1] += 1; by_gold[g][0] += c["method"]
        if p == g and c["method"]:
            tp[g] += 1
        else:
            fp[p] += 1; fn[g] += 1
    f1s = []
    for cls in by_gold:
        pr = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) else 0.0
        rc = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    m["method_f1"] = sum(f1s) / len(f1s) if f1s else 0.0
    for g, (ok, tot) in by_gold.items():
        m[f"m/{g}"] = ok / tot if tot else 0.0
    by_split = collections.defaultdict(list)          # per-split (real/synth/qr) difficulty ladder
    for c in comps:
        by_split[c.get("split") or "?"].append(c)
    for sp, cs in by_split.items():
        ns = len(cs)
        m[f"{sp}/method"] = sum(c["method"] for c in cs) / ns
        m[f"{sp}/treatment"] = sum(c["treatment"] for c in cs) / ns
        m[f"{sp}/outcome"] = sum(c["outcome"] for c in cs) / ns
        m[f"{sp}/conf_f1"] = sum(c["confounder_f1"] for c in cs) / ns
        m[f"{sp}/control_recall"] = sum(c.get("control_recall", 0.0) for c in cs) / ns
        m[f"{sp}/control_precision"] = sum(c.get("control_precision", 0.0) for c in cs) / ns
        m[f"{sp}/control_acc"] = sum(c.get("control_acc", 0.0) for c in cs) / ns
    return m


# ── oracle output (for --validate sanity): gold method + gold roles ─────────

def gold_output(gt):
    g1 = gt.get("step1") or {}
    menu = {"ols": "ols", "ps": "psm", "iv": "iv", "did": "did", "rdd": "rdd",
            "fd": "frontdoor", "glm": "glm"}.get(_norm_method(gt.get("step2")), "ols")
    conf = ", ".join(_controls_list(g1.get("controls"))) or "NA"
    v = {"treatment": g1.get("treatment"), "outcome": g1.get("outcome"), "confounders": conf,
         "instrument": g1.get("instrument") or "NA", "running_variable": g1.get("running_variable") or "NA",
         "cutoff": "NA", "time": g1.get("time_variable") or "NA", "group": g1.get("group_variable") or "NA",
         "mediator": g1.get("mediator") or "NA"}
    body = "\n".join(f"{k}: {v[k]}" for k in SLOTS)
    return f"<method>{menu}</method>\n<variables>\n{body}\n</variables>\n<answer>positive</answer>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="oracle (gold method+roles) sanity, no GPU")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    items = []
    for r in load_bench(args.limit):
        gt = {"step1": r["step1"], "step2": r["method"]}
        rollout = gold_output(gt) if args.validate else ""
        items.append((None, score_causci(rollout, csv_columns(r["csv_path"]), gt, r["source"])[1]))
    m = compute_causci_metrics(items)
    print("[causci overall] " + " ".join(f"{k}:{v:.3f}" for k, v in m.items()
                                          if isinstance(v, float) and "/" not in k))
    for sp in BENCH:
        print(f"  {sp}: method={m.get(f'{sp}/method',0):.3f} conf_f1={m.get(f'{sp}/conf_f1',0):.3f}")


if __name__ == "__main__":
    main()
