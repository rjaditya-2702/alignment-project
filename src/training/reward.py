"""
Reward functions for GRPO training.

CLadder (100 max, cascading -100 penalty per failed step):
    Step 1 (graph):        11   — arrow format check
    Step 2 (query type):   15   — exact match; cascade: wrong type → steps 3-5 also penalized
    Step 3 (derivation):   24   — non-empty + math notation proxy (LLM judge too slow for training)
    Step 4 (computation):  30   — code executes + result parseable
    Step 5 (answer):       20   — yes/no exact match

CauSciBench (100 max, independent -50 penalty per failed step, no cascade):
    Step 1 breakdown:
        treatment match:   5
        outcome match:     5
        control overlap:  15
        instrument/rv/tv:  5
    Step 2 (method):      30   — exact match
    Step 3 (spec):         0   — no ground truth, skip
    Step 4 (code runs):   10   — code executes + result parseable
    Step 5 (answer):      30   — relative error scoring
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.parser import parse_completion
from src.eval.sandbox import execute_batch, execute_code

# ── CLadder ───────────────────────────────────────────────────────────────────

_MATH_MARKERS = re.compile(
    r"E\[|P\(|do\(|Y_|_{|\\frac|\\sum|\\prod|\|do|<->|->|E\s*\[|P\s*\(",
    re.IGNORECASE,
)

CLADDER_QUERY_TYPES = {
    "marginal", "correlation", "ate", "backadj", "det-counterfactual",
    "ett", "nde", "nie", "collider_bias", "exp_away",
}


def _score_cladder(parsed: dict, gt: dict, sandbox: dict) -> float:
    total = 0.0

    # Step 1: causal graph — needs at least one arrow
    step1_ok = "->" in parsed["step1"] or "→" in parsed["step1"]
    total += 11 if step1_ok else -100

    # Step 2: query type exact match
    step2_ok = parsed["step2"] == gt["step2"]
    total += 15 if step2_ok else -100

    # Step 3: estimand derivation — non-empty with math notation
    # Skip LLM judge during training; use proxy
    if not step2_ok:
        # Cascade: wrong query type means derivation is also wrong
        total += -100
    else:
        has_math = bool(_MATH_MARKERS.search(parsed["step3"])) if parsed["step3"] else False
        total += 24 if has_math else -100

    # Step 4: code executed and produced a result
    code_ok = sandbox.get("ok") and sandbox.get("result") is not None
    total += 30 if code_ok else -100

    # Step 5: final answer yes/no
    gt_ans = str(gt.get("step5", "")).lower().strip()
    step5_ok = parsed["step5"] == gt_ans
    total += 20 if step5_ok else -100

    return total


# ── CauSciBench ───────────────────────────────────────────────────────────────

CAUSCI_METHODS = {
    "diff_in_means", "ols", "ipw", "matching", "did", "rdd", "iv", "frontdoor", "glm",
}

# Methods that require specific variable types beyond treatment/outcome/controls
_METHOD_EXTRA_VAR = {
    "iv":       "instrument",
    "frontdoor": "instrument",   # mediator plays same role
    "rdd":      "running_variable",
    "did":      "time_variable",
}


def _step1_score(parsed_step1: str, gt_step1: dict) -> float:
    """Detailed step1 scoring for CauSciBench. Returns score in [0, 30] or -50 penalty."""
    text = parsed_step1.lower()

    # Extract predicted values from structured output
    def _extract(label: str) -> str:
        m = re.search(rf"{label}\s*:\s*(\S+)", text)
        return m.group(1).strip(".,;") if m else ""

    pred_treatment = _extract("treatment")
    pred_outcome = _extract("outcome")

    gt_treatment = str(gt_step1.get("treatment") or "").lower()
    gt_outcome = str(gt_step1.get("outcome") or "").lower()

    # No treatment or outcome predicted → failed step
    if not pred_treatment and not pred_outcome:
        return -50

    score = 0.0
    # Treatment match: 5 pts
    if gt_treatment and pred_treatment and gt_treatment in pred_treatment:
        score += 5
    # Outcome match: 5 pts
    if gt_outcome and pred_outcome and gt_outcome in pred_outcome:
        score += 5

    # Control overlap: 15 pts — Jaccard-weighted
    gt_controls = set()
    raw_controls = gt_step1.get("controls")
    if isinstance(raw_controls, list):
        gt_controls = {str(c).lower() for c in raw_controls if c}
    elif isinstance(raw_controls, str) and raw_controls:
        gt_controls = {raw_controls.lower()}

    if gt_controls:
        # Extract controls list from predicted text
        ctrl_m = re.search(r"controls\s*:\s*\[?([^\]\n]+)", text)
        pred_controls = set()
        if ctrl_m:
            for tok in re.split(r"[,\s]+", ctrl_m.group(1)):
                t = tok.strip("[].,;\"'")
                if t:
                    pred_controls.add(t)
        if pred_controls or gt_controls:
            jaccard = len(pred_controls & gt_controls) / len(pred_controls | gt_controls) if (pred_controls | gt_controls) else 0
            score += 15 * jaccard

    # Special variables: instrument, running_variable, time_variable, group_variable — 5 pts
    # If gt says all null → model must not hallucinate any; if gt has one → model must identify it.
    _NONE_WORDS = {"none", "na", "n/a", "-", "null", ""}
    gt_special = {
        "instrument":       gt_step1.get("instrument"),
        "running_variable": gt_step1.get("running_variable"),
        "time_variable":    gt_step1.get("time_variable"),
        "group_variable":   gt_step1.get("group_variable"),
    }
    active = {k: str(v).lower() for k, v in gt_special.items() if v is not None}

    if not active:
        # All null — penalize if model predicts any non-none value
        hallucinated = any(
            _extract(k).lower() not in _NONE_WORDS
            for k in gt_special
        )
        score += 5 if not hallucinated else 0
    else:
        # One or more active special variables — check if model identifies them
        correct = sum(
            1 for k, gt_v in active.items()
            if gt_v in _extract(k).lower()
        )
        score += 5 if correct == len(active) else 0

    return score


def _score_causcibench(parsed: dict, gt: dict, sandbox: dict) -> float:
    total = 0.0

    # Step 1: variable identification
    s1 = _step1_score(parsed["step1"], gt.get("step1") or {})
    total += s1

    # Step 2: method exact match — 30 pts
    method_ok = parsed["step2"] == gt["step2"]
    total += 30 if method_ok else -50

    # Step 3: skip (no ground truth)

    # Step 4: code executed and produced a result — 10 pts
    code_ok = sandbox.get("ok") and sandbox.get("result") is not None
    total += 10 if code_ok else -50

    # Step 5: numeric answer — relative error scoring
    gt_val = gt.get("step5")
    pred_val = None

    # Prefer sandbox result, fall back to parsed step5 text
    if code_ok and sandbox.get("result"):
        try:
            pred_val = float(sandbox["result"])
        except (ValueError, TypeError):
            pass
    if pred_val is None:
        pred_val = parsed.get("step5")

    if pred_val is not None and gt_val is not None:
        try:
            pred_f = float(pred_val)
            gt_f = float(gt_val)
            denom = abs(gt_f) if gt_f != 0 else 1.0
            rel_err = abs(pred_f - gt_f) / denom
            if rel_err <= 0.10:
                total += 30
            elif rel_err <= 0.25:
                total += 20
            elif rel_err <= 0.50:
                total += 10
            else:
                total += -50
        except (ValueError, TypeError):
            total += -50
    else:
        total += -50

    return total


# ── Dispatch ──────────────────────────────────────────────────────────────────

def score_completion(completion: str, row: dict, sandbox: dict) -> float:
    """Score a single completion given pre-computed sandbox result."""
    parsed = parse_completion(completion, row["source"])
    if row["source"] == "cladder":
        return _score_cladder(parsed, row["groundtruth"], sandbox)
    return _score_causcibench(parsed, row["groundtruth"], sandbox)


def compute_rewards(completions: list[str], rows: list[dict], max_workers: int = 8) -> list[float]:
    """
    Score a batch of completions in parallel.
    completions and rows must be the same length.
    Returns list of scalar rewards (can be negative).
    """
    # Parse all completions first (CPU, fast)
    parsed_list = [parse_completion(c, r["source"]) for c, r in zip(completions, rows)]
    codes = [p.get("step4_code", "") for p in parsed_list]

    # Execute all code in parallel
    sandbox_results = execute_batch(codes, max_workers=max_workers)

    rewards = []
    for row, parsed, sandbox in zip(rows, parsed_list, sandbox_results):
        if row["source"] == "cladder":
            r = _score_cladder(parsed, row["groundtruth"], sandbox)
        else:
            r = _score_causcibench(parsed, row["groundtruth"], sandbox)
        rewards.append(float(r))

    return rewards
