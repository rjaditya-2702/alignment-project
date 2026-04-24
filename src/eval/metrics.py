"""
Per-step scoring and aggregate metrics.

CLadder per-step weights:
  step1 (structure):      11 pts  — format check (has arrows)
  step2 (query type):     15 pts  — exact match
  step3 (estimand):       24 pts  — LLM judge (semantic)
  step4 (code executed):  30 pts  — code runs + result parseable
  step5 (final answer):   20 pts  — exact yes/no match
  Total:                 100 pts

CauSciBench per-step weights:
  step1 (variable ID):     5 pts  — field presence check
  step2 (method):          5 pts  — exact match
  step3 (spec):           15 pts  — non-empty
  step4 (code executed):  30 pts  — code runs + result parseable
  step5 (numeric answer): 30 pts  — relative error < threshold
  bonus (method correct): 10 pts  — if step2 correct AND code ran
  step5_exact:             5 pts  — within 1% of ground truth
  Total:                 100 pts
"""

import math
from collections import defaultdict

from openai import OpenAI

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ── LLM judge for CLadder step3 ───────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are an expert in causal inference. "
    "Score whether a predicted estimand expression is semantically equivalent to the reference. "
    "Reply with a single integer 0, 1, or 2.\n"
    "2 = correct (equivalent expression, same estimand)\n"
    "1 = partially correct (right structure but minor errors)\n"
    "0 = wrong or missing"
)


def judge_estimand(predicted: str, reference: str) -> int:
    """Returns 0, 1, or 2."""
    if not predicted.strip():
        return 0
    prompt = f"Reference: {reference}\nPredicted: {predicted}\nScore:"
    try:
        resp = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            max_completion_tokens=8,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        val = int(raw[0])
        return min(max(val, 0), 2)
    except Exception:
        return 0


# ── CLadder scoring ───────────────────────────────────────────────────────────

def score_cladder(parsed: dict, gt: dict, sandbox_result: dict, use_llm_judge: bool = True) -> dict:
    scores = {}

    # Step 1: structure has at least one arrow
    has_arrow = "->" in parsed["step1"] or "→" in parsed["step1"]
    scores["step1"] = 11 if has_arrow else 0

    # Step 2: query type exact match
    scores["step2"] = 15 if parsed["step2"] == gt["step2"] else 0

    # Step 3: LLM judge (or skip if no_llm)
    if use_llm_judge and gt.get("step3"):
        judge_score = judge_estimand(parsed["step3"], gt["step3"])
        scores["step3"] = [0, 12, 24][judge_score]
    else:
        scores["step3"] = 12 if parsed["step3"].strip() else 0  # partial credit for non-empty

    # Step 4: code ran and produced a result
    if sandbox_result.get("ok") and sandbox_result.get("result") is not None:
        scores["step4"] = 30
    elif parsed["step4_code"]:
        scores["step4"] = 10  # code present but didn't run
    else:
        scores["step4"] = 0

    # Step 5: final answer exact match
    scores["step5"] = 20 if parsed["step5"] == str(gt["step5"]).lower() else 0

    scores["total"] = sum(scores.values())
    scores["correct"] = parsed["step5"] == str(gt["step5"]).lower()
    return scores


# ── CauSciBench scoring ────────────────────────────────────────────────────────

def score_causcibench(parsed: dict, gt: dict, sandbox_result: dict) -> dict:
    scores = {}

    # Step 1: variable identification — check key fields present in output
    step1_text = parsed["step1"].lower()
    has_treatment = "treatment" in step1_text
    has_outcome = "outcome" in step1_text
    scores["step1"] = 5 if (has_treatment and has_outcome) else (2 if has_treatment or has_outcome else 0)

    # Step 2: method exact match
    method_correct = parsed["step2"] == gt["step2"]
    scores["step2"] = 5 if method_correct else 0

    # Step 3: estimation spec non-empty
    scores["step3"] = 15 if parsed["step3"].strip() else 0

    # Step 4: code ran and produced numeric result
    code_ok = sandbox_result.get("ok") and sandbox_result.get("result") is not None
    if code_ok:
        scores["step4"] = 30
    elif parsed["step4_code"]:
        scores["step4"] = 10
    else:
        scores["step4"] = 0

    # Bonus: method correct + code ran
    scores["bonus"] = 10 if (method_correct and code_ok) else 0

    # Step 5: numeric answer — relative error vs ground truth
    gt_val = gt.get("step5")
    pred_val = parsed["step5"]
    scores["step5"] = 0
    scores["step5_exact"] = 0
    scores["rel_error"] = None

    if code_ok and sandbox_result.get("result") is not None:
        # prefer sandbox result over parsed step5 text
        try:
            pred_val = float(sandbox_result["result"])
        except (ValueError, TypeError):
            pass

    if pred_val is not None and gt_val is not None:
        try:
            pred_f = float(pred_val)
            gt_f = float(gt_val)
            if gt_f == 0:
                rel_err = abs(pred_f) if pred_f != 0 else 0.0
            else:
                rel_err = abs(pred_f - gt_f) / abs(gt_f)
            scores["rel_error"] = rel_err
            if rel_err <= 0.50:
                scores["step5"] = 30
            elif rel_err <= 1.00:
                scores["step5"] = 15
            if rel_err <= 0.01:
                scores["step5_exact"] = 5
        except (ValueError, TypeError):
            pass

    scores["total"] = (
        scores["step1"] + scores["step2"] + scores["step3"]
        + scores["step4"] + scores["bonus"] + scores["step5"] + scores["step5_exact"]
    )
    return scores


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def aggregate_metrics(results: list[dict]) -> dict:
    """
    results: list of dicts with keys: source, scores, parsed, groundtruth
    """
    cladder = [r for r in results if r["source"] == "cladder"]
    causci  = [r for r in results if r["source"] == "causcibench"]

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def _pct(vals):
        return _mean(vals) * 100

    metrics = {}

    # CLadder
    if cladder:
        metrics["cladder"] = {
            "n": len(cladder),
            "accuracy": _pct([r["scores"]["correct"] for r in cladder]),
            "avg_score": _mean([r["scores"]["total"] for r in cladder]),
            "step1_avg": _mean([r["scores"]["step1"] for r in cladder]),
            "step2_avg": _mean([r["scores"]["step2"] for r in cladder]),
            "step3_avg": _mean([r["scores"]["step3"] for r in cladder]),
            "step4_avg": _mean([r["scores"]["step4"] for r in cladder]),
            "step5_avg": _mean([r["scores"]["step5"] for r in cladder]),
        }
        # Per query type breakdown
        by_qt = defaultdict(list)
        for r in cladder:
            qt = r["groundtruth"].get("step2", "unknown")
            by_qt[qt].append(r["scores"]["correct"])
        metrics["cladder"]["by_query_type"] = {
            qt: {"n": len(v), "accuracy": _pct(v)} for qt, v in sorted(by_qt.items())
        }

    # CauSciBench
    if causci:
        method_correct = [r["scores"]["step2"] == 5 for r in causci]
        code_ran = [r["scores"]["step4"] == 30 for r in causci]
        rel_errors = [r["scores"]["rel_error"] for r in causci if r["scores"]["rel_error"] is not None]

        metrics["causcibench"] = {
            "n": len(causci),
            "avg_score": _mean([r["scores"]["total"] for r in causci]),
            "method_accuracy": _pct(method_correct),
            "code_execution_rate": _pct(code_ran),
            "step1_avg": _mean([r["scores"]["step1"] for r in causci]),
            "step2_avg": _mean([r["scores"]["step2"] for r in causci]),
            "step3_avg": _mean([r["scores"]["step3"] for r in causci]),
            "step4_avg": _mean([r["scores"]["step4"] for r in causci]),
            "step5_avg": _mean([r["scores"]["step5"] for r in causci]),
            "median_rel_error": sorted(rel_errors)[len(rel_errors)//2] if rel_errors else None,
        }
        by_method = defaultdict(list)
        for r in causci:
            m = r["groundtruth"].get("step2", "unknown")
            by_method[m].append(r["scores"]["step2"] == 5)
        metrics["causcibench"]["by_method"] = {
            m: {"n": len(v), "accuracy": _pct(v)} for m, v in sorted(by_method.items())
        }

    return metrics
