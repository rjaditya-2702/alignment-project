"""
Per-step scoring and aggregate metrics.

CLadder per-step weights:
  step1 (structure):      11 pts  — format check (has arrows)
  step2 (query type):     15 pts  — exact match
  step3 (estimand):       24 pts  — DeepSeek-Math judge (0/12/24)
  step5 (final answer):   20 pts  — exact yes/no match
  Total:                  70 pts

CauSciBench per-step weights:
  step1 (variable ID):     5 pts  — field presence check
  step2 (method):          5 pts  — exact match
  step3 (spec):           15 pts  — DeepSeek-Math judge (0/7/15)
  step5 (numeric answer): 30 pts  — relative error < threshold
  step5_exact:             5 pts  — within 1% of ground truth
  Total:                  60 pts
"""

import torch
from collections import defaultdict


# ── LLM judge (DeepSeek-Math) ─────────────────────────────────────────────────

CLADDER_JUDGE_SYSTEM = (
    "You are an expert in causal inference. "
    "Score whether a predicted estimand expression is semantically equivalent to the reference. "
    "Reply with a single integer only: 0 (wrong or missing), 1 (partially correct), 2 (correct)."
)

CAUSCI_JUDGE_SYSTEM = (
    "You are an expert in causal inference. "
    "Score whether the estimation specification is appropriate for the given method and identified variables. "
    "Reply with a single integer only: 0 (inappropriate), 1 (partially appropriate), 2 (appropriate)."
)


def _format_prompt(system: str, user_msg: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system}\nUser: {user_msg}\nAssistant:"


def _run_judge(prompt: str, judge_model, judge_tokenizer) -> int:
    inputs = judge_tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(judge_model.device)
    with torch.no_grad():
        out = judge_model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=judge_tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    text = judge_tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    try:
        return min(max(int(text[0]), 0), 2)
    except (ValueError, IndexError):
        return 0


def judge_estimand(predicted: str, reference: str, judge_model, judge_tokenizer) -> int:
    """Returns 0, 1, or 2."""
    if not predicted.strip():
        return 0
    user_msg = f"Reference: {reference}\nPredicted: {predicted}\nScore:"
    prompt = _format_prompt(CLADDER_JUDGE_SYSTEM, user_msg, judge_tokenizer)
    return _run_judge(prompt, judge_model, judge_tokenizer)


def judge_spec(spec: str, method: str, step1: str, judge_model, judge_tokenizer) -> int:
    """Returns 0, 1, or 2."""
    if not spec.strip():
        return 0
    user_msg = f"Method: {method}\nVariables identified: {step1}\nSpecification: {spec}\nScore:"
    prompt = _format_prompt(CAUSCI_JUDGE_SYSTEM, user_msg, judge_tokenizer)
    return _run_judge(prompt, judge_model, judge_tokenizer)


# ── CLadder scoring ───────────────────────────────────────────────────────────

def score_cladder(parsed: dict, gt: dict, judge_model, judge_tokenizer) -> dict:
    scores = {}

    # Step 1: structure has at least one arrow
    has_arrow = "->" in parsed["step1"] or "→" in parsed["step1"]
    scores["step1"] = 11 if has_arrow else 0

    # Step 2: query type exact match
    scores["step2"] = 15 if parsed["step2"] == gt["step2"] else 0

    # Step 3: estimand — DeepSeek judge
    if gt.get("step3"):
        judge_score = judge_estimand(parsed["step3"], gt["step3"], judge_model, judge_tokenizer)
        scores["step3"] = [0, 12, 24][judge_score]
    else:
        scores["step3"] = 0

    # Step 5: final answer exact match
    scores["step5"] = 20 if parsed["step5"] == str(gt["step5"]).lower() else 0

    scores["total"] = sum(scores.values())
    scores["correct"] = parsed["step5"] == str(gt["step5"]).lower()
    return scores


# ── CauSciBench scoring ────────────────────────────────────────────────────────

def score_causcibench(parsed: dict, gt: dict, judge_model, judge_tokenizer) -> dict:
    scores = {}

    # Step 1: variable identification
    step1_text = parsed["step1"].lower()
    has_treatment = "treatment" in step1_text
    has_outcome   = "outcome" in step1_text
    scores["step1"] = 5 if (has_treatment and has_outcome) else (2 if has_treatment or has_outcome else 0)

    # Step 2: method exact match
    scores["step2"] = 5 if parsed["step2"] == gt["step2"] else 0

    # Step 3: estimation spec — DeepSeek judge
    judge_score = judge_spec(parsed["step3"], parsed["step2"], parsed["step1"], judge_model, judge_tokenizer)
    scores["step3"] = [0, 7, 15][judge_score]

    # Step 5: numeric answer — relative error vs ground truth
    gt_val   = gt.get("step5")
    pred_val = parsed["step5"]
    scores["step5"]       = 0
    scores["step5_exact"] = 0
    scores["rel_error"]   = None

    if pred_val is not None and gt_val is not None:
        try:
            pred_f  = float(pred_val)
            gt_f    = float(gt_val)
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
        + scores["step5"] + scores["step5_exact"]
    )
    return scores


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def aggregate_metrics(results: list[dict]) -> dict:
    cladder = [r for r in results if r["source"] == "cladder"]
    causci  = [r for r in results if r["source"] == "causcibench"]

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def _pct(vals):
        return _mean(vals) * 100

    metrics = {}

    if cladder:
        metrics["cladder"] = {
            "n":          len(cladder),
            "accuracy":   _pct([r["scores"]["correct"] for r in cladder]),
            "avg_score":  _mean([r["scores"]["total"]  for r in cladder]),
            "step1_avg":  _mean([r["scores"]["step1"]  for r in cladder]),
            "step2_avg":  _mean([r["scores"]["step2"]  for r in cladder]),
            "step3_avg":  _mean([r["scores"]["step3"]  for r in cladder]),
            "step5_avg":  _mean([r["scores"]["step5"]  for r in cladder]),
        }
        by_qt = defaultdict(list)
        for r in cladder:
            qt = r["groundtruth"].get("step2", "unknown")
            by_qt[qt].append(r["scores"]["correct"])
        metrics["cladder"]["by_query_type"] = {
            qt: {"n": len(v), "accuracy": _pct(v)} for qt, v in sorted(by_qt.items())
        }

    if causci:
        rel_errors = [r["scores"]["rel_error"] for r in causci if r["scores"]["rel_error"] is not None]

        metrics["causcibench"] = {
            "n":               len(causci),
            "avg_score":       _mean([r["scores"]["total"]  for r in causci]),
            "method_accuracy": _pct([r["scores"]["step2"] == 5 for r in causci]),
            "step1_avg":       _mean([r["scores"]["step1"]  for r in causci]),
            "step2_avg":       _mean([r["scores"]["step2"]  for r in causci]),
            "step3_avg":       _mean([r["scores"]["step3"]  for r in causci]),
            "step5_avg":       _mean([r["scores"]["step5"]  for r in causci]),
            "median_rel_error": sorted(rel_errors)[len(rel_errors) // 2] if rel_errors else None,
        }
        by_method = defaultdict(list)
        for r in causci:
            m = r["groundtruth"].get("step2", "unknown")
            by_method[m].append(r["scores"]["step2"] == 5)
        metrics["causcibench"]["by_method"] = {
            m: {"n": len(v), "accuracy": _pct(v)} for m, v in sorted(by_method.items())
        }

    return metrics
