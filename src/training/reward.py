"""
Reward functions for GRPO training.

CLadder (max 100, cascading -100 penalty per failed step):
    Step 1 (graph):        11   — arrow format check
    Step 2 (query type):   15   — exact match; cascade: wrong type → step 3 also penalized
    Step 3 (derivation):   24   — DeepSeek-Math judge (0/12/24)
    Step 5 (answer):       20   — yes/no exact match

CauSciBench (max 105, independent -50 penalty per failed step, no cascade):
    Step 1 breakdown:
        treatment match:   5
        outcome match:     5
        control overlap:  15
        special var:       5
    Step 2 (method):      30   — exact match
    Step 3 (spec):        15   — DeepSeek-Math judge (0/7/15)
    Step 5 (answer):      30   — relative error scoring
"""

import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.parser import parse_completion

# ── Judge prompts ─────────────────────────────────────────────────────────────

_CLADDER_JUDGE_SYSTEM = (
    "You are an expert in causal inference. "
    "Score whether a predicted estimand expression is semantically equivalent to the reference. "
    "Reply with a single integer only: 0 (wrong or missing), 1 (partially correct), 2 (correct)."
)

_CAUSCI_JUDGE_SYSTEM = (
    "You are an expert in causal inference. "
    "Score whether the estimation specification is appropriate for the given method and identified variables. "
    "Reply with a single integer only: 0 (inappropriate), 1 (partially appropriate), 2 (appropriate)."
)


def _format_judge_prompts(system: str, user_messages: list[str], tokenizer) -> list[str]:
    prompts = []
    for user_msg in user_messages:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"System: {system}\nUser: {user_msg}\nAssistant:"
        prompts.append(prompt)
    return prompts


def _run_judge_batch(prompts: list[str], judge_model, judge_tokenizer) -> list[int]:
    """Run judge on a batch of prompts. Returns list of scores 0/1/2."""
    inputs = judge_tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(judge_model.device)

    with torch.no_grad():
        out = judge_model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=judge_tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    scores = []
    for o in out:
        text = judge_tokenizer.decode(o[prompt_len:], skip_special_tokens=True).strip()
        try:
            val = int(text[0])
            scores.append(min(max(val, 0), 2))
        except (ValueError, IndexError):
            scores.append(0)
    return scores


def _judge_cladder_step3(
    predicted_list: list[str],
    reference_list: list[str],
    judge_model,
    judge_tokenizer,
) -> list[int]:
    user_msgs = [
        f"Reference: {ref}\nPredicted: {pred}\nScore:"
        for pred, ref in zip(predicted_list, reference_list)
    ]
    prompts = _format_judge_prompts(_CLADDER_JUDGE_SYSTEM, user_msgs, judge_tokenizer)
    return _run_judge_batch(prompts, judge_model, judge_tokenizer)


def _judge_causci_step3(
    spec_list: list[str],
    method_list: list[str],
    step1_list: list[str],
    judge_model,
    judge_tokenizer,
) -> list[int]:
    user_msgs = [
        f"Method: {method}\nVariables identified: {step1}\nSpecification: {spec}\nScore:"
        for spec, method, step1 in zip(spec_list, method_list, step1_list)
    ]
    prompts = _format_judge_prompts(_CAUSCI_JUDGE_SYSTEM, user_msgs, judge_tokenizer)
    return _run_judge_batch(prompts, judge_model, judge_tokenizer)


# ── CLadder ───────────────────────────────────────────────────────────────────

CLADDER_QUERY_TYPES = {
    "marginal", "correlation", "ate", "backadj", "det-counterfactual",
    "ett", "nde", "nie", "collider_bias", "exp_away",
}


def _score_cladder(parsed: dict, gt: dict, step3_judge: int) -> float:
    total = 0.0

    # Step 1: causal graph — needs at least one arrow
    step1_ok = "->" in parsed["step1"] or "→" in parsed["step1"]
    total += 11 if step1_ok else -100

    # Step 2: query type exact match
    step2_ok = parsed["step2"] == gt["step2"]
    total += 15 if step2_ok else -100

    # Step 3: estimand — cascade from step 2; judge score → 0/12/24
    if not step2_ok:
        total += -100
    else:
        total += [0, 12, 24][step3_judge]

    # Step 5: final answer yes/no
    gt_ans = str(gt.get("step5", "")).lower().strip()
    total += 20 if parsed["step5"] == gt_ans else -100

    return total


# ── CauSciBench ───────────────────────────────────────────────────────────────

CAUSCI_METHODS = {
    "diff_in_means", "ols", "ipw", "matching", "did", "rdd", "iv", "frontdoor", "glm",
}


def _step1_score(parsed_step1: str, gt_step1: dict) -> float:
    text = parsed_step1.lower()

    def _extract(label: str) -> str:
        m = re.search(rf"{label}\s*:\s*(\S+)", text)
        return m.group(1).strip(".,;") if m else ""

    pred_treatment = _extract("treatment")
    pred_outcome   = _extract("outcome")
    gt_treatment   = str(gt_step1.get("treatment") or "").lower()
    gt_outcome     = str(gt_step1.get("outcome")   or "").lower()

    if not pred_treatment and not pred_outcome:
        return -50

    score = 0.0
    if gt_treatment and pred_treatment and gt_treatment in pred_treatment:
        score += 5
    if gt_outcome and pred_outcome and gt_outcome in pred_outcome:
        score += 5

    gt_controls = set()
    raw_controls = gt_step1.get("controls")
    if isinstance(raw_controls, list):
        gt_controls = {str(c).lower() for c in raw_controls if c}
    elif isinstance(raw_controls, str) and raw_controls:
        gt_controls = {raw_controls.lower()}

    if gt_controls:
        ctrl_m = re.search(r"controls\s*:\s*\[?([^\]\n]+)", text)
        pred_controls = set()
        if ctrl_m:
            for tok in re.split(r"[,\s]+", ctrl_m.group(1)):
                t = tok.strip("[].,;\"'")
                if t:
                    pred_controls.add(t)
        if pred_controls or gt_controls:
            jaccard = len(pred_controls & gt_controls) / len(pred_controls | gt_controls)
            score += 15 * jaccard

    _NONE_WORDS = {"none", "na", "n/a", "-", "null", ""}
    gt_special = {
        "instrument":       gt_step1.get("instrument"),
        "running_variable": gt_step1.get("running_variable"),
        "time_variable":    gt_step1.get("time_variable"),
        "group_variable":   gt_step1.get("group_variable"),
    }
    active = {k: str(v).lower() for k, v in gt_special.items() if v is not None}

    if not active:
        hallucinated = any(_extract(k).lower() not in _NONE_WORDS for k in gt_special)
        score += 5 if not hallucinated else 0
    else:
        correct = sum(1 for k, gt_v in active.items() if gt_v in _extract(k).lower())
        score += 5 if correct == len(active) else 0

    return score


def _score_causcibench(parsed: dict, gt: dict, step3_judge: int) -> float:
    total = 0.0

    # Step 1: variable identification
    total += _step1_score(parsed["step1"], gt.get("step1") or {})

    # Step 2: method exact match — 30 pts
    method_ok = parsed["step2"] == gt["step2"]
    total += 30 if method_ok else -50

    # Step 3: estimation spec — judge score → 0/7/15
    total += [0, 7, 15][step3_judge]

    # Step 5: numeric answer — relative error scoring
    gt_val   = gt.get("step5")
    pred_val = parsed.get("step5")

    if pred_val is not None and gt_val is not None:
        try:
            pred_f  = float(pred_val)
            gt_f    = float(gt_val)
            denom   = abs(gt_f) if gt_f != 0 else 1.0
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

def compute_rewards(
    completions: list[str],
    rows: list[dict],
    judge_model,
    judge_tokenizer,
) -> list[float]:
    """
    Score a batch of completions using heuristics + DeepSeek-Math judge for step 3.
    completions and rows must be the same length.
    Returns list of scalar rewards (can be negative).
    """
    parsed_list = [parse_completion(c, r["source"]) for c, r in zip(completions, rows)]

    cladder_idx = [i for i, r in enumerate(rows) if r["source"] == "cladder"]
    causci_idx  = [i for i, r in enumerate(rows) if r["source"] == "causcibench"]

    step3_judges = [0] * len(completions)

    if cladder_idx:
        predicted  = [parsed_list[i]["step3"] for i in cladder_idx]
        reference  = [rows[i]["groundtruth"].get("step3", "") for i in cladder_idx]
        scores = _judge_cladder_step3(predicted, reference, judge_model, judge_tokenizer)
        for i, s in zip(cladder_idx, scores):
            step3_judges[i] = s

    if causci_idx:
        specs   = [parsed_list[i]["step3"] for i in causci_idx]
        methods = [parsed_list[i]["step2"]  for i in causci_idx]
        step1s  = [parsed_list[i]["step1"]  for i in causci_idx]
        scores = _judge_causci_step3(specs, methods, step1s, judge_model, judge_tokenizer)
        for i, s in zip(causci_idx, scores):
            step3_judges[i] = s

    rewards = []
    for i, (row, parsed) in enumerate(zip(rows, parsed_list)):
        if row["source"] == "cladder":
            r = _score_cladder(parsed, row["groundtruth"], step3_judges[i])
        else:
            r = _score_causcibench(parsed, row["groundtruth"], step3_judges[i])
        rewards.append(float(r))

    return rewards
