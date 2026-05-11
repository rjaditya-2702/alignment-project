import re
import sys
import torch
import random
import numpy
# import pandas
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import JUDGE_MODEL
from src.eval.parser import parse_completion

# set seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
numpy.random.seed(SEED)
# pandas.util.testing.rands.seed(SEED)
# any other libraries with randomness should be seeded here as well

CLADDER_S1_SYSTEM = """You are a causal graph evaluator.
Given a reference causal structure and a predicted causal structure, output 1 if both the variable mappings and directed edges are fully correct, 0 otherwise.
Output 0 or 1 only."""

CLADDER_S1_USER = """Reference:
{reference}

Predicted:
{predicted}

Score:"""


CLADDER_S5_SYSTEM = """You are a causal reasoning evaluator.
Given a ground truth answer and a model response, output 1 if the final answer matches the ground truth AND the reasoning chain (estimand derivation and arithmetic) is fully correct, 0 otherwise.
Output 0 or 1 only."""

CLADDER_S5_USER = """Ground truth answer: {gt_answer}

Model response:
{predicted}

Score:"""


CAUSCI_S1_SYSTEM = """You are a causal inference evaluator.
Given a reference variable specification and a predicted variable specification, output 1 if treatment, outcome, controls, and any special variable (instrument, running variable, etc.) are all correctly identified, 0 otherwise.
Output 0 or 1 only."""

CAUSCI_S1_USER = """Reference:
{reference}

Predicted:
{predicted}

Score:"""


CAUSCI_S5_SYSTEM = """You are a causal inference evaluator.
Given a causal method, a ground truth effect estimate, and a model response, output 1 if the estimation procedure correctly follows the method AND the final answer is within 5% of the ground truth, 0 otherwise.
Output 0 or 1 only."""

CAUSCI_S5_USER = """Method: {method}
Ground truth effect: {gt_answer}

Model response:
{predicted}

Score:"""

def _format_judge_prompts(system: str, user_messages: list[str]) -> list[str]:
    prompts = []
    for user_msg in user_messages:
        prompts.append((system, user_msg))

        # if hasattr(tokenizer, "apply_chat_template"):
        #     prompt = tokenizer.apply_chat_template(
        #         messages, tokenize=False, add_generation_prompt=True
        #     )
        # else:
        #     prompt = f"System: {system}\nUser: {user_msg}\nAssistant:"
        # prompts.append(prompt)
    return prompts


# def _run_judge_batch(prompts: list[str], judge_model, judge_tokenizer) -> list[int]:
#     """Run judge on a batch of prompts. Returns list of scores 0/1/2."""
#     inputs = judge_tokenizer(
#         prompts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=512,
#     ).to(judge_model.device)

#     with torch.no_grad():
#         out = judge_model.generate(
#             **inputs,
#             max_new_tokens=8,
#             do_sample=False,
#             pad_token_id=judge_tokenizer.pad_token_id,
#         )

#     prompt_len = inputs["input_ids"].shape[1]
#     scores = []
#     for o in out:
#         text = judge_tokenizer.decode(o[prompt_len:], skip_special_tokens=True).strip()
#         try:
#             val = int(text[0])
#             scores.append(min(max(val, 0), 2))
#         except (ValueError, IndexError):
#             scores.append(0)
#     return scores


# def _judge_cladder_step3(
#     predicted_list: list[str],
#     reference_list: list[str],
#     judge_model,
#     judge_tokenizer,
# ) -> list[int]:
#     user_msgs = [
#         f"Reference: {ref}\nPredicted: {pred}\nScore:"
#         for pred, ref in zip(predicted_list, reference_list)
#     ]
#     prompts = _format_judge_prompts(CLADDER_JUDGE_SYSTEM, user_msgs, judge_tokenizer)
#     return _run_judge_batch(prompts, judge_model, judge_tokenizer)


# def _judge_causci_step3(
#     spec_list: list[str],
#     method_list: list[str],
#     step1_list: list[str],
#     judge_model,
#     judge_tokenizer,
# ) -> list[int]:
#     user_msgs = [
#         f"Method: {method}\nVariables identified: {step1}\nSpecification: {spec}\nScore:"
#         for spec, method, step1 in zip(spec_list, method_list, step1_list)
#     ]
#     prompts = _format_judge_prompts(CAUSCI_JUDGE_SYSTEM, user_msgs, judge_tokenizer)
#     return _run_judge_batch(prompts, judge_model, judge_tokenizer)


# ── CLadder ───────────────────────────────────────────────────────────────────

# def _score_cladder(parsed: dict, gt: dict, step3_judge: int) -> float:
#     total = 0.0

#     # Step 1: causal graph — needs at least one arrow
#     step1_ok = "->" in parsed["step1"] or "→" in parsed["step1"]
#     total += 11 if step1_ok else -100

#     # Step 2: query type exact match
#     step2_ok = parsed["step2"] == gt["step2"]
#     total += 15 if step2_ok else -100

#     # Step 3: estimand — cascade from step 2; judge score → 0/12/24
#     if not step2_ok:
#         total += -100
#     else:
#         total += [0, 12, 24][step3_judge]

#     # Step 5: final answer yes/no
#     gt_ans = str(gt.get("step5", "")).lower().strip()
#     total += 20 if parsed["step5"] == gt_ans else -100

#     return total


# ── CauSciBench ───────────────────────────────────────────────────────────────

# def _step1_score(parsed_step1: str, gt_step1: dict) -> float:
#     text = parsed_step1.lower()

#     def _extract(label: str) -> str:
#         m = re.search(rf"{label}\s*:\s*(\S+)", text)
#         return m.group(1).strip(".,;") if m else ""

#     pred_treatment = _extract("treatment")
#     pred_outcome   = _extract("outcome")
#     gt_treatment   = str(gt_step1.get("treatment") or "").lower()
#     gt_outcome     = str(gt_step1.get("outcome")   or "").lower()

#     if not pred_treatment and not pred_outcome:
#         return -50

#     score = 0.0
#     if gt_treatment and pred_treatment and gt_treatment in pred_treatment:
#         score += 5
#     if gt_outcome and pred_outcome and gt_outcome in pred_outcome:
#         score += 5

#     gt_controls = set()
#     raw_controls = gt_step1.get("controls")
#     if isinstance(raw_controls, list):
#         gt_controls = {str(c).lower() for c in raw_controls if c}
#     elif isinstance(raw_controls, str) and raw_controls:
#         gt_controls = {raw_controls.lower()}

#     if gt_controls:
#         ctrl_m = re.search(r"controls\s*:\s*\[?([^\]\n]+)", text)
#         pred_controls = set()
#         if ctrl_m:
#             for tok in re.split(r"[,\s]+", ctrl_m.group(1)):
#                 t = tok.strip("[].,;\"'")
#                 if t:
#                     pred_controls.add(t)
#         if pred_controls or gt_controls:
#             jaccard = len(pred_controls & gt_controls) / len(pred_controls | gt_controls)
#             score += 15 * jaccard

#     _NONE_WORDS = {"none", "na", "n/a", "-", "null", ""}
#     gt_special = {
#         "instrument":       gt_step1.get("instrument"),
#         "running_variable": gt_step1.get("running_variable"),
#         "time_variable":    gt_step1.get("time_variable"),
#         "group_variable":   gt_step1.get("group_variable"),
#     }
#     active = {k: str(v).lower() for k, v in gt_special.items() if v is not None}

#     if not active:
#         hallucinated = any(_extract(k).lower() not in _NONE_WORDS for k in gt_special)
#         score += 5 if not hallucinated else 0
#     else:
#         correct = sum(1 for k, gt_v in active.items() if gt_v in _extract(k).lower())
#         score += 5 if correct == len(active) else 0

#     return score


# def _score_causcibench(parsed: dict, gt: dict, step3_judge: int) -> float:
#     total = 0.0

#     # Step 1: variable identification
#     total += _step1_score(parsed["step1"], gt.get("step1") or {})

#     # Step 2: method exact match — 30 pts
#     method_ok = parsed["step2"] == gt["step2"]
#     total += 30 if method_ok else -50

#     # Step 3: estimation spec — judge score → 0/7/15
#     total += [0, 7, 15][step3_judge]

#     # Step 5: numeric answer — relative error scoring
#     gt_val   = gt.get("step5")
#     pred_val = parsed.get("step5")

#     if pred_val is not None and gt_val is not None:
#         try:
#             pred_f  = float(pred_val)
#             gt_f    = float(gt_val)
#             denom   = abs(gt_f) if gt_f != 0 else 1.0
#             rel_err = abs(pred_f - gt_f) / denom
#             if rel_err <= 0.10:
#                 total += 25
#             elif rel_err <= 0.25:
#                 total += 15
#             elif rel_err <= 0.50:
#                 total += 5
#             else:
#                 total += -50
#         except (ValueError, TypeError):
#             total += -50
#     else:
#         total += -50

#     return total

def compute_rewards(completions, rows, judge_input_queue, judge_output_queue, judge_process, ground_truth_key="groundtruth") -> list[float]:
    """
    Reward functions for GRPO training.

    CLadder (cascading -100 penalty):
        s1: judge  → 0 or 1 × 11
        s2: exact  → 0 or 1 × 15  |  s2==0 → s3=-100 cascade
        s3: exact  → 0 or 1 × 24  |  s3==-100 → s5=-100 cascade
        s5: judge  → 0 or 1 × 20

    CauSciBench:
        s1: judge  → 0 or 1 × 30
        s2: exact  → 0 or 1 × 30  |  s2==0 → s3=0 (no cascade)
        s3: exact  → 0 or 1 × 15  |  s3==0 → s5=-100
        s5: judge  → 0 or 1 × 25
    """
    
    def _run_vllm(prompts):
        """Send prompts to judge process, get back normalized scores."""
        judge_input_queue.put(prompts)
        scores = judge_output_queue.get()  # already a list of floats/ints

        # z-score normalize
        mean = sum(scores) / len(scores) if scores else 0.0
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5 if scores else 1.0
        normalized = [(s - mean) / (std + 1e-8) for s in scores]
        return normalized

    def _exact(pred, ref):
        return 1 if pred.strip().lower() == ref.strip().lower() else 0

    # 1. Parse each rollout into steps using ## Step N headers
    parsed_list = [parse_completion(c, r["source"]) for c, r in zip(completions, rows)]

    cladder_idx = [i for i, r in enumerate(rows) if r["source"] == "cladder"]
    causci_idx  = [i for i, r in enumerate(rows) if r["source"] == "causcibench"]

    s1 = [0.0] * len(completions)
    s2 = [0.0] * len(completions)
    s3 = [0.0] * len(completions)
    s5 = [0.0] * len(completions)

    # ── CLadder ───────────────────────────────────────────────────────────────────
    if cladder_idx:
        c_gt = [rows[i][ground_truth_key] for i in cladder_idx]

        # s1: judge binary × 11
        s1_prompts = _format_judge_prompts(
            CLADDER_S1_SYSTEM,
            [CLADDER_S1_USER.format(reference=c_gt[k].get("step1", ""), predicted=parsed_list[i]["step1"])
             for k, i in enumerate(cladder_idx)],
            # tokenizer,
        )
        for i, score in zip(cladder_idx, _run_vllm(s1_prompts)):
            s1[i] = score * 11

        # s2: exact match × 15
        for k, i in enumerate(cladder_idx):
            s2[i] = _exact(parsed_list[i]["step2"], c_gt[k].get("step2", "")) * 15

        # s3: cascade -100 if s2==0, else exact match × 24
        for k, i in enumerate(cladder_idx):
            if s2[i] == 0:
                s3[i] = -100.0
            else:
                s3[i] = _exact(parsed_list[i]["step3"], c_gt[k].get("step3", "")) * 24

        # s5: cascade -100 if s3==-100, else judge binary × 20
        s5_active = [i for i in cladder_idx if s3[i] != -100.0]
        for i in cladder_idx:
            if s3[i] == -100.0:
                s5[i] = -100.0
        if s5_active:
            idx_to_k = {i: k for k, i in enumerate(cladder_idx)}
            s5_prompts = _format_judge_prompts(
                CLADDER_S5_SYSTEM,
                [CLADDER_S5_USER.format(gt_answer=c_gt[idx_to_k[i]].get("step5", ""), predicted=completions[i])
                 for i in s5_active],
                # tokenizer,
            )
            for i, score in zip(s5_active, _run_vllm(s5_prompts)):
                s5[i] = score * 20

    # ── CauSciBench ───────────────────────────────────────────────────────────────
    if causci_idx:
        ci_gt = [rows[i][ground_truth_key] for i in causci_idx]

        # s1: judge binary × 30
        s1_prompts = _format_judge_prompts(
            CAUSCI_S1_SYSTEM,
            [CAUSCI_S1_USER.format(reference=str(ci_gt[k].get("step1", "")), predicted=parsed_list[i]["step1"])
             for k, i in enumerate(causci_idx)],
            # tokenizer,
        )
        for i, score in zip(causci_idx, _run_vllm(s1_prompts)):
            s1[i] = score * 30

        # s2: exact match × 30
        for k, i in enumerate(causci_idx):
            s2[i] = _exact(parsed_list[i]["step2"], ci_gt[k].get("step2", "")) * 30

        # s3: 0 if s2==0 (no cascade), else exact match × 15
        for k, i in enumerate(causci_idx):
            if s2[i] == 0:
                s3[i] = 0.0
            else:
                s3[i] = _exact(parsed_list[i]["step3"], str(ci_gt[k].get("step3", ""))) * 15

        # s5: -100 if s3==0, else judge binary × 25
        s5_active = [i for i in causci_idx if s3[i] > 0]
        for i in causci_idx:
            if s3[i] == 0.0:
                s5[i] = -100.0
        if s5_active:
            idx_to_k = {i: k for k, i in enumerate(causci_idx)}
            s5_prompts = _format_judge_prompts(
                CAUSCI_S5_SYSTEM,
                [CAUSCI_S5_USER.format(
                    method=parsed_list[i]["step2"],
                    gt_answer=ci_gt[idx_to_k[i]].get("step5", ""),
                    predicted=completions[i],
                ) for i in s5_active],
                # tokenizer,
            )
            for i, score in zip(s5_active, _run_vllm(s5_prompts)):
                s5[i] = score * 25

    rewards = [s1[i] + s2[i] + s3[i] + s5[i] for i in range(len(completions))]
    return rewards