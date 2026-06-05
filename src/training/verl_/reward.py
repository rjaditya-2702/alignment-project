"""
reward.py — veRL reward function for causal alignment training.

Scoring logic is identical to TRL version.
Only the interface layer changes to match veRL's RewardManager contract.

veRL calls:
    reward_fn(
        solution_strs:  list[str],           # decoded completions, one per rollout
        ground_truths:  list[str],           # JSON strings from parquet ground_truth column
        extra_infos:    list[dict],           # dicts from parquet extra_info column
    ) -> list[float]

Internal flow (identical to TRL):
    Phase 1 — parse all completions
    Phase 2 — batch all cladder judge calls, fire concurrently
    Phase 3 — score with pre-fetched judge results

Logging:
    Training calls print: [verl] call {N} reward={mean:+.3f} src={src}
    Eval passes print:    [verl_eval] eval_pass:{N} key:value ...
    veRL native console lines carry step, loss, KL — parsed separately.
"""

import re
import json
import atexit
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

from src.training.tool_calling import library_fn
from src.training.eval_metrics import compute_eval_metrics

_call_count   = [0]
_eval_buffer  = []   # accumulates (src, parsed, gt, csv_path) for test-split calls
_eval_pass    = [0]  # number of eval passes completed

# ---------------------------------------------------------------------------
# Judge client
# ---------------------------------------------------------------------------

_judge_client = OpenAI(base_url="http://localhost:8001/v1", api_key="token")
JUDGE_MODEL   = "Qwen/Qwen3-8B"  # must match what the judge server is serving

def _sanitize_col(name: str) -> str:
    """Replace characters patsy treats as operators (dots, spaces, hyphens) with underscores."""
    return re.sub(r'[.\s\-]', '_', str(name))

def _judge_one(prompt: str) -> float:
    try:
        r = _judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=2,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            messages=[
                {"role": "system", "content": "You are a binary scorer. Reply with only 0 or 1. No other text."},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = re.sub(r'[^01]', '', r.choices[0].message.content.strip())
        return float(int(raw[0]))
    except Exception:
        return 0.0


def batch_judge(prompts: list[str]) -> list[float]:
    if not prompts:
        return []
    with ThreadPoolExecutor(max_workers=min(len(prompts), 16)) as pool:
        futures = [pool.submit(_judge_one, p) for p in prompts]
        return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# Judge prompt builders — unchanged from TRL
# ---------------------------------------------------------------------------

def _make_step1_prompt(parsed: dict, gt: dict) -> str:
    return f"""You are a causal inference expert evaluating graph extraction.

Reference causal graph: {gt.get('step1', '')}
Predicted causal graph: {parsed.get('step1', '')}

Rules:
- Ignore variable name formatting differences (e.g., X→Y same as X -> Y)
- Edge direction must be correct
- All reference edges must be present
- Extra edges = wrong
- Missing edges = wrong

Reply with exactly one character: 1 (correct) or 0 (wrong)."""

def _make_step3_prompt(parsed: dict, gt: dict) -> str:
    return f"""You are a causal inference expert evaluating estimand equivalence.

Causal graph: {gt.get('step1', '')}
Query type: {gt.get('step2', '')}

Reference estimand: {gt.get('step3', '')}
Predicted estimand: {parsed.get('step3', '')}

Given the causal graph and query type above, are the two estimands mathematically equivalent?
Equivalent means: same quantity being estimated, potentially expressed differently.
Count as equivalent:
- Backdoor adjustment expanded vs compact form
- Algebraic rearrangements
- Summation order differences
- Using ATE/ATT notation vs explicit do-expression when unambiguous

Count as NOT equivalent:
- Missing do-operator when intervention is required
- Wrong conditioning set (e.g., conditioning on a collider)
- Wrong rung (e.g., associational expression for a counterfactual query)
- Estimating a different variable than the reference

Reply with exactly one character: 1 (equivalent) or 0 (not equivalent)."""

# -------------------------- V1 ----------------------------------------

# def _make_step1_prompt(parsed: dict, gt: dict) -> str:
#     return f"""
# You are a causal inference expert.
# Return 1 if correct, 0 if wrong. Nothing else. You have just 1 token to respond.
# Predicted causal graph: {parsed.get('step1', '')}
# Reference causal graph: {gt.get('step1', '')}

# Does the predicted graph correctly identify:
# 1. The right variables
# 2. All directed edges in the correct direction
# 3. No spurious edges added
# """


# def _make_step3_prompt(parsed: dict, gt: dict) -> str:
#     return f"""
# You are a causal inference expert.
# Return 1 if equivalent, 0 if wrong. Nothing else. You have just 1 token to respond.
# Query type: {gt.get('step2', '')}
# Predicted estimand: {parsed.get('step3', '')}
# Reference estimand: {gt.get('step3', '')}

# Are these mathematically equivalent?
# """


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def reward_cladder_precomputed(
    prediction: dict,
    ground_truth: dict,
    s1_score: int,
    s3_score: int | None,
) -> tuple[float, dict]:

    scores = {}

    scores['step1'] = s1_score  # graph extraction
    scores['step2'] = 1 if prediction.get('step2', '').strip().lower() == \
                           ground_truth.get('step2', '').strip().lower() else 0  # query type
    scores['step3'] = s3_score if s3_score is not None else 0  # formalization
    scores['step5'] = 1 if prediction.get('step5', '').strip().lower() == \
                           ground_truth.get('step5', '').strip().lower() else 0  # final answer

    reasoning_score = (
        0.30 * scores['step1'] +
        0.40 * scores['step2'] +
        0.30 * scores['step3']
    )  # in [0, 1]

    final_correct = scores['step5']  # 0 or 1

    # Require reasoning to be nonzero to get credit for correct final answer
    # This kills the "guess and get rewarded" shortcut
    if final_correct and reasoning_score == 0.0:
        # lucky guess with no reasoning — penalize
        reward = -0.5
    elif final_correct and reasoning_score > 0.0:
        # correct answer with some reasoning — scale by reasoning quality
        reward = 0.5 + 0.5 * reasoning_score  # in [0.5, 1.0]
    elif not final_correct and reasoning_score > 0.5:
        # good reasoning but wrong answer — small positive (likely a hard question)
        reward = 0.1 * reasoning_score
    else:
        # wrong answer, poor reasoning
        reward = -1.0 + reasoning_score  # in [-1.0, -0.7] depending on partial credit

    return reward, scores

# -------------------------------- V2 -------------------------------

# def reward_cladder_precomputed(
#     prediction: dict,
#     ground_truth: dict,
#     s1_score: int,
#     s3_score: int | None,
# ) -> tuple[float, dict]:
    
#     scores = {}
    
#     scores['step1'] = s1_score  # 0 or 1
#     scores['step2'] = 1 if prediction.get('step2','').strip().lower() == ground_truth.get('step2','').strip().lower() else 0
#     scores['step3'] = s3_score if s3_score is not None else 0
#     scores['step5'] = 1 if prediction.get('step5','').strip().lower() == ground_truth.get('step5','').strip().lower() else 0
    
#     # weighted additive — always produces variance across rollouts
#     reward = (
#         0.15 * scores['step1'] +
#         0.25 * scores['step2'] +
#         0.10 * scores['step3'] +
#         0.50 * scores['step5']
#     )
#     # shift to [-1, +1]
#     reward = 2 * reward - 1
    
#     return reward, scores

# ------------------------------- V1 --------------------------------

# def reward_cladder_precomputed(
#     prediction: dict,
#     ground_truth: dict,
#     s1_score: int,
#     s3_score: int | None,
# ) -> tuple[float, dict]:
#     scores = {}

#     scores['step1'] = s1_score
#     if scores['step1'] == 0:
#         return -1.0, scores

#     pred_step2 = prediction.get('step2', '').strip().lower()
#     ref_step2  = ground_truth.get('step2', '').strip().lower()
#     scores['step2'] = 1 if pred_step2 == ref_step2 else 0

#     if scores['step2'] == 0:
#         scores['step3'] = 0
#         scores['step5'] = 0
#         return -0.5, scores

#     scores['step3']  = s3_score if s3_score is not None else 0
#     step3_penalty    = 0.0 if scores['step3'] == 1 else -0.25

#     pred_step5 = prediction.get('step5', '').strip().lower()
#     ref_step5  = ground_truth.get('step5', '').strip().lower()
#     scores['step5'] = 1 if pred_step5 == ref_step5 else 0

#     if scores['step5'] == 1:
#         reward = 1.0 + step3_penalty
#     else:
#         reward = -0.75 + step3_penalty

#     return reward, scores

def reward_causci(prediction, ground_truth, library_effect, library_success):
    scores = {}
    
    # 1. method (exact match)
    p = prediction.get('step2', '')
    if p is not None:
        p = p.strip().lower()
    r = ground_truth.get('step2', '')
    if r is not None:
        r = r.strip().lower()
    scores['method'] = int(
        p == r
    )
    
    # 2. treatment / outcome (exact match, normalized)
    for k in ['treatment', 'outcome']:
        p = prediction['step1']
        if p is not None:
            p = p.get(k, '')
        if p is not None:
            p = p.strip().lower()
        r = ground_truth['step1']
        if r is not None:
            r = r.get(k, '')
        if r is not None:
            r = r.strip().lower()
        scores[k] = int(p == r)
    
    # 3. controls (Jaccard-style, matches VSA)
    prediction_step1 = prediction.get('step1', {})
    if prediction_step1 is None:
        prediction_step1 = {}
    ground_truth_step1 = ground_truth.get('step1', {})
    if ground_truth_step1 is None:
        ground_truth_step1 = {}
    pc = {_sanitize_col(c.strip().lower()) for c in (prediction_step1.get('controls') or [])}
    rc = {_sanitize_col(c.strip().lower()) for c in (ground_truth_step1.get('controls') or [])}
    if rc:
        scores['controls'] = len(pc & rc) / len(rc)
    else:
        scores['controls'] = 1.0 if not pc else 0.0
    
    # 4. effect (paper's EA, with zero-handling)
    ref = ground_truth.get('step5')
    if not library_success or ref is None:
        scores['effect'] = 0.0
    elif abs(ref) < 1e-6:
        scores['effect'] = float(abs(library_effect) < 1e-3)
    else:
        mre = abs(library_effect - ref) / abs(ref)
        scores['effect'] = float(mre <= 0.05)
    
    # weighted linear combination
    reward = (
        0.30 * scores['method']
        + 0.15 * scores['treatment']
        + 0.10 * scores['outcome']
        + 0.15 * scores['controls']
        + 0.30 * scores['effect']
    ) * 2 - 1  # rescale [0,1] -> [-1,1]
    
    return reward, scores

# ------------------------------- V1 --------------------------------

# def reward_causci(prediction: dict, ground_truth: dict, library_effect: float, ) -> tuple[float, dict]:
#     scores = {}
#     pred_method = prediction.get('step2', '').strip().lower()
#     ref_method  = ground_truth.get('step2', '')
#     if ref_method is not None:
#         ref_method = ref_method.strip().lower()
#     scores['method'] = 1 if pred_method == ref_method else 0

#     if scores['method'] == 0:
#         return -1.0, scores

#     pred_treat = prediction.get('step1', {}).get('treatment', '').strip()
#     ref_treat  = ground_truth.get('step1', {}).get('treatment', '').strip()
#     if ref_treat is not None:
#         ref_treat = ref_treat.strip()
#     scores['treatment'] = 1 if pred_treat == ref_treat else 0

#     pred_outcome = prediction.get('step1', {}).get('outcome', '').strip()
#     ref_outcome  = ground_truth.get('step1', {}).get('outcome', '')
#     if ref_outcome is not None:
#         ref_outcome = ref_outcome.strip()
#     scores['outcome'] = 1 if pred_outcome == ref_outcome else 0

#     if scores['treatment'] == 0 or scores['outcome'] == 0:
#         return -0.5, scores

#     pred_controls = set(prediction.get('step1', {}).get('controls') or [])
#     ref_controls  = set(ground_truth.get('step1', {}).get('controls') or [])
#     if len(ref_controls) > 0:
#         scores['controls'] = len(pred_controls & ref_controls) / len(ref_controls)
#     else:
#         scores['controls'] = 1.0 if len(pred_controls) == 0 else 0.0

#     controls_good = scores['controls'] >= 0.75

#     ref_effect = ground_truth.get('step5')
#     if ref_effect is not None and ref_effect != 0:
#         mre = abs(library_effect - ref_effect) / abs(ref_effect)
#         effect_correct = mre <= 0.05
#     else:
#         effect_correct = False

#     scores['effect'] = 1 if effect_correct else 0

#     if controls_good and effect_correct:
#         reward = 1.0
#     elif not controls_good and effect_correct:
#         reward = 0.5
#     elif controls_good and not effect_correct:
#         reward = -0.25
#     else:
#         reward = -0.25

#     return reward, scores


# ---------------------------------------------------------------------------
# Extraction — unchanged from TRL
# ---------------------------------------------------------------------------

def extract_json(model_output: str) -> dict | None:
    if '</think>' in model_output:
        model_output = model_output.split('</think>')[-1]

    start = model_output.find('{')
    end   = model_output.rfind('}')

    if start == -1 or end == -1:
        return None

    json_str = model_output[start:end+1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        try:
            return json.loads(json_str)
        except Exception:
            return None


CLADDER_REQUIRED    = {'step1', 'step2', 'step3', 'step4', 'step5'}
CLADDER_QUERY_TYPES = {
    'marginal', 'correlation', 'ate', 'backadj',
    'det-counterfactual', 'ett', 'nde', 'nie',
    'collider_bias', 'exp_away'
}
CAUSCI_METHODS = {
    'diff_in_means', 'ols', 'ipw', 'matching',
    'did', 'rdd', 'iv', 'frontdoor', 'glm'
}


def extract_cladder(model_output: str) -> dict | None:
    parsed = extract_json(model_output)
    if parsed is None:
        return None
    if not CLADDER_REQUIRED.issubset(parsed.keys()):
        return None
    step2 = parsed.get('step2', '').strip().lower()
    if step2 not in CLADDER_QUERY_TYPES:
        return None
    step5 = parsed.get('step5', '')
    if isinstance(step5, (int, float)):
        step5 = 'yes' if abs(step5) > 0 else 'no'
    else:
        step5 = step5.strip().lower()
    if step5 not in {'yes', 'no'}:
        return None
    return {
        'step1': str(parsed['step1']),
        'step2': step2,
        'step3': str(parsed['step3']),
        'step4': str(parsed['step4']),
        'step5': step5,
    }


def extract_causci(model_output: str, dataset_columns: list[str]) -> dict | None:
    parsed = extract_json(model_output)
    if parsed is None:
        return None
    if 'step1' not in parsed or 'step2' not in parsed:
        return None

    step1  = parsed['step1']
    method = parsed.get('step2', '').strip().lower()
    if method not in CAUSCI_METHODS:
        return None

    def _str_field(val):
        if isinstance(val, list):
            val = val[0] if val else ''
        return (val or '').strip()

    treatment = _str_field(step1.get('treatment'))
    outcome   = _str_field(step1.get('outcome'))
    if treatment not in dataset_columns:
        return None
    if outcome not in dataset_columns:
        return None

    controls = [c for c in (step1.get('controls') or []) if c in dataset_columns]

    if method == 'iv':
        if _str_field(step1.get('instrument')) not in dataset_columns:
            return None
    if method == 'rdd':
        if _str_field(step1.get('running_variable')) not in dataset_columns:
            return None
        if step1.get('cutoff') is None:
            return None
    if method == 'did':
        if _str_field(step1.get('time_variable')) not in dataset_columns:
            return None
        if _str_field(step1.get('group_variable')) not in dataset_columns:
            return None
    if method == 'frontdoor':
        if _str_field(step1.get('mediator')) not in dataset_columns:
            return None

    return {
        'step1': {
            'treatment':        treatment,
            'outcome':          outcome,
            'controls':         controls,
            'instrument':       _str_field(step1.get('instrument')),
            'running_variable': _str_field(step1.get('running_variable')),
            'cutoff':           step1.get('cutoff'),
            'time_variable':    _str_field(step1.get('time_variable')),
            'group_variable':   _str_field(step1.get('group_variable')),
            'mediator':         _str_field(step1.get('mediator')),
            'estimand':         _str_field(step1.get('estimand')).lower(),
        },
        'step2': method,
    }


# ---------------------------------------------------------------------------
# library_fn cache — unchanged from TRL
# ---------------------------------------------------------------------------
@lru_cache(maxsize=512)
def cached_library_fn(
    csv_path, method, treatment, outcome, controls_tuple,
    estimand=None, instrument=None, running_variable=None,
    cutoff=None, time_variable=None, group_variable=None, mediator=None,
):
    return library_fn({
        "step1": {
            "csv_path":        csv_path,
            "treatment":       treatment,
            "outcome":         outcome,
            "controls":        list(controls_tuple),
            "estimand":        estimand,
            "instrument":      instrument,
            "running_variable": running_variable,
            "cutoff":          cutoff,
            "time_variable":   time_variable,
            "group_variable":  group_variable,
            "mediator":        mediator,
        },
        "step2": method,
    })

# ---------------------------------------------------------------------------
# Eval flush
# ---------------------------------------------------------------------------

def _flush_eval_buffer():
    """Compute and log eval metrics from accumulated test-split samples."""
    if not _eval_buffer:
        return
    _eval_pass[0] += 1
    metrics = compute_eval_metrics(_eval_buffer)
    parts = " ".join(f"{k}:{v:.4f}" for k, v in sorted(metrics.items()))
    print(f"[verl_eval] eval_pass:{_eval_pass[0]} {parts}", flush=True)
    _eval_buffer.clear()

# Flush any remaining eval samples when the process exits (e.g. training ends
# immediately after an eval pass with no subsequent training call to trigger flush).
atexit.register(_flush_eval_buffer)

# ---------------------------------------------------------------------------
# veRL reward interface
# ---------------------------------------------------------------------------
import time

def reward_fn(
    solution_strs: list[str],
    ground_truths: list[str],   # JSON strings: {"ground_truth": {...}}
    extra_infos:   list[dict],  # {"csv_path": str, "dataset_columns": list[str], "split": str}
) -> list[float]:
    """
    veRL RewardManager entry point.

    solution_strs  — decoded completions, one per rollout sample
    ground_truths  — JSON strings from parquet ground_truth column
    extra_infos    — dicts from parquet extra_info column
    """

    _call_count[0] += 1
    call = _call_count[0]

    # Phase 1 — unpack and parse all completions
    items = []
    t0 = time.time()
    for solution, gt_str, extra_info in zip(solution_strs, ground_truths, extra_infos):
        gt         = json.loads(gt_str)
        ei         = json.loads(extra_info) if isinstance(extra_info, str) else extra_info
        source     = "cladder" if ei["csv_path"] == "" else "causcibench"
        cols       = ei["dataset_columns"]
        csv_path   = ei["csv_path"]
        split      = ei.get("split", "train")

        parsed = extract_cladder(solution) if source == "cladder" else extract_causci(solution, cols)
        items.append((source, parsed, gt, cols, csv_path, split))

    # Phase 2 — collect all cladder judge prompts and fire concurrently
    judge_prompts = []
    prompt_idx    = {}  # item index → {"step1": int, "step3": int}

    for i, (source, parsed, gt, cols, csv_path, split) in enumerate(items):
        if source != "cladder" or parsed is None:
            continue
        prompt_idx[i] = {}
        prompt_idx[i]["step1"] = len(judge_prompts)
        judge_prompts.append(_make_step1_prompt(parsed, gt))
        if parsed.get('step2', '').strip().lower() == gt.get('step2', '').strip().lower():
            prompt_idx[i]["step3"] = len(judge_prompts)
            judge_prompts.append(_make_step3_prompt(parsed, gt))

    judge_scores = batch_judge(judge_prompts) if judge_prompts else []

    # Phase 3 — score using pre-fetched judge results
    rewards = []
    for i, (source, parsed, gt, cols, csv_path, split) in enumerate(items):

        if source == "cladder":
            if parsed is None:
                rewards.append(-1.0)
                continue
            idxs = prompt_idx.get(i, {})
            s1   = int(round(judge_scores[idxs["step1"]])) if "step1" in idxs else 0
            s3   = int(round(judge_scores[idxs["step3"]])) if "step3" in idxs else None
            reward, _ = reward_cladder_precomputed(parsed, gt, s1, s3)

        elif source == "causcibench":
            if parsed is None:
                rewards.append(-1.0)
                continue
            step1          = parsed["step1"]
            library_effect, library_success = cached_library_fn(
                csv_path                  = csv_path,
                method                    = parsed["step2"],
                treatment                 = step1["treatment"],
                outcome                   = step1["outcome"],
                controls_tuple            = tuple(step1.get("controls") or []),
                estimand                  = step1.get("estimand"),
                instrument                = step1.get("instrument"),
                running_variable          = step1.get("running_variable"),
                cutoff                    = step1.get("cutoff"),
                time_variable             = step1.get("time_variable"),
                group_variable            = step1.get("group_variable"),
                mediator                  = step1.get("mediator"),
            ) if parsed["step2"] == gt.get("step2") else (0.0, False)
            reward, _ = reward_causci(parsed, gt, library_effect, library_success)

        else:
            raise ValueError(f"Unknown source: {source!r}")

        rewards.append(reward)

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    split_tag   = items[0][5] if items else "train"

    # If switching from eval back to train, flush accumulated eval buffer
    if split_tag == "train" and _eval_buffer:
        _flush_eval_buffer()

    if split_tag == "test":
        print(f"[verl_eval] eval_pass:{_eval_pass[0]} call:{call:5d} reward={mean_reward:+.3f} src={items[0][0]}", flush=True)
        eval_items = [(src, parsed, gt, csv_path) for (src, parsed, gt, cols, csv_path, split) in items]
        _eval_buffer.extend(eval_items)
    else:
        print(f"[verl] call {call:5d}  reward={mean_reward:+.3f}  src={items[0][0]}", flush=True)
    print(f"[reward] n={len(rewards)} dt={time.time()-t0:.2f}s", flush=True)
    return rewards


# ---------------------------------------------------------------------------
# Batch reward manager — patches NaiveRewardManager to process the whole
# batch at once so all CLaDDer judge calls fire concurrently via batch_judge.
#
# Timing: veRL loads this file (exec_module) BEFORE instantiating
# NaiveRewardManager in the same Ray actor process, so the patch is in place
# when the manager is created.  compute_score still has to exist (veRL loads
# it by name), but the patched __call__ bypasses it entirely.
# ---------------------------------------------------------------------------

def _install_batch_reward_manager() -> None:
    try:
        import torch
        import verl.workers.reward_manager.naive as _naive

        def _batch_call(self_rm, data, return_dict=False):
            responses = data.batch['responses']          # [N, max_response_len]
            n = responses.shape[0]

            # Decode all completions
            if 'response_length' in data.batch:
                resp_lens = [int(data.batch['response_length'][i]) for i in range(n)]
                solution_strs = [
                    self_rm.tokenizer.decode(
                        responses[i, :resp_lens[i]], skip_special_tokens=True
                    )
                    for i in range(n)
                ]
            else:
                pad = self_rm.tokenizer.pad_token_id
                solution_strs, resp_lens = [], []
                for i in range(n):
                    tokens = responses[i]
                    valid  = tokens[tokens != pad]
                    resp_lens.append(len(valid))
                    solution_strs.append(
                        self_rm.tokenizer.decode(valid, skip_special_tokens=True)
                    )

            rm_batch      = data.non_tensor_batch['reward_model']
            ground_truths = [rm_batch[i]['ground_truth'] for i in range(n)]
            ei_batch      = data.non_tensor_batch.get('extra_info')
            extra_infos   = [ei_batch[i] for i in range(n)] if ei_batch is not None else [{} for _ in range(n)]

            # One call — concurrent judge calls via batch_judge
            scores = reward_fn(solution_strs, ground_truths, extra_infos)

            # Place scalar reward at last valid token (veRL GRPO convention)
            reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
            for i, (score, length) in enumerate(zip(scores, resp_lens)):
                if length > 0:
                    reward_tensor[i, length - 1] = float(score)

            if return_dict:
                return {'reward_tensor': reward_tensor}
            return reward_tensor

        _naive.NaiveRewardManager.__call__ = _batch_call
        print('[reward] BatchRewardManager installed — judge calls now concurrent', flush=True)

    except Exception as e:
        print(f'[reward] BatchRewardManager install skipped ({e}), per-sample fallback active', flush=True)


_install_batch_reward_manager()


# ---------------------------------------------------------------------------
# Per-sample fallback — loaded by name via custom_reward_function.name.
# With BatchRewardManager active this is never called; it exists only so
# veRL can find the symbol and construct NaiveRewardManager(compute_score).
# ---------------------------------------------------------------------------

def compute_score(
    data_source:  str,
    solution_str: str,
    ground_truth: str,
    extra_info:   dict,
) -> float:
    return reward_fn([solution_str], [ground_truth], [extra_info])[0]
