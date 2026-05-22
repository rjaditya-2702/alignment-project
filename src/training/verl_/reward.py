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
"""

import re
import json
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

from src.training.tool_calling import library_fn
from src.training.eval_metrics import compute_eval_metrics, save_eval_plots
from src.config import PLOT_DIR

PLOT_DIR.mkdir(parents=True, exist_ok=True)

_call_count       = [0]
_eval_steps       = []
_eval_history     = {}
_metric_buffer    = []   # per-call metrics, flushed every LOG_WINDOW calls
_reward_buffer    = []   # per-call mean reward
_response_printed = [False]
LOG_WINDOW        = 50

# ---------------------------------------------------------------------------
# Judge client
# ---------------------------------------------------------------------------

_judge_client = OpenAI(base_url="http://localhost:8001/v1", api_key="token")
JUDGE_MODEL   = "Qwen/Qwen3-8B"  # must match what the judge server is serving

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
    return f"""
You are a causal inference expert.
Return 1 if correct, 0 if wrong. Nothing else. You have just 1 token to respond.
Predicted causal graph: {parsed.get('step1', '')}
Reference causal graph: {gt.get('step1', '')}

Does the predicted graph correctly identify:
1. The right variables
2. All directed edges in the correct direction
3. No spurious edges added
"""


def _make_step3_prompt(parsed: dict, gt: dict) -> str:
    return f"""
You are a causal inference expert.
Return 1 if equivalent, 0 if wrong. Nothing else. You have just 1 token to respond.
Query type: {gt.get('step2', '')}
Predicted estimand: {parsed.get('step3', '')}
Reference estimand: {gt.get('step3', '')}

Are these mathematically equivalent?
"""


# ---------------------------------------------------------------------------
# Scoring — unchanged from TRL
# ---------------------------------------------------------------------------

def reward_cladder_precomputed(
    prediction: dict,
    ground_truth: dict,
    s1_score: int,
    s3_score: int | None,
) -> tuple[float, dict]:
    scores = {}

    scores['step1'] = s1_score
    if scores['step1'] == 0:
        return -1.0, scores

    pred_step2 = prediction.get('step2', '').strip().lower()
    ref_step2  = ground_truth.get('step2', '').strip().lower()
    scores['step2'] = 1 if pred_step2 == ref_step2 else 0

    if scores['step2'] == 0:
        scores['step3'] = 0
        scores['step5'] = 0
        return -0.5, scores

    scores['step3']  = s3_score if s3_score is not None else 0
    step3_penalty    = 0.0 if scores['step3'] == 1 else -0.25

    pred_step5 = prediction.get('step5', '').strip().lower()
    ref_step5  = ground_truth.get('step5', '').strip().lower()
    scores['step5'] = 1 if pred_step5 == ref_step5 else 0

    if scores['step5'] == 1:
        reward = 1.0 + step3_penalty
    else:
        reward = -0.75 + step3_penalty

    return reward, scores


def reward_causci(
    prediction: dict,
    ground_truth: dict,
    library_effect: float,
) -> tuple[float, dict]:
    scores = {}

    pred_method = prediction.get('step2', '').strip().lower()
    ref_method  = ground_truth.get('step2', '').strip().lower()
    scores['method'] = 1 if pred_method == ref_method else 0

    if scores['method'] == 0:
        return -1.0, scores

    pred_treat = prediction.get('step1', {}).get('treatment', '').strip()
    ref_treat  = ground_truth.get('step1', {}).get('treatment', '').strip()
    scores['treatment'] = 1 if pred_treat == ref_treat else 0

    pred_outcome = prediction.get('step1', {}).get('outcome', '').strip()
    ref_outcome  = ground_truth.get('step1', {}).get('outcome', '').strip()
    scores['outcome'] = 1 if pred_outcome == ref_outcome else 0

    if scores['treatment'] == 0 or scores['outcome'] == 0:
        return -0.5, scores

    pred_controls = set(prediction.get('step1', {}).get('controls', []))
    ref_controls  = set(ground_truth.get('step1', {}).get('controls', []))
    if len(ref_controls) > 0:
        scores['controls'] = len(pred_controls & ref_controls) / len(ref_controls)
    else:
        scores['controls'] = 1.0 if len(pred_controls) == 0 else 0.0

    controls_good = scores['controls'] >= 0.75

    ref_effect = ground_truth.get('step5')
    if ref_effect is not None and ref_effect != 0:
        mre = abs(library_effect - ref_effect) / abs(ref_effect)
        effect_correct = mre <= 0.05
    else:
        effect_correct = False

    scores['effect'] = 1 if effect_correct else 0

    if controls_good and effect_correct:
        reward = 1.0
    elif not controls_good and effect_correct:
        reward = 0.5
    elif controls_good and not effect_correct:
        reward = -0.25
    else:
        reward = -0.25

    return reward, scores


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
    step5 = parsed.get('step5', '').strip().lower()
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

    treatment = step1.get('treatment', '').strip()
    outcome   = step1.get('outcome', '').strip()
    if treatment not in dataset_columns:
        return None
    if outcome not in dataset_columns:
        return None

    controls = [c for c in step1.get('controls', []) if c in dataset_columns]

    if method == 'iv':
        if step1.get('instrument') not in dataset_columns:
            return None
    if method == 'rdd':
        if step1.get('running_variable') not in dataset_columns:
            return None
        if step1.get('cutoff') is None:
            return None
    if method == 'did':
        if step1.get('time_variable') not in dataset_columns:
            return None
        if step1.get('group_variable') not in dataset_columns:
            return None
    if method == 'frontdoor':
        if step1.get('mediator') not in dataset_columns:
            return None

    return {
        'step1': {
            'treatment':        treatment,
            'outcome':          outcome,
            'controls':         controls,
            'instrument':       step1.get('instrument'),
            'running_variable': step1.get('running_variable'),
            'cutoff':           step1.get('cutoff'),
            'time_variable':    step1.get('time_variable'),
            'group_variable':   step1.get('group_variable'),
            'mediator':         step1.get('mediator'),
            'estimand':         step1.get('estimand', '').strip().lower(),
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
# veRL reward interface
# ---------------------------------------------------------------------------

def reward_fn(
    solution_strs: list[str],
    ground_truths: list[str],   # JSON strings: {"ground_truth": {...}}
    extra_infos:   list[dict],  # {"csv_path": str, "dataset_columns": list[str]}
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
    for solution, gt_str, extra_info in zip(solution_strs, ground_truths, extra_infos):
        gt         = json.loads(gt_str)   # veRL passes reward_model["ground_truth"] directly
        ei         = json.loads(extra_info) if isinstance(extra_info, str) else extra_info
        source     = "cladder" if ei["csv_path"] == "" else "causcibench"
        cols       = ei["dataset_columns"]
        csv_path   = ei["csv_path"]

        parsed = extract_cladder(solution) if source == "cladder" else extract_causci(solution, cols)
        items.append((source, parsed, gt, cols, csv_path))

    # Phase 2 — collect all cladder judge prompts and fire concurrently
    judge_prompts = []
    prompt_idx    = {}  # item index → {"step1": int, "step3": int}

    for i, (source, parsed, gt, cols, csv_path) in enumerate(items):
        if source != "cladder" or parsed is None:
            continue
        prompt_idx[i] = {}
        prompt_idx[i]["step1"] = len(judge_prompts)
        judge_prompts.append(_make_step1_prompt(parsed, gt))
        # step3 judge only fires if step2 is already correct — avoids a wasted call
        if parsed.get('step2', '').strip().lower() == gt.get('step2', '').strip().lower():
            prompt_idx[i]["step3"] = len(judge_prompts)
            judge_prompts.append(_make_step3_prompt(parsed, gt))

    judge_scores = batch_judge(judge_prompts) if judge_prompts else []

    # Phase 3 — score using pre-fetched judge results
    rewards = []
    for i, (source, parsed, gt, cols, csv_path) in enumerate(items):

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
            library_effect = cached_library_fn(
                csv_path                  = csv_path,
                method                    = parsed["step2"],
                treatment                 = step1["treatment"],
                outcome                   = step1["outcome"],
                controls_tuple            = tuple(step1["controls"]),
                estimand                  = step1.get("estimand"),
                instrument                = step1.get("instrument"),
                running_variable          = step1.get("running_variable"),
                cutoff                    = step1.get("cutoff"),
                time_variable             = step1.get("time_variable"),
                group_variable            = step1.get("group_variable"),
                mediator                  = step1.get("mediator"),
            ) if parsed["step2"] == gt.get("step2") else 0.0
            reward, _ = reward_causci(parsed, gt, library_effect)

        else:
            raise ValueError(f"Unknown source: {source!r}")

        rewards.append(reward)

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"[verl] call {call:5d}  reward={mean_reward:+.3f}  src={items[0][0]}", flush=True)

    # buffer metrics and reward; flush every LOG_WINDOW calls
    eval_items = [(src, parsed, gt, csv_path) for (src, parsed, gt, cols, csv_path) in items]
    _metric_buffer.append(compute_eval_metrics(eval_items))
    _reward_buffer.append(mean_reward)

    if not _response_printed[0]:
        print(f"\n[verl] sample response:\n{solution_strs[0][:500]}")
        _response_printed[0] = True

    if call % LOG_WINDOW == 0:
        all_keys = set(k for m in _metric_buffer for k in m)
        avg = {k: sum(m.get(k, 0.0) for m in _metric_buffer) / len(_metric_buffer) for k in all_keys}
        avg["reward/mean"] = sum(_reward_buffer) / len(_reward_buffer)

        _eval_steps.append(call)
        for k, v in avg.items():
            _eval_history.setdefault(k, []).append(v)

        print(f"\n[verl call {call}] avg over last {LOG_WINDOW} samples:")
        for k, v in sorted(avg.items()):
            print(f"  {k}: {v:.4f}")

        with open(PLOT_DIR / "eval_log.jsonl", "a") as f:
            f.write(json.dumps({"call": call, **avg}) + "\n")

        save_eval_plots(_eval_history, _eval_steps, PLOT_DIR)
        _metric_buffer.clear()
        _reward_buffer.clear()

    return rewards


# ---------------------------------------------------------------------------
# Per-sample fallback — for veRL's default compute_score interface if needed
# ---------------------------------------------------------------------------

def compute_score(
    data_source:  str,
    solution_str: str,
    ground_truth: str,
    extra_info:   dict,
) -> float:
    return reward_fn([solution_str], [ground_truth], [extra_info])[0]
