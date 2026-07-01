"""reward.py — veRL reward function for CauSciBench RL (GRPO).

veRL calls:
    reward_fn(solution_strs, ground_truths, extra_infos) -> list[float]
        solution_strs  — decoded completions, one per rollout
        ground_truths  — JSON strings from the parquet reward_model.ground_truth column
        extra_infos    — dicts from the parquet extra_info column
                         {"csv_path", "dataset_columns", "split", "id"}

Per completion: parse the JSON, run the chosen estimator via library_fn, and score
method / treatment / outcome / controls / effect (see reward_causci). On the test split,
samples are buffered and flushed through compute_eval_metrics → [verl_eval] log lines.
"""

import atexit
import json
import math
import os
import re
import sys
import time
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make new_code modules importable

from library import library_fn
from eval_metrics import compute_eval_metrics, _norm_method

_call_count  = [0]
_eval_buffer = []   # accumulates (parsed, gt, csv_path) for test-split calls
_eval_pass   = [0]  # number of eval passes completed

CAUSCI_METHODS = {"diff_in_means", "ols", "ipw", "matching", "did", "rdd", "iv", "frontdoor", "glm"}


def _sanitize_col(name: str) -> str:
    """Replace characters patsy treats as operators (dots, spaces, hyphens) with underscores."""
    return re.sub(r"[.\s\-]", "_", str(name))


# ── Scoring ─────────────────────────────────────────────────────────────

def _effect_score(mre: float, k: float = 2.0) -> float:
    # accuracy in log-MRE space: ~1 for small error, ~0 for large; smooth across many orders
    u = math.log10(max(mre, 1e-9))
    return 1.0 / (1.0 + math.exp(k * u))


def reward_causci(prediction, ground_truth, library_effect, library_success):
    scores = {}

    # 1. method — bucketed match (ipw≡matching, diff_in_means≡ols, …), consistent with eval
    pm, gm = _norm_method(prediction.get("step2")), _norm_method(ground_truth.get("step2"))
    scores["method"] = int(pm is not None and pm == gm)

    # 2. treatment / outcome — exact match, normalized
    for k in ("treatment", "outcome"):
        p = str((prediction.get("step1") or {}).get(k) or "").strip().lower()
        r = str((ground_truth.get("step1") or {}).get(k) or "").strip().lower()
        scores[k] = int(p == r)

    # 3. controls — Jaccard (penalises spurious extras, unlike pure coverage)
    pc = {_sanitize_col(c.strip().lower()) for c in ((prediction.get("step1") or {}).get("controls") or [])}
    rc = {_sanitize_col(c.strip().lower()) for c in ((ground_truth.get("step1") or {}).get("controls") or [])}
    scores["controls"] = 1.0 if not pc and not rc else len(pc & rc) / len(pc | rc)

    # 4. effect — accuracy vs reference; zero-effect handled separately
    ref = ground_truth.get("step5")
    if not library_success or ref is None:
        scores["effect"] = 0.0
    elif abs(ref) < 1e-6:
        scores["effect"] = float(abs(library_effect) < 1e-3)
    else:
        scores["effect"] = _effect_score(abs(library_effect - ref) / abs(ref))

    # weighted combo — effect is co-primary (only term that needs real causal reasoning)
    reward = (
        0.25 * scores["method"]
        + 0.10 * scores["treatment"]
        + 0.10 * scores["outcome"]
        + 0.15 * scores["controls"]
        + 0.40 * scores["effect"]
    ) * 2 - 1  # rescale [0,1] -> [-1,1]

    return reward, scores


# ── Extraction ──────────────────────────────────────────────────────────

def extract_json(model_output: str) -> dict | None:
    if "</think>" in model_output:
        model_output = model_output.split("</think>")[-1]
    start, end = model_output.find("{"), model_output.rfind("}")
    if start == -1 or end == -1:
        return None
    js = model_output[start:end + 1]
    try:
        return json.loads(js)
    except json.JSONDecodeError:
        js = re.sub(r",\s*([}\]])", r"\1", js)   # strip trailing commas
        try:
            return json.loads(js)
        except Exception:
            return None


def _match_col(value, col_map):
    """Model-emitted name -> canonical dataset column, or None. Case/space/dot/hyphen tolerant."""
    if isinstance(value, list):
        value = value[0] if value else ""
    if not isinstance(value, str):       # model may emit a dict/number/None — ignore it
        return None
    return col_map.get(_sanitize_col(value.strip().lower()))


def extract_causci(model_output: str, dataset_columns: list[str]) -> dict | None:
    parsed = extract_json(model_output)
    if not isinstance(parsed, dict) or "step1" not in parsed or "step2" not in parsed:
        return None

    step2 = parsed.get("step2")
    if not isinstance(step2, str):       # model sometimes nests step2 as a dict/list
        return None
    method = step2.strip().lower()
    if method not in CAUSCI_METHODS:
        return None

    step1 = parsed.get("step1")
    if not isinstance(step1, dict):
        return None
    col_map = {_sanitize_col(c.strip().lower()): c for c in dataset_columns}

    treatment = _match_col(step1.get("treatment"), col_map)
    outcome   = _match_col(step1.get("outcome"),   col_map)
    if treatment is None or outcome is None:
        return None

    controls         = [m for c in (step1.get("controls") or []) if (m := _match_col(c, col_map))]
    instrument       = _match_col(step1.get("instrument"),       col_map)
    running_variable = _match_col(step1.get("running_variable"), col_map)
    time_variable    = _match_col(step1.get("time_variable"),    col_map)
    group_variable   = _match_col(step1.get("group_variable"),   col_map)
    mediator         = _match_col(step1.get("mediator"),         col_map)

    if method == "iv"        and instrument       is None:                                   return None
    if method == "rdd"       and (running_variable is None or step1.get("cutoff") is None):  return None
    if method == "did"       and (time_variable    is None or group_variable is None):       return None
    if method == "frontdoor" and mediator         is None:                                   return None

    estimand = step1.get("estimand")
    if isinstance(estimand, list):
        estimand = estimand[0] if estimand else ""
    if not isinstance(estimand, str):
        estimand = ""

    return {
        "step1": {
            "treatment": treatment, "outcome": outcome, "controls": controls,
            "instrument": instrument or "", "running_variable": running_variable or "",
            "cutoff": step1.get("cutoff"), "time_variable": time_variable or "",
            "group_variable": group_variable or "", "mediator": mediator or "",
            "estimand": (estimand or "").strip().lower(),
        },
        "step2": method,
    }


@lru_cache(maxsize=512)
def cached_library_fn(csv_path, method, treatment, outcome, controls_tuple,
                      estimand=None, instrument=None, running_variable=None,
                      cutoff=None, time_variable=None, group_variable=None, mediator=None):
    try:
        return library_fn({
            "step1": {"csv_path": csv_path, "treatment": treatment, "outcome": outcome,
                      "controls": list(controls_tuple), "estimand": estimand, "instrument": instrument,
                      "running_variable": running_variable, "cutoff": cutoff, "time_variable": time_variable,
                      "group_variable": group_variable, "mediator": mediator},
            "step2": method,
        })
    except Exception:
        return 0.0, False   # malformed/insufficient spec → no effect credit, never crash training



# ── Eval flush ──────────────────────────────────────────────────────────

def _flush_eval_buffer():
    """Compute + log eval metrics from the accumulated test-split samples."""
    if not _eval_buffer:
        return
    _eval_pass[0] += 1
    metrics = compute_eval_metrics(_eval_buffer)
    parts = " ".join(f"{k}:{v:.4f}" for k, v in sorted(metrics.items()))
    print(f"[verl_eval] eval_pass:{_eval_pass[0]} {parts}", flush=True)
    _eval_buffer.clear()


atexit.register(_flush_eval_buffer)   # flush if training ends right after an eval pass


# ── veRL reward interface ───────────────────────────────────────────────

def reward_fn(solution_strs, ground_truths, extra_infos):
    _call_count[0] += 1
    call = _call_count[0]
    t0 = time.time()

    rewards, items = [], []
    for solution, gt_str, extra_info in zip(solution_strs, ground_truths, extra_infos):
        ei       = json.loads(extra_info) if isinstance(extra_info, str) else extra_info
        csv_path = ei["csv_path"]
        split    = ei.get("split", "train")
        gt, parsed = {}, None
        try:
            gt     = json.loads(gt_str)
            parsed = extract_causci(solution, ei["dataset_columns"])
            if parsed is None:
                # valid JSON but unusable → milder floor so GRPO keeps signal; pure garbage → -1
                reward = -0.5 if extract_json(solution) is not None else -1.0
            else:
                s1 = parsed["step1"]
                # ungated: run the model's OWN chosen method/vars and reward the number vs reference —
                # rewards correct reasoning regardless of whether the method name matches the reference
                effect, ok = cached_library_fn(
                    csv_path=csv_path, method=parsed["step2"],
                    treatment=s1["treatment"], outcome=s1["outcome"],
                    controls_tuple=tuple(s1.get("controls") or []),
                    estimand=s1.get("estimand"), instrument=s1.get("instrument"),
                    running_variable=s1.get("running_variable"), cutoff=s1.get("cutoff"),
                    time_variable=s1.get("time_variable"), group_variable=s1.get("group_variable"),
                    mediator=s1.get("mediator"))
                reward, _ = reward_causci(parsed, gt, effect, ok)
        except Exception as e:           # one bad rollout must never kill the batch / training
            print(f"[reward] item error → -1.0: {type(e).__name__}: {e}", flush=True)
            reward, parsed = -1.0, None
        rewards.append(reward)
        items.append((parsed, gt, csv_path, split))

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    split_tag   = items[0][3] if items else "train"

    if split_tag == "train" and _eval_buffer:   # eval pass just ended → flush before training resumes
        _flush_eval_buffer()

    if split_tag == "test":
        print(f"[verl_eval] eval_pass:{_eval_pass[0]} call:{call:5d} reward={mean_reward:+.3f}", flush=True)
        _eval_buffer.extend([(p, g, c) for (p, g, c, _s) in items])
    else:
        print(f"[verl] call {call:5d}  reward={mean_reward:+.3f}", flush=True)
    print(f"[reward] n={len(rewards)} dt={time.time()-t0:.2f}s", flush=True)
    return rewards


# ---------------------------------------------------------------------------
# Batch reward manager — patches NaiveRewardManager to score the whole batch in
# one reward_fn call (one library_fn pass over all rollouts).
#
# Timing: veRL loads this file (exec_module) BEFORE instantiating
# NaiveRewardManager in the same Ray actor process, so the patch is in place
# when the manager is created. compute_score must still exist (veRL loads it by
# name), but the patched __call__ bypasses it.
# ---------------------------------------------------------------------------

def _install_batch_reward_manager() -> None:
    try:
        import torch
        import verl.workers.reward_manager.naive as _naive

        def _batch_call(self_rm, data, return_dict=False):
            responses = data.batch["responses"]          # [N, max_response_len]
            n = responses.shape[0]

            if "response_length" in data.batch:
                resp_lens = [int(data.batch["response_length"][i]) for i in range(n)]
                solution_strs = [self_rm.tokenizer.decode(responses[i, :resp_lens[i]], skip_special_tokens=True)
                                 for i in range(n)]
            else:
                pad = self_rm.tokenizer.pad_token_id
                solution_strs, resp_lens = [], []
                for i in range(n):
                    valid = responses[i][responses[i] != pad]
                    resp_lens.append(len(valid))
                    solution_strs.append(self_rm.tokenizer.decode(valid, skip_special_tokens=True))

            rm_batch      = data.non_tensor_batch["reward_model"]
            ground_truths = [rm_batch[i]["ground_truth"] for i in range(n)]
            ei_batch      = data.non_tensor_batch.get("extra_info")
            extra_infos   = [ei_batch[i] for i in range(n)] if ei_batch is not None else [{} for _ in range(n)]

            scores = reward_fn(solution_strs, ground_truths, extra_infos)

            # place scalar reward at last valid token (veRL GRPO convention)
            reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
            for i, (score, length) in enumerate(zip(scores, resp_lens)):
                if length > 0:
                    reward_tensor[i, length - 1] = float(score)
            return {"reward_tensor": reward_tensor} if return_dict else reward_tensor

        _naive.NaiveRewardManager.__call__ = _batch_call
        print("[reward] BatchRewardManager installed", flush=True)

    except Exception as e:
        print(f"[reward] BatchRewardManager install skipped ({e}), per-sample fallback active", flush=True)


_install_batch_reward_manager()


# Per-sample fallback — loaded by name via custom_reward_function.name. With the batch
# manager active this is never called; it exists so veRL can construct NaiveRewardManager.
def compute_score(data_source, solution_str, ground_truth, extra_info) -> float:
    return reward_fn([solution_str], [ground_truth], [extra_info])[0]
