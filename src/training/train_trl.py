import os
import re
import json
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)) # allow imports from project root

def _resolve_csv_path(stored: str) -> str:
    """Re-anchor stored csv_path to current PROJECT_ROOT via known anchor segments."""
    p = Path(stored)
    for anchor in ("dataset", "original_data"):
        for i, part in enumerate(p.parts):
            if part == anchor:
                return str(PROJECT_ROOT / Path(*p.parts[i:]))
    raise ValueError(f"Cannot resolve csv_path — no anchor found: {stored}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
from functools import lru_cache
# from openai import AsyncOpenAI
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, TaskType

from src.data.preprocess import preprocess
from src.config import (
    POLICY_MODEL as MODEL_NAME,
    OUTPUT_DIR_RL,
    CHECKPOINT_DIR as OUTPUT_DIR,
    JUDGE_MODEL,
    TRAIN_DATA_RL as TRAIN_DATA,
    TEST_DATA_RL as TEST_DATA,
    TRAIN_BATCH_SIZE,
    N_ROLLOUTS,
    FINAL_MODEL,
    TRAIN_MAX_TOKENS,
    MAX_PROMPT_LEN,
    PLOT_DIR,
)
from src.training.tool_calling import library_fn
from src.training.eval_metrics import compute_eval_metrics, save_eval_plots

# ---------------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------------

EVAL_EVERY = 100   # steps between test-set evaluations
EVAL_N_CLADDER = 100
EVAL_N_CAUSCI  = 50

class EvalCallback(TrainerCallback):
    def __init__(self, test_dataset, tokenizer, plot_dir):
        self.test_dataset = test_dataset
        self.tokenizer    = tokenizer
        self.plot_dir     = Path(plot_dir)
        self.eval_steps   = []
        self.history      = {}

    def on_step_end(self, args, state, control, **kwargs):

        print("Performing evaluation...")

        if state.global_step % EVAL_EVERY != 0:
            return
        if int(os.environ.get("LOCAL_RANK", 0)) != 0:
            return

        model = kwargs.get("model")
        if model is None:
            return

        ds = self.test_dataset
        cladder_idxs = [i for i, s in enumerate(ds["source"]) if s == "cladder"][:EVAL_N_CLADDER]
        causci_idxs  = [i for i, s in enumerate(ds["source"]) if s == "causcibench"][:EVAL_N_CAUSCI]

        model.eval()
        items    = []
        first_out = None
        for idx in tqdm(cladder_idxs + causci_idxs):
            row      = ds[idx]
            src      = row["source"]
            gt       = json.loads(row["groundtruth"])
            cols     = row["dataset_columns"]
            csv_path = row["csv_path"]

            input_ids = self.tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            out = self.tokenizer.decode(gen_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
            if first_out is None:
                first_out = out
            parsed = extract_cladder(out) if src == "cladder" else extract_causci(out, cols)
            items.append((src, parsed, gt, csv_path))

        model.train()

        step    = state.global_step
        metrics = compute_eval_metrics(items)
        self.eval_steps.append(step)
        for k, v in metrics.items():
            self.history.setdefault(k, []).append(v)

        print(f"\n[eval step {step}]")
        if first_out is not None:
            print(f"  sample response: {first_out}")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.4f}")

        self.plot_dir.mkdir(parents=True, exist_ok=True)
        with open(self.plot_dir / "eval_log.jsonl", "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")

        save_eval_plots(self.history, self.eval_steps, self.plot_dir)

class MemoryCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        # torch.cuda.synchronize()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # rank — 16 is a good start for reasoning tasks
    lora_alpha=32,           # scaling factor, typically 2x r
    lora_dropout=0.05,
    target_modules=[         # Qwen3 attention + MLP projections
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
)

training_args = GRPOConfig(
    # --- core ---
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=1,
    use_liger_kernel=True,
    gradient_checkpointing=False,

    # --- GRPO-specific ---
    num_generations=N_ROLLOUTS,
    max_completion_length=TRAIN_MAX_TOKENS,

    # --- KL penalty ---
    beta=0.04,

    # --- vLLM ---
    use_vllm=True,
    vllm_mode="colocate",
    # vllm_enable_sleep_mode=True,          # was False — this is the fix
    vllm_gpu_memory_utilization=0.3,        # was 0.25 — safe now that sleep releases during training
    # vllm_enable_prefix_caching=True,      # add this — 8 rollouts share the same prompt prefix
    vllm_max_model_length=4096,

    # --- logging / saving ---
    logging_steps=10,
    save_steps=100,
    report_to="none",

    # --- model loading --- only attn_implementation here, dtype handled above
    model_init_kwargs={
        "attn_implementation": "flash_attention_3",
        "dtype": "bfloat16"
    },

    # --- misc ---
    bf16=True,
    seed=42,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    dataloader_persistent_workers=True,
    shuffle_dataset=True,

    # --- evaluation during training ---
    eval_strategy="steps",
    eval_steps=1000,
)

# _async_judge = AsyncOpenAI(base_url="http://localhost:8001/v1", api_key="token")

# async def _judge_one(prompt: str) -> float:
#     r = await _async_judge.chat.completions.create(
#         model=JUDGE_MODEL,
#         max_tokens=2,
#         temperature=0.0,
#         messages=[
#             {"role": "system", "content": "You are a binary scorer. Reply with only 0 or 1. No other text."},
#             {"role": "user", "content": prompt}
#         ],
#     )
#     raw = r.choices[0].message.content
#     raw = raw.strip()
#     raw = re.sub(r'[^01]', '', raw)   # strip everything except 0 and 1
#     return float(int(raw[0]))          # take first character, convert

# def batch_judge(prompts: list[str]) -> list[float]:
#     """Fire all judge calls concurrently, return results in order."""
#     return asyncio.run(asyncio.gather(*[_judge_one(p) for p in prompts]))
    
# def batch_judge(prompts: list[str]) -> list[float]:
#     results = []

#     def _run():
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         try:
#             results.extend(loop.run_until_complete(asyncio.gather(*[_judge_one(p) for p in prompts])))
#         finally:
#             loop.close()

#     t = threading.Thread(target=_run)
#     t.start()
#     t.join()
#     return results


_judge_client = OpenAI(base_url="http://localhost:8001/v1", api_key="token")

@lru_cache(maxsize=512)
def _judge_one(prompt: str) -> float:
    try:
        r = _judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=2,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
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
# 1. DATA LOADING  ← fill this in
# ---------------------------------------------------------------------------
CLADDER_SYSTEM_PROMPT = """You are a causal inference expert. Think step by step inside <think> tags, 
then output a JSON object and no other text. No explanations, no leading or tailing sentences. Just answer with the JSON object. 
"""

CLADDER_USER_PROMPT = """
## Query Types

| Type | Formula | Use when |
|------|---------|----------|
| marginal | P(Y=y) | Baseline probability of an outcome, no conditions or interventions |
| correlation | P(Y=y\\|X=x) | Observing X changes probability of Y, no intervention |
| ate | E[Y\\|do(X=1)] - E[Y\\|do(X=0)] | Forcing X to a value — what is the causal effect on Y |
| backadj | Does set S block all backdoor paths X→Y? | Question asks whether adjusting for a variable set is valid |
| det-counterfactual | P(Y_x=y \\| evidence) | What would Y have been if X were different, given observed facts |
| ett | E[Y₁-Y₀ \\| X=1] | Among those who received treatment, what if they hadn't |
| nde | E[Y_{1,M₀} - Y_{0,M₀}] | Direct effect of X on Y, holding mediator at its natural value |
| nie | E[Y_{0,M₁} - Y_{0,M₀}] | Indirect effect of X on Y, only through the mediator |
| collider_bias | Does do(X) affect Y when Z is a collider? | X and Y share only a common effect, no common cause |
| exp_away | Does P(Y\\|X) change when conditioning on collider Z? | Conditioning on a common effect creates spurious association |

## Estimation Rules

- **ate — backdoor (confounders exist)**: Σ_z P(Z=z) [P(Y=1|X=1,Z=z) - P(Y=1|X=0,Z=z)]
- **ate — frontdoor (mediator, confounded treatment)**: Σ_m P(M=m|X=1) Σ_x P(X=x) P(Y=1|M=m,X=x) — same with X=0, subtract
- **ate — instrumental variable (instrument V2 exists)**: [P(Y=1|V2=1) - P(Y=1|V2=0)] / [P(X=1|V2=1) - P(X=1|V2=0)]
- **ett**: Σ_z P(Z=z|X=1) [P(Y=1|X=1,Z=z) - P(Y=1|X=0,Z=z)]
- **det-counterfactual**: (1) Abduction — infer U from evidence, (2) Action — set X=x, (3) Prediction — compute P(Y)
- **nde**: Σ_m P(M=m|X=0) [P(Y=1|X=1,M=m) - P(Y=1|X=0,M=m)]
- **nie**: Σ_m [P(M=m|X=1) - P(M=m|X=0)] P(Y=1|X=0,M=m)
- **backadj / collider_bias / exp_away**: graph analysis only — trace paths, check d-separation, no arithmetic

## Answer Interpretation

- **ate / ett / nde / nie**: compute the value. Positive → treatment increases outcome. Negative → decreases. Match to what the question asks.
- **marginal**: compare P(Y=1) to threshold or what the question asks.
- **correlation**: compare P(Y=1|X=1) vs P(Y=1|X=0).
- **det-counterfactual**: compare computed probability to prior or threshold.
- **backadj / collider_bias / exp_away**: yes or no from graph structure alone.

## Scenario

{verbalized_story}

## Task

Step 1 — Causal Structure: Assign short variable names (X, Y, Z, M, V1, V2, ...) to each entity in the scenario. List every directed edge as A -> B.

Step 2 — Query Type: Classify as exactly one type from the table above. One word only.

Step 3 — Estimand: Write the mathematical expression for the query. Apply backdoor / frontdoor / IV / abduction-action-prediction as needed. No numbers yet.

Step 4 — Compute: Substitute every numeric value from the scenario into the estimand. Show each arithmetic step explicitly. End with the final number. For backadj / collider_bias / exp_away, trace the graph paths and state your conclusion.

Then output this JSON and nothing else:

{{
  "step1": "<variable assignments and all directed edges>",
  "step2": "<query type>",
  "step3": "<estimand expression>",
  "step4": "<full arithmetic or graph reasoning, final value at the end>",
  "step5": "<yes or no>"
}}
"""

CAUSCI_SYSTEM_PROMPT = """You are a causal inference expert. Analyze the study design carefully 
before selecting variables and methods. Think through your reasoning, 
then output only the JSON.
"""

CAUSCI_USER_PROMPT = """
## Study Description
{dataset_description}

## Dataset
Path: {file_path}
Shape: {shape}

Columns and types:
{columns_and_types}

First 5 rows:
{df_head}

Summary statistics:
{df_describe}

Missing values per column:
{missing_values}

Low-cardinality columns (≤10 unique values):
{low_cardinality}

## Question
{query}

---

## Method Reference

| Method | Use when |
|--------|----------|
| diff_in_means | RCT with enforced compliance. Groups comparable by design. No confounding. |
| ols | Observational. All confounders observed and included. No unobserved confounding. |
| ipw | Observational. Confounders observed. Reweight by propensity score. Needs overlap: 0 < e(X) < 1. |
| matching | Observational. Confounders observed. Use when propensity score overlap is poor. |
| did | Panel data. Treatment introduced at one point in time to one group. Time variable must be treatment timing, not a covariate. Parallel trends must hold. |
| rdd | Treatment assigned by a running variable crossing a known cutoff. Units just above and below cutoff are comparable. |
| iv | Unobserved confounders exist. Valid instrument available — correlated with treatment, affects outcome only through treatment. |
| frontdoor | Unobserved confounders exist. Full mediator pathway T→M→Y with no unobserved T→M or M→Y confounding. |
| glm | Binary outcome (logistic) or count outcome (Poisson). Confounders observed. |

## Estimand Reference

| Method | Estimand |
|--------|----------|
| diff_in_means | ATE |
| ols | ATE |
| ipw | ATE, ATT, or ATC — based on whether question asks about population, treated group, or control group |
| matching | ATE or ATT |
| did | ATT |
| rdd | Local ATE at the cutoff |
| iv | LATE |
| frontdoor | ATE |
| glm | Conditional effect (log-odds for binary, incidence rate ratio for counts) |

---

Think through the following before answering:
- Was treatment randomly assigned or self-selected?
- Are confounders observed or unobserved?
- Is there a time variable marking treatment timing (not just a covariate)?
- Is there a continuous running variable with a cutoff?
- Is there a variable that affects treatment but not outcome directly?
- Is the outcome binary, count, or continuous?
- Does the question ask about the full population (ATE), treated units (ATT), or local effect (LATE)?

Then output this JSON and nothing else after your thinking:

{{
  "step1": {{
    "treatment": "<exact column name>",
    "outcome": "<exact column name>",
    "controls": ["<col1>", "<col2>"],
    "instrument": null,
    "running_variable": null,
    "cutoff": null,
    "time_variable": null,
    "group_variable": null,
    "mediator": null,
    "estimand": "<ATE, ATT, ATC, LATE, or conditional>"
  }},
  "step2": "<method name>"
}}
"""

CLADDER_TRAIN_LIMIT = 10_000  # set to None to use full dataset


def _limit_cladder(rows: list[dict], limit: int) -> list[dict]:
    cladder = [r for r in rows if r["source"] == "cladder"]
    other   = [r for r in rows if r["source"] != "cladder"]

    # step 1: count unique (query_type, formal_form, polarity) buckets
    bucket_counts: dict[tuple, int] = {}
    for r in cladder:
        gt  = r.get("groundtruth") or {}
        key = (gt.get("step2") or "", gt.get("step3") or "", str(r.get("label", "")))
        bucket_counts[key] = bucket_counts.get(key, 0) + 1

    n_per_bucket = limit // len(bucket_counts)

    # step 2: stream with per-bucket counters
    counters: dict[tuple, int] = {}
    kept = []
    for r in cladder:
        gt  = r.get("groundtruth") or {}
        key = (gt.get("step2") or "", gt.get("step3") or "", str(r.get("label", "")))
        cnt = counters.get(key, 0)
        if cnt < n_per_bucket:
            kept.append(r)
            counters[key] = cnt + 1

    print(f"CLadder limit: {len(cladder)} → {len(kept)} "
          f"({len(bucket_counts)} buckets × {n_per_bucket}, target={limit})")
    return kept + other


def load_dataset_for_grpo(path) -> Dataset:
    """
    Load preprocessed JSONL and return a HuggingFace Dataset.

    Each row contains:
      - prompt: list of chat messages (system + user + assistant prefill)
      - source: "cladder" or "causcibench"
      - groundtruth: dict with step1..step5
      - dataset_columns: list of CSV column names (causcibench only, else [])
    """
    with open(path, "r") as f:
        raw = [json.loads(line) for line in f]

    if CLADDER_TRAIN_LIMIT is not None and str(path).endswith("train.jsonl"):
        raw = _limit_cladder(raw, CLADDER_TRAIN_LIMIT)

    new_data = []
    for r in raw:
        if r["source"] == "cladder":
            system_prompt = CLADDER_SYSTEM_PROMPT
            dataset_columns = []
            csv_path = ""
        else:
            system_prompt = CAUSCI_SYSTEM_PROMPT
            if "csv_path" in r:
                csv_path = _resolve_csv_path(r["csv_path"])
            else:
                # preprocess crashed before writing csv_path — extract from prompt's "Path: " line
                m = re.search(r"^Path: (.+)$", r["prompt"], re.MULTILINE)
                assert m, f"No 'Path: ' line in causcibench prompt for row {r.get('id')}"
                csv_path = _resolve_csv_path(m.group(1).strip())
            dataset_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()

        messages = [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": r["prompt"]},
            {"role": "assistant", "content": "<think>\n"},
        ]
        new_data.append({
            "prompt":          messages,
            "source":          r["source"],
            "groundtruth":     json.dumps(r["groundtruth"], default=str),
            "dataset_columns": dataset_columns,
            "csv_path":        csv_path,
        })
    return Dataset.from_list(new_data)

# ---------------------------------------------------------------------------
# 2. REWARD FUNCTION  ← fill this in
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


def reward_causci(
    prediction: dict,
    ground_truth: dict,
    library_effect: float,
) -> float:

    scores = {}

    # method — exact match, binary
    # this is the core causal reasoning decision
    pred_method = prediction.get('step2', '').strip().lower()
    ref_method  = ground_truth.get('step2', '').strip().lower()
    scores['method'] = 1 if pred_method == ref_method else 0

    # cascade gate — wrong method means the entire
    # estimation strategy is invalid
    if scores['method'] == 0:
        return -1.0, scores

    # treatment — exact match, binary
    pred_treat = prediction.get('step1', {}).get('treatment', '').strip()
    ref_treat  = ground_truth.get('step1', {}).get('treatment', '').strip()
    scores['treatment'] = 1 if pred_treat == ref_treat else 0

    # outcome — exact match, binary
    pred_outcome = prediction.get('step1', {}).get('outcome', '').strip()
    ref_outcome  = ground_truth.get('step1', {}).get('outcome', '').strip()
    scores['outcome'] = 1 if pred_outcome == ref_outcome else 0

    # cascade gate — wrong treatment or outcome means
    # the library ran the wrong regression entirely
    if scores['treatment'] == 0 or scores['outcome'] == 0:
        return -0.5, scores

    # controls — overlap score, continuous
    pred_controls = set(prediction.get('step1', {}).get('controls', []))
    ref_controls  = set(ground_truth.get('step1', {}).get('controls', []))
    if len(ref_controls) > 0:
        scores['controls'] = len(pred_controls & ref_controls) / len(ref_controls)
    else:
        scores['controls'] = 1.0 if len(pred_controls) == 0 else 0.0

    controls_good = scores['controls'] >= 0.75

    # effect accuracy — clipped relative error
    ref_effect = ground_truth.get('step5')
    # if ref_effect != 0:
    if ref_effect is not None and ref_effect != 0:
        mre = abs(library_effect - ref_effect) / abs(ref_effect)
        effect_correct = mre <= 0.05  # within 5% — matches paper's EA metric
    else:
        effect_correct = False  # undefined when true effect is exactly 0

    scores['effect'] = 1 if effect_correct else 0

    # final reward
    if controls_good and effect_correct:
        reward = 1.0
    elif not controls_good and effect_correct:
        reward = 0.5   # got lucky with bad controls
    elif controls_good and not effect_correct:
        reward = -0.25  # right setup, wrong number
    else:
        reward = -0.25  # poor controls, wrong number

    return reward, scores

def reward_cladder_precomputed(prediction: dict, ground_truth: dict, s1_score: int, s3_score: int | None) -> tuple[float, dict]:
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

    scores['step3'] = s3_score if s3_score is not None else 0
    step3_penalty = 0.0 if scores['step3'] == 1 else -0.25

    pred_step5 = prediction.get('step5', '').strip().lower()
    ref_step5  = ground_truth.get('step5', '').strip().lower()
    scores['step5'] = 1 if pred_step5 == ref_step5 else 0

    if scores['step5'] == 1:
        reward = 1.0 + step3_penalty
    else:
        reward = -0.75 + step3_penalty

    return reward, scores

@lru_cache(maxsize=512)
def cached_library_fn(
    csv_path, method, treatment, outcome, controls_tuple,
    estimand=None, instrument=None, running_variable=None,
    cutoff=None, time_variable=None, group_variable=None, mediator=None,
):
    return library_fn({
        "step1": {
            "csv_path":         csv_path,
            "treatment":        treatment,
            "outcome":          outcome,
            "controls":         list(controls_tuple),
            "estimand":         estimand,
            "instrument":       instrument,
            "running_variable": running_variable,
            "cutoff":           cutoff,
            "time_variable":    time_variable,
            "group_variable":   group_variable,
            "mediator":         mediator,
        },
        "step2": method,
    })

def reward_fn(completions: list, **kwargs) -> list[float]:
    sources      = kwargs["source"]
    groundtruths = kwargs["groundtruth"]
    dataset_cols = kwargs["dataset_columns"]
    csv_paths    = kwargs["csv_path"]

    # Phase 1 — parse all completions
    items = []
    for completion, source, gt, cols, csv_path in zip(
        completions, sources, groundtruths, dataset_cols, csv_paths
    ):
        if isinstance(completion, list):
            completion = completion[-1]["content"]
        gt = json.loads(gt)
        parsed = extract_cladder(completion) if source == "cladder" else extract_causci(completion, cols)
        items.append((source, parsed, gt, cols, csv_path))

    # Phase 2 — collect judge prompts for all cladder items and fire concurrently
    # step1 always needed; step3 only if step2 is an exact match (avoids a wasted call)
    judge_prompts = []
    prompt_idx = {}  # item index → {"step1": int, "step3": int}

    for i, (source, parsed, gt, cols, csv_path) in enumerate(items):
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
    for i, (source, parsed, gt, cols, csv_path) in enumerate(items):
        if source == "cladder":
            if parsed is None:
                rewards.append(-1.0)
                continue
            idxs = prompt_idx.get(i, {})
            s1 = int(round(judge_scores[idxs["step1"]])) if "step1" in idxs else 0
            s3 = int(round(judge_scores[idxs["step3"]])) if "step3" in idxs else None
            reward, _ = reward_cladder_precomputed(parsed, gt, s1, s3)

        elif source == "causcibench":
            if parsed is None:
                rewards.append(-1.0)
                continue
            step1          = parsed["step1"]
            library_effect, _ = cached_library_fn(
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
            ) if parsed["step2"] == gt.get("step2") else (0.0, False)
            reward, _ = reward_causci(parsed, gt, library_effect)

        else:
            raise ValueError(f"Unknown source: {source!r}")

        rewards.append(reward)

    return rewards

def extract_json(model_output: str) -> dict | None:
    """
    Extracts JSON from model output.
    Handles three cases:
    - Clean output: just the JSON
    - JSON after </think> block
    - JSON embedded in text
    """
    # strip think block if present
    if '</think>' in model_output:
        model_output = model_output.split('</think>')[-1]

    # find first { to last }
    start = model_output.find('{')
    end   = model_output.rfind('}')

    if start == -1 or end == -1:
        return None

    json_str = model_output[start:end+1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # try cleaning common model mistakes
        # trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        try:
            return json.loads(json_str)
        except:
            return None
        
CLADDER_REQUIRED = {'step1', 'step2', 'step3', 'step4', 'step5'}
CLADDER_QUERY_TYPES = {
    'marginal', 'correlation', 'ate', 'backadj',
    'det-counterfactual', 'ett', 'nde', 'nie',
    'collider_bias', 'exp_away'
}

def extract_cladder(model_output: str) -> dict | None:
    parsed = extract_json(model_output)

    if parsed is None:
        return None

    # check required fields
    if not CLADDER_REQUIRED.issubset(parsed.keys()):
        return None

    # validate step2 is a known query type
    step2 = parsed.get('step2', '').strip().lower()
    if step2 not in CLADDER_QUERY_TYPES:
        return None

    # validate step5 is yes or no
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

CAUSCI_METHODS = {
    'diff_in_means', 'ols', 'ipw', 'matching',
    'did', 'rdd', 'iv', 'frontdoor', 'glm'
}

CAUSCI_ESTIMANDS = {
    'ate', 'att', 'atc', 'late', 'conditional'
}

def extract_causci(model_output: str, dataset_columns: list[str]) -> dict | None:
    parsed = extract_json(model_output)

    if parsed is None:
        return None

    # validate top level
    if 'step1' not in parsed or 'step2' not in parsed:
        return None

    step1  = parsed['step1']
    method = parsed.get('step2', '').strip().lower()

    if method not in CAUSCI_METHODS:
        return None

    # validate treatment and outcome are real columns
    treatment = step1.get('treatment', '').strip()
    outcome   = step1.get('outcome', '').strip()

    if treatment not in dataset_columns:
        return None
    if outcome not in dataset_columns:
        return None

    # validate controls are all real columns
    controls = step1.get('controls', [])
    controls = [c for c in controls if c in dataset_columns]

    # validate method-specific required fields
    if method == 'iv':
        instrument = step1.get('instrument')
        if instrument not in dataset_columns:
            return None

    if method == 'rdd':
        running_var = step1.get('running_variable')
        cutoff      = step1.get('cutoff')
        if running_var not in dataset_columns:
            return None
        if cutoff is None:
            return None

    if method == 'did':
        time_var  = step1.get('time_variable')
        group_var = step1.get('group_variable')
        if time_var not in dataset_columns:
            return None
        if group_var not in dataset_columns:
            return None

    if method == 'frontdoor':
        mediator = step1.get('mediator')
        if mediator not in dataset_columns:
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
# MAIN
# ---------------------------------------------------------------------------

def main():
    # --- tokenizer (used only to sanity-check chat template exists) ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    assert tokenizer.chat_template is not None, (
        f"{MODEL_NAME} has no chat_template. "
        "Set one with tokenizer.chat_template = '...' before training."
    )

    # --- data ---
    dataset = load_dataset_for_grpo(TRAIN_DATA)
    test_dataset = load_dataset_for_grpo(TEST_DATA)
    assert "prompt" in dataset.column_names, (
        "Dataset must have a 'prompt' column (list of chat messages per row)."
    )

    # optional: quick sanity check on reward function with dummy data
    # dummy = ["<answer>42</answer>"] * training_args.num_generations
    # print("Reward sanity check:", reward_fn(dummy))

    eval_callback   = EvalCallback(test_dataset, tokenizer, PLOT_DIR)
    memory_callback = MemoryCallback()

    # --- trainer ---
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        args=training_args,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        callbacks=[eval_callback, memory_callback],
        peft_config=lora_config,
    )

    trainer.train()

    # save final model after training to FINAL_MODEL
    trainer.save_model(FINAL_MODEL)
    tokenizer.save_pretrained(FINAL_MODEL)
    print(f"Model saved to {FINAL_MODEL}")


if __name__ == "__main__":
    import fcntl, time
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.makedirs(OUTPUT_DIR_RL, exist_ok=True)

    sentinel = Path(OUTPUT_DIR_RL) / ".preprocess.done"

    if local_rank == 0:
        sentinel.unlink(missing_ok=True)  # clear any stale sentinel from prior run
        preprocess(
            cladder_prompt=CLADDER_USER_PROMPT,
            causci_prompt=CAUSCI_USER_PROMPT,
            output_dir=Path(OUTPUT_DIR_RL)
        )
        sentinel.touch()  # signal to other ranks that preprocessing is done
    else:
        # poll until rank0 writes the sentinel
        while not sentinel.exists():
            time.sleep(1)

    main()