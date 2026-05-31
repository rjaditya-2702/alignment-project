"""
SFT QLoRA Training — Qwen3-8B on CLaDDer + CauSciBench
Loss: CE over thinking tokens + λ * CE over answer token (cladder)
      CE over all response tokens (causcibench)
"""
import sys
import os
import re
import json
import time
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────

from src.config import (TRAIN_DATA_SFT_LORA as TRAIN_DATA,
                        TEST_DATA_SFT_LORA  as TEST_DATA,
                        SFT_LORA_OUTPUT_DIR,
                        SFT_LORA_PLOT_DIR, SFT_LORA_CHECKPOINT_DIR,
                        TRAIN_BATCH_SIZE)
from src.data.preprocess import preprocess
from src.training.tool_calling import library_fn
from src.training.eval_metrics import compute_eval_metrics, save_eval_plots

CLADDER_PROMPT = """You are given a scenario describing relationships between variables, along with numerical data and a question. Your task is to determine the answer by following these steps precisely.
---

Strict rules (follow these exactly):
- Nothing before "## Step 1" and nothing after the single word in Step 5.
- Write each step exactly once.
- Each step must be short and direct. No long paragraphs or verbosity.
- Do not repeat content from previous steps.
- Output Steps 1–4 inside the thinking block only.
- After Step 4, close the thinking block.
- After </think>, output exactly one word: "Yes" or "No". No quotes, no punctuation, no extra text.
- Stop immediately after that word.
- Do not repeat any step, any code block, or the word "Yes".

### Query Type Definitions

1. **marginal** — What is the overall probability of a variable?
   Formula: P(Y = y)
   Use when: The question asks about the baseline likelihood of an outcome across the whole population, with no conditions or interventions.

2. **correlation** — Does observing one variable change the probability of another?
   Formula: P(Y = y | X = x)
   Use when: The question asks whether knowing or observing one variable's value changes the likelihood of another. No intervention — just observation.

3. **ate** — What is the effect of actively changing (intervening on) a variable?
   Formula: E[Y | do(X=1)] - E[Y | do(X=0)]
   Use when: The question asks whether forcing or setting a variable to a value increases or decreases an outcome. The key word is "intervention" or "effect of doing X."
   Key technique: Use backdoor adjustment if confounders exist: Σ_z P(Z=z)[P(Y=1|X=1,Z=z) - P(Y=1|X=0,Z=z)]. Use frontdoor adjustment if treatment is confounded but a mediator satisfies the frontdoor criterion.

4. **backadj** — Should we adjust for a set of variables when estimating an effect?
   Formula: Check if the set S blocks all backdoor paths between treatment X and outcome Y in the graph.
   Use when: The question asks whether to look at the overall correlation between X and Y, or to look at it stratified by (adjusted for) other variables.
   Answer is yes if S is a valid adjustment set (blocks all non-causal paths), no otherwise.

5. **det-counterfactual** — What would have happened under a different condition?
   Formula: P(Y_x = y | evidence)
   Use when: The question asks what the outcome would have been if the treatment had been different, given specific observed facts. Uses the three-step procedure: (1) Abduction — update P(U) given evidence, (2) Action — set X = x in the structural equations, (3) Prediction — compute P(Y = y) in the modified model.

6. **ett** — For those who received treatment, what would have happened without it?
   Formula: E[Y₁ - Y₀ | X = 1]
   Use when: The question focuses specifically on the treated subgroup and asks how their outcome would change in the absence of treatment. Also called Average Treatment Effect on the Treated (ATT).

7. **nde** — What is the direct effect, not through any mediator?
   Formula: E[Y_{1,M₀} - Y_{0,M₀}]
   Use when: The question asks about the effect of X on Y while holding the mediator at its natural value under no treatment. Also called Natural Direct Effect.

8. **nie** — What is the indirect effect, only through the mediator?
   Formula: E[Y_{0,M₁} - Y_{0,M₀}]
   Use when: The question asks about the effect of X on Y that operates only through an intermediate variable (mediator), not directly. Also called Natural Indirect Effect.

9. **collider_bias** — Does intervening on one cause of a common effect create a spurious association with another cause?
   Formula: Check whether do(X) changes Y when X and Y share only a common effect (collider), not a common cause.
   Use when: The question involves a variable that is caused by both X and Y (a collider), and asks whether intervening on X affects Y. The answer is always no if X and Y have no common causes — the apparent association through the collider is spurious.

10. **exp_away** — Does conditioning on a common effect change the association between its causes?
    Formula: Compare P(Y | X) versus P(Y | X, Z) where Z is a collider.
    Use when: The question asks whether holding fixed (conditioning on) a common effect of X and Y changes how X and Y are associated. This is the "explaining away" phenomenon — conditioning on a collider can create a spurious association between its parents.

---

Now solve the problem in the following way:

```
<think>
## Step 1: Causal Structure
Assign algebraic variables (e.g., X, Y, Z) to each entity mentioned in the scenario. Identify all directed causal edges.
For example: V1 -> V2, V2 -> V3

## Step 2: Query Classification
Based on the question and the definitions above, classify this query. Return exactly one of:
marginal, correlation, ate, backadj, det-counterfactual, ett, nde, nie, collider_bias, exp_away

## Step 3: Derive Estimand
Using the causal graph from Step 1 and the query type from Step 2, write the mathematical expression that answers the question.
- If the query involves do(), apply do-calculus rules (backdoor adjustment, frontdoor adjustment) to eliminate do() terms and express everything in terms of observable probabilities.
- If the query is counterfactual, apply the three-step abduction-action-prediction procedure.
- If the query is about adjustment sets or collider bias, reason about the graph structure (paths, d-separation).

Show your derivation.

## Step 4: Compute
Using the estimand from Step 3 and the numerical values given in the Data section, compute the result step by step. Show the arithmetic explicitly — substitute each probability value and simplify to a final number.
</think>
```

Based on the computed result and what the question is asking, answer Yes or No. One word only.
- For ate/ett/nde/nie: positive result → Yes if question asks "does X increase Y", No if "decrease". Flip if question asks the opposite.
- For marginal: P(Y) > 0.5 and question asks "is Y more likely than not" → Yes.
- For correlation: P(Y|X=1) > P(Y|X=0) and question asks "does observing X increase Y" → Yes.
- For backadj/collider_bias/exp_away: Yes or No based on graph analysis.
- For det-counterfactual: Yes or No based on computed probability.

IMPORTANT: After writing answer with a single word, STOP. No more text is allowed.

## Scenario
{verbalized_story}

Respond now. Begin directly with <think>
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

CAUSCI_METHODS = {
    'diff_in_means', 'ols', 'ipw', 'matching',
    'did', 'rdd', 'iv', 'frontdoor', 'glm'
}

CLADDER_QUERY_TYPES_SFT = {                                                                                                                                                                          
    'marginal', 'correlation', 'ate', 'backadj',                                                                                                                                              
    'det-counterfactual', 'ett', 'nde', 'nie',                                                                                                                                                       
    'collider_bias', 'exp_away'                                                                                                                                                               
}  

for d in [SFT_LORA_OUTPUT_DIR, SFT_LORA_PLOT_DIR, SFT_LORA_CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
POLICY_MODEL       = "Qwen/Qwen3-8B"
MAX_PROMPT_LEN     = 6000
TRAIN_MAX_TOKENS   = 1200
LR                 = 2e-5
WEIGHT_DECAY       = 0.01
GRAD_ACCUM         = 1
MAX_GRAD_NORM      = 1.0
MAX_EPOCHS         = 3
SAVE_EVERY         = 500
LOG_EVERY          = 10
LORA_R             = 32
LORA_ALPHA         = 64
LORA_DROPOUT       = 0.05
ANSWER_LAMBDA      = 5.0
DTYPE              = torch.bfloat16
EVAL_EVERY         = 100

# ── Tokenizer ──────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL)
tokenizer.padding_side = "right"

# Special token IDs we need for loss masking
THINK_CLOSE_STR = "<|im_end|>"   # end of assistant turn
IM_END_ID       = tokenizer.convert_tokens_to_ids("<|im_end|>")
# </think> token — Qwen3 uses token id 151668
THINK_CLOSE_ID  = tokenizer.convert_tokens_to_ids("</think>")


# ── CSV path resolution ────────────────────────────────────────────────────────
def _resolve_csv_path(stored: str) -> str:
    p = Path(stored)
    for anchor in ("dataset", "original_data"):
        for i, part in enumerate(p.parts):
            if part == anchor:
                return str(PROJECT_ROOT / Path(*p.parts[i:]))
    raise ValueError(f"Cannot resolve csv_path — no anchor found: {stored}")


# ── CausCI scoring ─────────────────────────────────────────────────────────────

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
        except:
            return None


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

    controls = step1.get('controls', [])
    controls = [c for c in controls if c in dataset_columns]

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


def extract_cladder_sft(output_text: str) -> dict | None:
    """Parse step2 (query type) and step5 (yes/no) from a CLaDDer SFT response.

    The model generates: <thinking content> </think> Yes/No
    When decoded with skip_special_tokens=True, </think> may be stripped,
    leaving the answer as the last word.
    """
    if '</think>' in output_text:
        thinking, _, tail = output_text.partition('</think>')
    else:
        thinking = output_text
        tail     = ""

    # Final answer: first yes/no in tail, or last word of the full text
    step5 = None
    for tok in tail.strip().lower().split():
        if tok.rstrip('.,!?') in ('yes', 'no'):
            step5 = tok.rstrip('.,!?')
            break
    if step5 is None:
        words = output_text.strip().lower().split()
        if words and words[-1].rstrip('.,!?') in ('yes', 'no'):
            step5 = words[-1].rstrip('.,!?')
    if step5 is None:
        return None

    # Step 2: query type from "## Step 2: ..." section
    step2 = None
    m = re.search(r'##\s*Step\s*2[^\n]*\n([^\n#]+)', thinking, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().lower()
        if candidate in CLADDER_QUERY_TYPES_SFT:
            step2 = candidate

    return {'step5': step5, 'step2': step2}


def reward_causci(prediction: dict, ground_truth: dict, library_effect: float) -> tuple[float, dict]:
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


# ── Build Full Sequence ────────────────────────────────────────────────────────
def build_sequence(prompt: str, thinking: str, answer: str, source: str = "cladder") -> dict:
    """
    Constructs the full token sequence for one training sample.

    Both sources:
        [prompt tokens] [<think> ... thinking ... </think>] [response] [<|im_end|>\n]

    Loss is applied ONLY to the response after </think> — thinking tokens carry zero loss:
        both sources: 1.0 on all JSON response tokens
    If </think> is not found the sample is excluded from loss (mask stays all-zero).
    """
    system = (
        "You are a causal reasoning expert and a helpful assistant. Don't explain. just do the task"
        if source == "cladder"
        else CAUSCI_SYSTEM_PROMPT
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
        # {"role": "assistant", "content": "<think>"}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    response_text = "<think>\n" + thinking + "\n</think>" + answer + "<|im_end|>"
    full_text     = prompt_text + response_text

    full_ids = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    ).input_ids[0]

    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    ).input_ids[0]

    prompt_len = len(prompt_ids)

    max_len = MAX_PROMPT_LEN + TRAIN_MAX_TOKENS
    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]

    seq_len   = len(full_ids)
    loss_mask = torch.zeros(seq_len, dtype=DTYPE)

    # Find </think> — loss starts only after it, thinking tokens get zero weight
    think_close_pos = None
    for i in range(prompt_len, seq_len):
        if full_ids[i].item() == THINK_CLOSE_ID:
            think_close_pos = i
            break

    if think_close_pos is not None:
        response_start = think_close_pos + 1
        if source == "cladder":
            if response_start < seq_len:
                loss_mask[response_start] = ANSWER_LAMBDA
        else:
            if response_start < seq_len:
                loss_mask[response_start:] = 1.0
    # else: </think> not found — mask stays all-zero, sample excluded from loss

    return {"input_ids": full_ids, "loss_mask": loss_mask, "prompt_len": prompt_len}


# ── Dataset ────────────────────────────────────────────────────────────────────
class SFTDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    """Left-pad to the longest sequence in the batch."""
    max_len    = max(s["input_ids"].shape[0] for s in batch)
    input_ids  = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    loss_masks = torch.zeros(len(batch), max_len, dtype=DTYPE)

    for i, s in enumerate(batch):
        L = s["input_ids"].shape[0]
        input_ids[i, -L:]  = s["input_ids"]
        loss_masks[i, -L:] = s["loss_mask"]

    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return {
        "input_ids":         input_ids,
        "attention_mask":    attention_mask,
        "loss_mask":         loss_masks,
        "prompt_lens":       [s.get("prompt_len", 0) for s in batch],
        "source":            [s.get("source", "cladder") for s in batch],
        "groundtruth":       [s.get("groundtruth", {}) for s in batch],
        "csv_path":          [s.get("csv_path", "") for s in batch],
        "dataset_columns":   [s.get("dataset_columns", []) for s in batch],
    }


# ── Custom Loss ────────────────────────────────────────────────────────────────
def compute_loss(logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """
    logits    : (B, L, V)
    input_ids : (B, L)
    loss_mask : (B, L)  — 0=ignore, 1=thinking/response, LAMBDA=answer (cladder only)

    Shift by 1: logits[t] predicts input_ids[t+1]
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask   = loss_mask[:, 1:].contiguous()

    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.shape)

    weighted_loss = (per_token_loss * shift_mask).sum()
    denom         = (shift_mask > 0).float().sum().clamp(min=1)
    return weighted_loss / denom


# ── Data Loading ────────────────────────────────────────────────────────────────
def format_groundtruth(gt: dict) -> str:
    return "\n\n".join(
        f"## Step {i}: {gt[f'step{i}']}"
        for i in range(1, 5)
        if f"step{i}" in gt
    )

TOKENIZED_CACHE = SFT_LORA_OUTPUT_DIR / "tokenized_data_v3.pt"

def load_and_tokenize() -> tuple[SFTDataset, SFTDataset]:
    if not os.path.exists(TOKENIZED_CACHE):
        train_samples, test_samples = [], []
        for path, bucket in tqdm([(TRAIN_DATA, train_samples), (TEST_DATA, test_samples)]):
            with open(path, "r", encoding="utf-8") as f:
                for line in tqdm(f):
                    line = line.strip()
                    if not line:
                        continue
                    item   = json.loads(line)
                    source = item["source"]
                    gt     = item["groundtruth"]

                    if source == "cladder":
                        seq = build_sequence(
                            prompt   = item["prompt"],
                            thinking = format_groundtruth(gt),
                            answer   = str(gt.get("step5", "")).strip(),
                            source   = "cladder",
                        )
                        seq["source"]          = "cladder"
                        seq["groundtruth"]     = gt
                        seq["csv_path"]        = ""
                        seq["dataset_columns"] = []

                    else:  # causcibench
                        answer_json = json.dumps({"step1": gt["step1"], "step2": gt["step2"]})
                        seq = build_sequence(
                            prompt   = item["prompt"],
                            thinking = "",
                            answer   = answer_json,
                            source   = "causcibench",
                        )
                        seq["source"]      = "causcibench"
                        seq["groundtruth"] = gt

                        raw_csv = item.get("csv_path", "")
                        csv_path = _resolve_csv_path(raw_csv) if raw_csv else ""
                        seq["csv_path"]        = csv_path
                        seq["dataset_columns"] = (
                            pd.read_csv(csv_path, nrows=0).columns.tolist()
                            if csv_path else []
                        )

                    bucket.append(seq)

        n_train_cl = sum(1 for s in train_samples if s["source"] == "cladder")
        n_train_cs = sum(1 for s in train_samples if s["source"] == "causcibench")
        n_test_cl  = sum(1 for s in test_samples  if s["source"] == "cladder")
        n_test_cs  = sum(1 for s in test_samples  if s["source"] == "causcibench")
        print(f"Train: {n_train_cl} cladder  {n_train_cs} causcibench")
        print(f"Test:  {n_test_cl} cladder  {n_test_cs} causcibench")

        tmp = str(TOKENIZED_CACHE) + f".{os.getpid()}.tmp"
        torch.save({"train": train_samples, "test": test_samples}, tmp)
        os.replace(tmp, TOKENIZED_CACHE)

    saved = torch.load(TOKENIZED_CACHE, weights_only=False, map_location="cpu")
    return SFTDataset(saved["train"]), SFTDataset(saved["test"])

# ── Optimizer ──────────────────────────────────────────────────────────────────
def build_optimizer(model) -> AdamW:
    return AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

# -- Evaluate ------------------------------------------------------------------
def evaluate(ddp_model: torch.nn.Module, dataloader: DataLoader, device: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)
    ddp_model.eval()

    cladder_pending = []
    causci_pending  = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            source         = batch["source"][0]
            gt             = batch["groundtruth"][0]
            prompt_len     = batch["prompt_lens"][0]
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            item = {
                "prompt_ids":      input_ids[:, :prompt_len],
                "prompt_attn":     attention_mask[:, :prompt_len],
                "gt":              gt,
                "csv_path":        batch["csv_path"][0],
                "dataset_columns": batch["dataset_columns"][0],
            }
            if source == "cladder":
                cladder_pending.append(item)
            else:
                causci_pending.append(item)

    def _batch_generate(pending, max_new_tokens, batch_size=8):
        results = []
        intermediate_generations = []
        for i in tqdm(range(0, len(pending), batch_size), desc="Batch Generating"):
            chunk    = pending[i : i + batch_size]
            max_plen = max(p["prompt_ids"].shape[1] for p in chunk)
            batch_ids  = torch.full((len(chunk), max_plen), tokenizer.pad_token_id, dtype=torch.long, device=device)
            batch_attn = torch.zeros(len(chunk), max_plen, dtype=torch.long, device=device)
            for j, p in enumerate(chunk):
                plen = p["prompt_ids"].shape[1]
                batch_ids[j, max_plen - plen:]  = p["prompt_ids"][0]
                batch_attn[j, max_plen - plen:] = p["prompt_attn"][0]
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                generated = ddp_model.module.generate(
                    input_ids=batch_ids,
                    attention_mask=batch_attn,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                intermediate_generations.append((generated, chunk, max_plen))
        for generated, chunk, max_plen in intermediate_generations:
            for j, p in enumerate(chunk):
                text = tokenizer.decode(generated[j, max_plen:], skip_special_tokens=True)
                results.append((text, p))
        del intermediate_generations
        return results

    # Pearl's ladder of causation — mirrors eval_metrics.py
    RUNG_MAP = {
        "marginal": 1, "correlation": 1,
        "ate": 2, "backadj": 2, "collider_bias": 2, "exp_away": 2, "ett": 2,
        "det-counterfactual": 3, "nde": 3, "nie": 3,
    }

    # ── CLaDDer ────────────────────────────────────────────────────────────────
    cladder_all    = []
    cladder_step2  = []
    by_rung        = {1: [], 2: [], 3: []}
    by_cstype      = {"commonsensical": [], "nonsensical": [], "anti_commonsensical": []}

    cladder_results = _batch_generate(cladder_pending, max_new_tokens=1500)
    cladder_results = cladder_results.detach().cpu() if isinstance(cladder_results, torch.Tensor) else cladder_results
    for k, (text, item) in tqdm(enumerate(cladder_results), desc="Scoring CLaDDer"):
        if k == 0:
            print(f"\n[eval cladder sample]\n{text}\n")
        gt     = item["gt"]
        parsed = extract_cladder_sft(text)

        pred5  = (parsed or {}).get("step5", "")
        true5  = str(gt.get("step5", "")).strip().lower()
        correct = int(pred5 == true5)
        cladder_all.append(correct)

        rung = RUNG_MAP.get(gt.get("step2", ""), 0)
        if rung in by_rung:
            by_rung[rung].append(correct)

        # 3-way commonsense split (mirrors _commonsense_type in eval_metrics.py)
        story_id = gt.get("story_id") or ""
        is_cs    = gt.get("is_commonsense")
        if story_id.startswith("nonsense"):
            cstype = "nonsensical"
        elif is_cs is False:
            cstype = "anti_commonsensical"
        elif is_cs is True:
            cstype = "commonsensical"
        else:
            cstype = None
        if cstype is not None:
            by_cstype[cstype].append(correct)

        # Step 2 intermediate: query type accuracy
        pred2 = (parsed or {}).get("step2")
        if pred2 is not None:
            cladder_step2.append(int(pred2 == gt.get("step2", "").strip().lower()))

    # ── CauSciBench ────────────────────────────────────────────────────────────
    causci_method    = []
    causci_treatment = []
    causci_outcome   = []
    causci_controls  = []
    causci_effect    = []
    causci_mres      = []

    causci_results = _batch_generate(causci_pending, max_new_tokens=1500)
    causci_results = causci_results.detach().cpu() if isinstance(causci_results, torch.Tensor) else causci_results
    for k, (text, item) in tqdm(enumerate(causci_results), desc="Scoring CauSciBench"):
        if k == 0:
            print(f"\n[eval causci sample]\n{text}\n")
        gt       = item["gt"]
        csv_path = item["csv_path"]
        parsed   = extract_causci(text, item["dataset_columns"])

        if parsed is None:
            causci_method.append(0)
            continue

        pred_method = parsed.get("step2", "").strip().lower()
        gt_method   = (gt.get("step2") or "").strip().lower()
        m_ok        = int(pred_method == gt_method)
        causci_method.append(m_ok)
        if not m_ok:
            continue

        pred_s1 = parsed.get("step1") or {}
        gt_s1   = gt.get("step1") or {}

        t_ok = int(pred_s1.get("treatment", "").strip() == str(gt_s1.get("treatment", "")).strip())
        causci_treatment.append(t_ok)
        o_ok = int(pred_s1.get("outcome", "").strip() == str(gt_s1.get("outcome", "")).strip())
        causci_outcome.append(o_ok)

        # Control coverage (Jaccard recall) — mirrors eval_metrics.py
        pc = set(pred_s1.get("controls") or [])
        rc = set(gt_s1.get("controls") or [])
        if rc:
            causci_controls.append(len(pc & rc) / len(rc))
        else:
            causci_controls.append(1.0 if not pc else 0.0)

        if t_ok and o_ok and csv_path:
            parsed["step1"]["csv_path"] = csv_path
            effect, _ = library_fn(parsed)
            ref = gt.get("step5")
            if ref is not None and ref != 0:
                mre = abs(effect - ref) / abs(ref)
                causci_mres.append(mre)
                causci_effect.append(int(mre <= 0.05))

    # ── Aggregate ──────────────────────────────────────────────────────────────
    metrics = {}

    if cladder_all:
        metrics["cladder/overall_acc"] = sum(cladder_all) / len(cladder_all)
        for r in [1, 2, 3]:
            if by_rung[r]:
                metrics[f"cladder/rung{r}_acc"] = sum(by_rung[r]) / len(by_rung[r])
        for cstype, vals in by_cstype.items():
            if vals:
                metrics[f"cladder/{cstype}_acc"] = sum(vals) / len(vals)
        if cladder_step2:
            metrics["cladder/step2_acc"] = sum(cladder_step2) / len(cladder_step2)

    if causci_method:
        metrics["causci/method_acc"]   = sum(causci_method)   / len(causci_method)
    if causci_treatment:
        metrics["causci/treatment_acc"] = sum(causci_treatment) / len(causci_treatment)
    if causci_outcome:
        metrics["causci/outcome_acc"]  = sum(causci_outcome)  / len(causci_outcome)
    if causci_controls:
        metrics["causci/control_acc"]  = sum(causci_controls) / len(causci_controls)
    if causci_effect:
        metrics["causci/effect_acc"]   = sum(causci_effect)   / len(causci_effect)
    if causci_mres:
        metrics["causci/mre"]          = sum(causci_mres)     / len(causci_mres)

    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    ddp_model.train()
    print("Resuming training...\n")
    return metrics

import traceback
import datetime

# ── Training Loop ──────────────────────────────────────────────────────────────
def train(train_dataset, test_dataset):
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)

    try:
        # ── QLoRA BnB Config ───────────────────────────────────────────────────────
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_use_double_quant=True,
        )

        # ── Load & Freeze Base Model ───────────────────────────────────────────────
        model = AutoModelForCausalLM.from_pretrained(
            POLICY_MODEL,
            quantization_config=None,
            device_map={"": local_rank},
            torch_dtype=DTYPE,
            attn_implementation="flash_attention_2",
        )
        # model = prepare_model_for_kbit_training(model)

        for param in model.parameters():
            param.requires_grad = False

        # ── LoRA Config ────────────────────────────────────────────────────────────
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)

        if local_rank == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in model.parameters())
            print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        train_sampler    = DistributedSampler(train_dataset, shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=2, sampler=train_sampler, collate_fn=collate_fn, pin_memory=False)
        test_dataloader  = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, pin_memory=False)

        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

        optimizer = build_optimizer(ddp_model)
        ddp_model.train()

        yes_id = tokenizer.convert_tokens_to_ids("yes")
        no_id  = tokenizer.convert_tokens_to_ids("no")

        metric_steps      = []
        metric_train_loss = []
        metric_train_acc  = []
        eval_steps        = []
        eval_history      = {}

        global_step = 0
        window_loss, window_correct, window_total = 0.0, 0, 0
        window_causci_correct, window_causci_total = 0, 0

        for epoch in range(MAX_EPOCHS):
            train_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                loss_mask      = batch["loss_mask"].to(device)

                with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                    outputs = ddp_model(input_ids=input_ids, attention_mask=attention_mask)
                    loss    = compute_loss(outputs.logits, input_ids, loss_mask)
                    loss    = loss / GRAD_ACCUM

                # Train accuracy — cladder uses logit at answer token; causci decodes response tokens
                with torch.no_grad():
                    for i in range(input_ids.shape[0]):
                        src = batch["source"][i]

                        if src == "cladder":
                            ans_pos_list = (loss_mask[i] == ANSWER_LAMBDA).nonzero(as_tuple=True)[0]
                            if len(ans_pos_list) == 0:
                                continue
                            ans_pos  = ans_pos_list[0].item()
                            pred_idx = outputs.logits[i, ans_pos - 1, [yes_id, no_id]].argmax().item()
                            pred_tok = yes_id if pred_idx == 0 else no_id
                            if pred_tok == input_ids[i, ans_pos].item():
                                window_correct += 1
                            window_total += 1

                        else:  # causcibench
                            resp_pos = (loss_mask[i] == 1.0).nonzero(as_tuple=True)[0]
                            if len(resp_pos) == 0:
                                continue
                            pred_ids  = outputs.logits[i, resp_pos - 1, :].argmax(dim=-1)
                            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                            parsed    = extract_causci(pred_text, batch["dataset_columns"][i])
                            gt        = batch["groundtruth"][i]
                            if parsed is not None:
                                if parsed.get("step2", "").strip().lower() == (gt.get("step2") or "").strip().lower():
                                    window_causci_correct += 1
                            window_causci_total += 1

                loss.backward()
                window_loss  += loss.item() * GRAD_ACCUM
                epoch_loss   += loss.item() * GRAD_ACCUM

                if (step + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in ddp_model.parameters() if p.requires_grad],
                        MAX_GRAD_NORM,
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % LOG_EVERY == 0 and local_rank == 0:
                        avg = epoch_loss / (step + 1)
                        print(f"[epoch {epoch+1} | step {global_step}] loss: {avg:.4f}")

                    if global_step % SAVE_EVERY == 0:
                        dist.barrier()  # hold all ranks while rank 0 saves + evaluates
                        if local_rank == 0:
                            train_acc       = window_correct / window_total if window_total > 0 else 0.0
                            causci_train_acc = window_causci_correct / window_causci_total if window_causci_total > 0 else 0.0
                            train_loss_avg  = window_loss / SAVE_EVERY
                            metric_steps.append(global_step)
                            metric_train_loss.append(train_loss_avg)
                            metric_train_acc.append(train_acc)
                            window_loss, window_correct, window_total = 0.0, 0, 0
                            window_causci_correct, window_causci_total = 0, 0

                            ckpt_path = SFT_LORA_CHECKPOINT_DIR / f"step_{global_step}"
                            ddp_model.module.save_pretrained(ckpt_path)
                            tokenizer.save_pretrained(ckpt_path)
                            print(f"Checkpoint saved → {ckpt_path} | train_loss={train_loss_avg:.4f} cladder_train_acc={train_acc:.4f} causci_train_acc={causci_train_acc:.4f}")

                            torch.cuda.empty_cache()  # free up GPU memory before evaluation
                        
                        dist.barrier()  # ensure checkpoint is saved before any rank tries to evaluate

                    if global_step % EVAL_EVERY == 0:
                        dist.barrier()  # hold all ranks while rank 0 evaluates
                        if local_rank == 0:
                            step_metrics = evaluate(ddp_model, test_dataloader, device)
                            eval_steps.append(global_step)
                            for k, v in step_metrics.items():
                                eval_history.setdefault(k, []).append(v)
                            eval_history.setdefault("train/loss", []).append(train_loss_avg)
                            eval_history.setdefault("train/cladder_acc", []).append(train_acc)
                            eval_history.setdefault("train/causci_method_acc", []).append(causci_train_acc)
                            with open(SFT_LORA_PLOT_DIR / "eval_log.jsonl", "a") as f:
                                f.write(json.dumps({"step": global_step, "train/loss": train_loss_avg, "train/cladder_acc": train_acc, "train/causci_method_acc": causci_train_acc, **step_metrics}) + "\n")
                            save_eval_plots(eval_history, eval_steps, SFT_LORA_PLOT_DIR)
                        dist.barrier()  # resume all ranks together

            if local_rank == 0:
                print(f"Epoch {epoch+1} complete. Avg loss: {epoch_loss / len(train_dataloader):.4f}")

        dist.barrier()
        if local_rank == 0:
            ddp_model.module.save_pretrained(SFT_LORA_OUTPUT_DIR / "final")
            tokenizer.save_pretrained(SFT_LORA_OUTPUT_DIR / "final")
            print("Training complete.")
        dist.barrier()
        dist.destroy_process_group()

        if local_rank == 0:
            print("Running final evaluation...")
            final_metrics = evaluate(ddp_model, test_dataloader, device)
            eval_steps.append(global_step)
            for k, v in final_metrics.items():
                eval_history.setdefault(k, []).append(v)
            with open(SFT_LORA_PLOT_DIR / "eval_log.jsonl", "a") as f:
                f.write(json.dumps({"step": global_step, **final_metrics}) + "\n")
            save_eval_plots(eval_history, eval_steps, SFT_LORA_PLOT_DIR)
    except Exception as e:
        print(f"rank {local_rank} CRASHED: {e}", flush=True)
        traceback.print_exc()
        dist.destroy_process_group()
        raise

if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    sentinel = SFT_LORA_OUTPUT_DIR / ".preprocess.done"

    if local_rank == 0:
        sentinel.unlink(missing_ok=True)
        preprocess(
            cladder_prompt=CLADDER_PROMPT,
            causci_prompt=CAUSCI_USER_PROMPT,
            output_dir=SFT_LORA_OUTPUT_DIR,
        )
        sentinel.touch()
    else:
        while not sentinel.exists():
            time.sleep(1)

    train_dataset, test_dataset = load_and_tokenize()
    train(train_dataset, test_dataset)
