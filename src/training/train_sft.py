"""
SFT QLoRA Training — Qwen3-8B on CLaDDer
Loss: CE over thinking tokens + λ * CE over answer token
"""

# def train():
    # Fine tune a Lora. The model will have two components - Thinking and response
    # Thinking will have to do the reasoning, the answer will be 'yes' or 'no'
    # Loss = Next token prediction loss on thinking and Binary cross entropy on answser

    # load base model on 8bit quantization and freeze it.
    # YOUR CODE HERE:

    # load lora config and model
    # YOUR CODE HERE:

    # make sure lora parameters are the only trainable parameters
    # YOUR CODE HERE:

    # make sure lora and base model are on the same device
    # YOUR CODE HERE:

    # load training and testing data and tokenize # for this setup, we will consider only one source - cladder and ignore causci.
    # cache this to avoid tokenizing on every run.
    # YOUR CODE HERE:

    # use tqdm to track progress.
    # for each epoch:
    # YOUR CODE HERE:
        # for each batch:
        # YOUR CODE HERE:
            # take the prompt and generate answer
            # YOUR CODE HERE:
            
            # extract thinking part
            # YOUR CODE HERE:

            # extract answer part
            # YOUR CODE HERE:

            # Loss term 1: next token prediction loss on thinking part
            # YOUR CODE HERE:

            # Loss term 2: binary cross entropy on answer part (yes -> 1, no -> 0)
            # YOUR CODE HERE:

            # Backprop and optimize
            # YOUR CODE HERE:

        # Test model on test set and save metrics for plotting
        # YOUR CODE HERE:

        # Loss and accuracy logging
        # Save checkpoint every SAVE_EVERY steps
        # collect metrics for plotting
        # YOUR CODE HERE:
    
    # Save plots
    # YOUR CODE HERE:
    # pass

import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

from src.data.preprocess import preprocess
from src.config import (TRAIN_DATA_SFT_LORA as TRAIN_DATA,
                        TEST_DATA_SFT_LORA  as TEST_DATA,
                        SFT_LORA_OUTPUT_DIR,
                        SFT_LORA_PLOT_DIR, SFT_LORA_CHECKPOINT_DIR, ROOT,
                        
                        POLICY_MODEL,
                        TRAIN_BATCH_SIZE,
                        MAX_PROMPT_LEN,
                        TRAIN_MAX_TOKENS,
                        LR,
                        WEIGHT_DECAY,
                        GRAD_ACCUM,
                        MAX_GRAD_NORM,
                        SAVE_EVERY,
                        LOG_EVERY,
                        LORA_R,
                        MAX_EPOCHS)


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

CAUSCI_PROMPT = """You are given a dataset from a research study along with a description of how the data was collected. Your task is to estimate the effect of one variable on another by following these steps precisely.

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

Use the following reference to guide your reasoning.

### Method Definitions

1. **diff_in_means (Difference in Means)**
   When to use: The data comes from a randomized experiment where units were randomly assigned to treatment or control, and compliance was enforced. Random assignment ensures both groups are comparable on average.
   Estimand: ATE (Average Treatment Effect)
   Formula: τ = (1/n₁)Σ Yᵢ(treated) - (1/n₀)Σ Yᵢ(control)
   Equivalent regression: Y = α + τT + ε. The coefficient on T is the treatment effect.
   If pre-treatment covariates are available, include them to improve precision: Y = α + τT + Xβ + ε. The coefficient on T remains the causal effect.

2. **ols (Ordinary Least Squares with Controls)**
   When to use: Observational data where all confounders (variables affecting both treatment and outcome) are observed and included as controls. No unobserved confounding.
   Estimand: ATE
   Formula: Y = α + τT + Xβ + ε, where X includes all confounders. The coefficient τ on T is the causal effect.
   Key assumption: Conditional ignorability — Y(0),Y(1) ⊥ T | X. After controlling for X, treatment assignment is as good as random.
   Warning: If there are unobserved confounders, OLS is biased. Consider IV or other methods.

3. **ipw (Inverse Probability Weighting)**
   When to use: Observational data where treatment is not random but confounders are observed. Particularly useful when the treatment model (propensity score) is well-specified.
   Estimand: ATE, ATT, or ATC depending on the question.
   Formula for ATE: τ_ATE = [Σ Yᵢ·Tᵢ/e(Xᵢ)] / [Σ Tᵢ/e(Xᵢ)] - [Σ Yᵢ·(1-Tᵢ)/(1-e(Xᵢ))] / [Σ (1-Tᵢ)/(1-e(Xᵢ))]
   where e(X) is the propensity score, estimated via logistic regression of T on X.
   Key assumption: Conditional ignorability (same as OLS) plus overlap — every unit must have nonzero probability of receiving either treatment level: 0 < e(X) < 1.
   Warning: Unstable when propensity scores are near 0 or 1. Consider matching instead.

4. **matching (Propensity Score Matching)**
   When to use: Observational data with observed confounders. Preferred over IPW when propensity score overlap is poor. Think of it as a preprocessing step that makes treatment and control groups more comparable.
   Estimand: ATE or ATT.
   Procedure: For each treated unit, find the nearest control unit(s) based on covariates or propensity score. Compute effect as average difference in outcomes between matched pairs.
   Formula for ATT: τ_ATT = (1/n₁) Σᵢ∈treated (Yᵢ - (1/K) Σₖ Y_matched_k)
   Key assumption: Conditional ignorability plus overlap, same as IPW.

5. **did (Difference-in-Differences)**
   When to use: Panel data (observations over multiple time periods) where a treatment was introduced to one group at a specific time. There must be a clear pre-period and post-period, and a treatment group versus control group.
   Estimand: ATT (Average Treatment Effect on the Treated)
   Formula (canonical 2×2): Y = α + β·POST + γ·TREAT + δ·(POST × TREAT) + Xβ + ε. The coefficient δ is the DiD estimator.
   Formula (TWFE, staggered treatment): Y_it = αᵢ + λₜ + δ·D_it + X_it·β + ε_it. The coefficient δ is the effect. αᵢ are unit fixed effects, λₜ are time fixed effects.
   Key assumptions: Parallel trends — in the absence of treatment, treated and control groups would have followed the same trajectory. No anticipatory effects.
   How to identify: Look for a time variable that indicates treatment timing (not just a covariate), and group indicators for who received treatment.

6. **rdd (Regression Discontinuity Design)**
   When to use: Treatment is assigned based on whether a continuous variable (the running variable) crosses a threshold/cutoff. Units just above and below the cutoff are comparable.
   Estimand: Local ATE (at the cutoff)
   Formula: τ_RDD = lim(r→r₀⁺) E[Y|R=r] - lim(r→r₀⁻) E[Y|R=r]
   Key assumption: Potential outcomes are continuous at the cutoff. The only thing that changes discontinuously at the threshold is treatment status.
   How to identify: Look for a continuous variable where a threshold determines eligibility or assignment. Examples: test scores determining program eligibility, age cutoffs, income thresholds.

7. **iv (Instrumental Variables / Two-Stage Least Squares)**
   When to use: Unobserved confounders exist between treatment and outcome, but an instrument is available. The instrument must affect the outcome only through the treatment.
   Estimand: LATE (Local Average Treatment Effect) or CACE (Complier Average Causal Effect)
   Procedure: Stage 1 — regress treatment T on instrument Z (and controls X): T = π₀ + π₁Z + Xγ + ν. Stage 2 — regress outcome Y on predicted treatment T̂ (and controls X): Y = β₀ + τT̂ + Xδ + ε. The coefficient τ is the causal effect.
   Key assumptions: (1) Relevance — Z is correlated with T (testable: first-stage F-statistic > 10). (2) Exclusion restriction — Z affects Y only through T (untestable, requires domain justification). (3) Independence — Z is independent of unobserved confounders. (4) Monotonicity — Z moves T in the same direction for everyone.
   How to identify: Look for a variable that plausibly affects treatment uptake but has no direct effect on the outcome. Common examples: geographic proximity as instrument for schooling, lottery assignments as instruments for program participation.

8. **frontdoor (Frontdoor Adjustment)**
   When to use: Unobserved confounders exist between treatment and outcome, but a mediator M exists such that (1) T → M → Y captures the full causal path, (2) there are no unobserved confounders between T and M, and (3) there are no unobserved confounders between M and Y after controlling for T.
   Estimand: ATE
   Formula: P(Y|do(T)) = Σ_m P(M=m|T) · Σ_t P(Y|M=m, T=t) · P(T=t)
   How to identify: Look for a mediator that fully transmits the treatment's effect. Rare in practice. The data description may mention an intermediate step or mechanism.

9. **glm (Generalized Linear Model)**
   When to use: The outcome is non-linear — binary (logistic regression), count data (Poisson regression), bounded/proportional (beta regression). Confounders are observed.
   Estimand: Conditional effect (log-odds ratio, incidence rate ratio, etc., depending on the link function)
   Formula: g(E[Y]) = α + τT + Xβ, where g() is the link function (logit for binary, log for counts).
   The coefficient τ represents the effect of treatment on the transformed outcome scale.
   How to identify: Check the outcome variable. If it's binary (0/1), use logistic regression. If it's a count (0, 1, 2, ...), consider Poisson. If it's continuous and unbounded, OLS is likely more appropriate.

---

Respond with the five numbered steps below in order. Do not write any introduction, explanation, or preamble before Step 1. Write each step exactly once. Stop after Step 5.

## Step 1: Causal Structure
Using the study description and dataset columns, identify:
- treatment: <column_name>
- outcome: <column_name>
- controls: [<col1>, <col2>, ...]
- instrument: <column_name> or none
- running_variable: <column_name> or none
- time_variable: <column_name> or none
- group_variable: <column_name> or none

## Step 2: Method Selection
Based on the study description, data collection process, and the method definitions above, select the most appropriate method. Return exactly one of:
diff_in_means, ols, ipw, matching, did, rdd, iv, frontdoor, glm

Justify in one sentence based on the study design and the assumptions that can be invoked.

## Step 3: Estimation Specification
Write the formal estimation setup:
- The regression formula or procedure
- The estimand (ATE, ATT, LATE, etc.)
- The key identification assumption being invoked

## Step 4: Compute
Using the estimation specification from Step 3 and the data summary above (column types, summary statistics, sample rows), compute the effect estimate numerically. Show the arithmetic step by step — substitute values and simplify to a final number.

## Step 5: Answer
Report the estimated effect as a single number.
"""

# ── Paths ──────────────────────────────────────────────────────────────────────

for d in [SFT_LORA_OUTPUT_DIR, SFT_LORA_PLOT_DIR, SFT_LORA_CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
LORA_R             = 32
LORA_ALPHA         = 64
LORA_DROPOUT       = 0.05
ANSWER_LAMBDA      = 5.0
DTYPE              = torch.bfloat16
DEVICE             = "cuda"

# ── Tokenizer ──────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL)
tokenizer.padding_side = "right"

# Special token IDs we need for loss masking
THINK_CLOSE_STR = "<|im_end|>"   # end of assistant turn
IM_END_ID       = tokenizer.convert_tokens_to_ids("<|im_end|>")
# </think> token — Qwen3 uses token id 151668
THINK_CLOSE_ID  = tokenizer.convert_tokens_to_ids("</think>")

# ── QLoRA BnB Config ───────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=DTYPE,
    bnb_4bit_use_double_quant=True,
)

# ── Load & Freeze Base Model ───────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    POLICY_MODEL,
    quantization_config=bnb_config,
    device_map={"": DEVICE},
    torch_dtype=DTYPE,
    attn_implementation="flash_attention_2",
)
model = prepare_model_for_kbit_training(model)

# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# ── LoRA Config ────────────────────────────────────────────────────────────────
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
model = torch.compile(model)
# model.gradient_checkpointing_enable()

# Confirm only LoRA params are trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# ── Build Full Sequence ────────────────────────────────────────────────────────
def build_sequence(prompt: str, thinking: str, answer: str) -> dict:
    """
    Constructs the full token sequence for one training sample.

    Layout:
        [prompt tokens] [<think>\n thinking \n</think>\n] [answer] [<|im_end|>\n]

    Returns:
        input_ids  : (L,)   full token sequence
        loss_mask  : (L,)   0=ignore, 1=thinking CE, LAMBDA=answer CE
    """
    messages = [{"role": "user", "content": prompt}]

    # apply_chat_template with enable_thinking=True ends with:
    # <|im_start|>assistant\n<think>\n
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    # Full target: thinking content + closing tags + answer + end token
    # The prompt_text already opens <think>, so we continue from there
    response_text = thinking + "\n</think>" + answer + "<|im_end|>\n"
    full_text     = prompt_text + response_text

    # Tokenize full sequence (no truncation yet — we handle it below)
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

    # Truncate if over budget
    max_len = MAX_PROMPT_LEN + TRAIN_MAX_TOKENS
    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]

    seq_len   = len(full_ids)
    loss_mask = torch.zeros(seq_len, dtype=DTYPE)

    # Find </think> position in the full sequence
    think_close_pos = None
    for i in range(prompt_len, seq_len):
        if full_ids[i].item() == THINK_CLOSE_ID:
            think_close_pos = i
            break

    if think_close_pos is not None:
        # Thinking tokens: prompt_len → think_close_pos (inclusive of </think>)
        loss_mask[prompt_len : think_close_pos + 1] = 1.0
        # Answer token: immediately after </think>
        answer_pos = think_close_pos + 1
        if answer_pos < seq_len:
            loss_mask[answer_pos] = ANSWER_LAMBDA
    else:
        # Fallback: CE on everything after prompt
        loss_mask[prompt_len:] = 1.0

    return {"input_ids": full_ids, "loss_mask": loss_mask}


# ── Dataset ────────────────────────────────────────────────────────────────────
class CladderDataset(Dataset):
    def __init__(self, samples: list[dict]):
        """
        Each sample dict must have:
            "input_ids"  : torch.LongTensor  (L,)
            "loss_mask"  : torch.FloatTensor (L,)
        """
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
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "loss_mask":      loss_masks,
    }


# ── Custom Loss ────────────────────────────────────────────────────────────────
def compute_loss(logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """
    logits    : (B, L, V)
    input_ids : (B, L)
    loss_mask : (B, L)  — 0=ignore, 1=thinking, LAMBDA=answer

    Shift by 1: logits[t] predicts input_ids[t+1]
    """
    shift_logits = logits[:, :-1, :].contiguous()           # (B, L-1, V)
    shift_labels = input_ids[:, 1:].contiguous()             # (B, L-1)
    shift_mask   = loss_mask[:, 1:].contiguous()             # (B, L-1)

    # Per-token CE, no reduction
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.shape)                               # (B, L-1)

    # Apply mask (0 = no gradient, 1 = full CE, LAMBDA = upweighted)
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

def load_and_tokenize_cladder() -> tuple[CladderDataset, CladderDataset]:
    train_samples, test_samples = [], []

    for path, bucket in tqdm([(TRAIN_DATA, train_samples), (TEST_DATA, test_samples)]):
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                if item["source"] != "cladder":
                    continue
                seq = build_sequence(
                    prompt   = item["prompt"],
                    thinking = format_groundtruth(item["groundtruth"]),
                    answer   = item["label"],
                )
                bucket.append(seq)

    print(f"Train: {len(train_samples)} | Test: {len(test_samples)} CLaDDer samples.")
    return CladderDataset(train_samples), CladderDataset(test_samples)

# ── Optimizer ──────────────────────────────────────────────────────────────────
def build_optimizer(model) -> AdamW:
    return AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

# -- Evalluate ------------------------------------------------------------------
def evaluate(dataset: CladderDataset) -> float:
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    correct, total = 0, 0
    yes_id = tokenizer.convert_tokens_to_ids("Yes")
    no_id  = tokenizer.convert_tokens_to_ids("No")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            loss_mask      = batch["loss_mask"][0]

            # Find answer token position (where loss_mask == ANSWER_LAMBDA)
            answer_positions = (loss_mask == ANSWER_LAMBDA).nonzero(as_tuple=True)[0]
            if len(answer_positions) == 0:
                continue
            answer_pos = answer_positions[0].item()

            with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # logits at answer_pos - 1 predicts the token at answer_pos
            answer_logits = outputs.logits[0, answer_pos - 1, :]
            pred_id       = answer_logits[[yes_id, no_id]].argmax().item()
            pred_label    = "Yes" if pred_id == 0 else "No"
            true_label    = tokenizer.decode(input_ids[0, answer_pos]).strip()

            if pred_label.lower() == true_label.lower():
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {correct}/{total} = {accuracy:.4f}")
    model.train()
    return accuracy

# ── Training Loop ──────────────────────────────────────────────────────────────
def train():
    train_dataset, test_dataset = load_and_tokenize_cladder()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    optimizer  = build_optimizer(model)
    model.train()

    global_step = 0
    for epoch in range(MAX_EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            loss_mask      = batch["loss_mask"].to(DEVICE)

            with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = compute_loss(outputs.logits, input_ids, loss_mask)
                loss    = loss / GRAD_ACCUM

            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    MAX_GRAD_NORM,
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % LOG_EVERY == 0:
                    avg = epoch_loss / (step + 1)
                    print(f"[epoch {epoch+1} | step {global_step}] loss: {avg:.4f}")

                if global_step % SAVE_EVERY == 0:
                    ckpt_path = SFT_LORA_CHECKPOINT_DIR / f"step_{global_step}"
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    print(f"Checkpoint saved → {ckpt_path}")

        print(f"Epoch {epoch+1} complete. Avg loss: {epoch_loss / len(train_dataloader):.4f}")
        # Evaluate at the end of each epoch
        evaluate(test_dataset)

    # Final save
    model.save_pretrained(SFT_LORA_OUTPUT_DIR / "final")
    tokenizer.save_pretrained(SFT_LORA_OUTPUT_DIR / "final")
    print("Training complete.")


if __name__ == "__main__":
    preprocess(cladder_prompt = CLADDER_PROMPT, causci_prompt = CAUSCI_PROMPT, output_dir = SFT_LORA_OUTPUT_DIR)
    train()