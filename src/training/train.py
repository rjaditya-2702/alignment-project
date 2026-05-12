"""
GRPO training script with vLLM acceleration (TRL).

Fill in the two clearly marked sections:
  1. DATA LOADING  — load your dataset and convert to the expected format
  2. REWARD FUNCTION — parse the model's answer and return a score

Expected dataset format (list of dicts):
    [
        {
            "prompt": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": "What is 2 + 2?"},
            ]
        },
        ...
    ]
Each item must have a "prompt" key whose value is a list of chat messages.
The model will generate completions for each prompt during training.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.data.preprocess import preprocess

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

from src.config import (
    POLICY_MODEL as MODEL_NAME, 
    OUTPUT_DIR_RL,
    CHECKPOINT_DIR as OUTPUT_DIR,
    JUDGE_MODEL,
    TRAIN_DATA_RL as TRAIN_DATA,
    TEST_DATA_RL as TEST_DATA,
    TRAIN_BATCH_SIZE,
    N_ROLLOUTS,
    FINAL_MODEL
)

training_args = GRPOConfig(
    # --- core ---
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,     # prompts per GPU per step
    gradient_accumulation_steps=1,

    # --- GRPO-specific ---
    num_generations=N_ROLLOUTS,                 # N rollouts per prompt (the group size)
    max_prompt_length=4096,
    max_completion_length=1200,

    # --- KL penalty ---
    beta=0.04,                         # weight on KL(π_θ ∥ π_ref); 0 disables it

    # --- vLLM ---
    use_vllm=True,
    vllm_mode="colocate",              # shares GPUs with training; use "server"
                                       # if you have dedicated inference GPUs
    vllm_gpu_memory_utilization=0.4,   # leave headroom for training weights

    # --- logging / saving ---
    logging_steps=10,
    save_steps=100,
    report_to="none",                  # swap to "wandb" or "tensorboard"

    # --- misc ---
    bf16=True,
    seed=42,
)

# ---------------------------------------------------------------------------
# 1. DATA LOADING  ← fill this in
# ---------------------------------------------------------------------------
CLADDER_PROMPT = """You are a causal inference expert. You are given a scenario describing relationships between variables, along with numerical data and a question. Your task is to perform causal reasoning and answer by following these steps precisely.
---
Strict rules (follow these exactly):
- Nothing before "## Step 1" and nothing after the JSON block.
- Write each step exactly once.
- Each step must be short and direct. No long paragraphs or verbosity.
- Do not repeat content from previous steps.
- Output Steps 1–4 inside the thinking block only.
- After Step 4, close the thinking block.
- After </think>, output the JSON object. No quotes, no punctuation, no extra text.
- Stop immediately after that JSON object completion.
- Do not repeat any step, any code block.

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

Based on the computed result and what the question is asking, return a JSON payload with each step.
- For ate/ett/nde/nie: positive result → Yes if question asks "does X increase Y", No if "decrease". Flip if question asks the opposite.
- For marginal: P(Y) > 0.5 and question asks "is Y more likely than not" → Yes.
- For correlation: P(Y|X=1) > P(Y|X=0) and question asks "does observing X increase Y" → Yes.
- For backadj/collider_bias/exp_away: Yes or No based on graph analysis.
- For det-counterfactual: Yes or No based on computed probability.
---

Now solve the problem in the following way:
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

## Step 4: Compute
Using the estimand from Step 3 and the numerical values given in the Data section, compute the result step by step. Show the arithmetic explicitly — substitute each probability value and simplify to a final number.
Show your derivation.

## Step 5: Answer - look at the scenario and the question, and decide whether the answer is yes or no based on the computed result and what the question is asking.
</think>

{
    "step1" : "Causal Structure, nodes and edges defined",
    "step2" : "Query classified as one of the 10 types",
    "step3" : "Mathematical estimand derived using do-calculus or graph reasoning",
    "step4" : "Numerical computation shown with all steps",
    "step5" : "yes/no"
}

IMPORTANT: After the closing } of the JSON object, STOP. No more text is allowed.

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


class CladderDataset(Dataset):
    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"prompt": self.sequences[idx]}  # GRPOTrainer expects a "prompt" key

def get_reasoning_from_groundtruth(gt: dict) -> str:
    return "\n\n".join(
        f"## Step {i}: {gt[f'step{i}']}"
        for i in range(1, 5)
        if f"step{i}" in gt
    )

def load_dataset_for_grpo() -> CladderDataset:
    """
    Load your data and return a HuggingFace Dataset whose rows each look like:

        {
            "prompt": [
                {"role": "system", "content": "<system prompt>"},
                {"role": "user",   "content": "<question>"},
            ]
        }

    The "prompt" value is a list-of-dicts in OpenAI chat format.
    GRPOTrainer will apply the model's chat template automatically.

    You can add extra keys (e.g. "ground_truth") — they will be passed
    as keyword arguments to your reward function, so you can use them
    for scoring without hard-coding answers.
    """

    import json
    with open(TRAIN_DATA, "r") as f:
        raw = [json.loads(line) for line in f]
    
    rows = [r for r in raw if r["source"] != "causcibench"]  # optional filtering
    new_data = []
    for r in rows:
        prompt = [
            {"role": "system", "content": "You are a helpful assistant. Solve the problem. Wrap your final answer in <answer>...</answer>."},
            {"role": "user",   "content": r["prompt"]},
        ]
        label = r["label"]
        reasoning = get_reasoning_from_groundtruth(r["groundtruth"])
        new_data.append({
            "prompt": prompt,
            # "ground_truth": r["groundtruth"],
            "label": label,
            "reasoning": reasoning,
        })
    return CladderDataset(new_data)


# ---------------------------------------------------------------------------
# 2. REWARD FUNCTION  ← fill this in
# ---------------------------------------------------------------------------

def reward_fn(completions: list[str], **kwargs) -> list[float]:
    """
    Score each completion and return a list of floats (one per completion).

    Args:
        completions:  list of generated strings (length = batch * num_generations)
        **kwargs:     any extra columns from your dataset are forwarded here,
                      e.g. kwargs["ground_truth"] gives you the reference answers
                      as a list aligned with `completions`

    Returns:
        list of float rewards, same length as `completions`

    Tips:
        - Keep rewards bounded (e.g. 0.0 – 1.0 or -1.0 – 1.0).
        - You can combine multiple signals (format + correctness + length, etc.).
        - Avoid NaN / Inf — they will silently corrupt training.
    """

    # ------------------------------------------------------------------ #
    # YOUR CODE HERE                                                        #
    # Example: reward 1.0 if the answer tag contains the correct number,  #
    #          0.0 otherwise (binary reward for math problems).            #
    #                                                                       #
    #   ground_truths = kwargs.get("ground_truth", [None] * len(completions))
    #   rewards = []                                                        #
    #   for completion, truth in zip(completions, ground_truths):          #
    #       match = re.search(r"<answer>(.*?)</answer>", completion, re.S) #
    #       if match and truth is not None:                                 #
    #           predicted = match.group(1).strip()                         #
    #           rewards.append(1.0 if predicted == str(truth) else 0.0)   #
    #       else:                                                           #
    #           rewards.append(0.0)   # no answer tag → no reward          #
    #   return rewards                                                      #
    # ------------------------------------------------------------------ #

    raise NotImplementedError("Fill in reward_fn()")


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
    dataset = load_dataset_for_grpo()
    assert "prompt" in dataset.column_names, (
        "Dataset must have a 'prompt' column (list of chat messages per row)."
    )

    # optional: quick sanity check on reward function with dummy data
    # dummy = ["<answer>42</answer>"] * training_args.num_generations
    # print("Reward sanity check:", reward_fn(dummy))

    # --- trainer ---
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        args=training_args,
        reward_funcs=reward_fn,        # can also be a list for multiple signals
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    # save checkpoints to OUTPUT_DIR
    trainer.save_model(OUTPUT_DIR)

    # save final model after training to FINAL_MODEL
    trainer.save_model(FINAL_MODEL)

    print(f"Model saved to {FINAL_MODEL}")




if __name__ == "__main__":
    preprocess(cladder_prompt=CLADDER_PROMPT, causci_prompt=CAUSCI_PROMPT, output_dir=OUTPUT_DIR_RL)
    main()

# import json
# import os
# import multiprocessing as mp
# import random
# import sys
# import time
# import numpy
# import bitsandbytes as bnb
# from pathlib import Path
# from tqdm import tqdm

# ROOT = Path(__file__).resolve().parents[2]
# sys.path.insert(0, str(ROOT))

# from src.config import (
#         POLICY_MODEL, JUDGE_MODEL, TRAIN_DATA, CHECKPOINT_DIR,
#         TRAIN_BATCH_SIZE as BATCH_SIZE, N_ROLLOUTS, MAX_PROMPT_LEN,
#         TRAIN_MAX_TOKENS as MAX_NEW_TOKENS, TEMPERATURE, TOP_P,
#         BETA, LR, WEIGHT_DECAY, GRAD_ACCUM, MAX_GRAD_NORM,
#         MAX_EPOCHS, SAVE_EVERY, LOG_EVERY, LORA_R as R,
#         IS_CLIP_RANGE,
#         PLOT_DIR,
#     )
# from src.training.reward import compute_rewards
# import torch
# from peft import LoraConfig, get_peft_model
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from transformers import Qwen2Tokenizer as _Qwen2Tok
# if not hasattr(_Qwen2Tok, "all_special_tokens_extended"):
#     _Qwen2Tok.all_special_tokens_extended = property(lambda self: self.all_special_tokens)


# # set seed for reproducibility
# SEED = 42
# random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# numpy.random.seed(SEED)
# # pandas.util.testing.rands.seed(SEED)
# # any other libraries with randomness should be seeded here as well


# def format_prompt(tokenizer, prompt: str) -> str:
#     """
#     Helper function: Format a prompt into the style expected by the policy model. 
#     This is important for consistent tokenization and to ensure the model generates in the right format.
#     For Qwen-2.5, we use a simple chat template with system and user roles. 
#     If the tokenizer supports an "apply_chat_template" method, we use that for better compatibility
#     """
    
#     return tokenizer.apply_chat_template(
#         [{"role": "user", "content": prompt}],
#         tokenize=False,
#         add_generation_prompt=True,
#         enable_thinking=False,
#     )

# def _find_latest_checkpoint(out_dir: Path):
#     """
#     Helper function:
#     Find the latest checkpoint in the output directory by parsing subdirectory names.
#     Checkpoint subdirectories are expected to be named in the format "step_{number}".
#     Returns the path to the latest checkpoint and its step number, or (None, 0) if no checkpoints are found.
#     """
#     latest_step, latest_ckpt = 0, None
#     for p in out_dir.glob("step_*"):
#         if p.is_dir():
#             try:
#                 step = int(p.name.split("_")[1])
#                 if step > latest_step:
#                     latest_step, latest_ckpt = step, p
#             except (IndexError, ValueError):
#                 pass
#     return latest_ckpt, latest_step

# def get_device_map(model_name: str, policy_gpu_frac: float) -> dict:
#     """
#     Helper function:
#     Get a device map for loading a model with Hugging Face's accelerate library. 
#     This allows us to specify which GPUs to use for different parts of the model.
#     """
#     from transformers import AutoConfig
#     config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
#     num_layers = config.num_hidden_layers
#     split_at = int(num_layers * policy_gpu_frac)
#     device_map = {
#         "model.embed_tokens": 1,
#         "model.norm": 1,
#         "lm_head": 2
#     }
#     for i in range(num_layers):
#         device_map[f"model.layers.{i}"] = 1 if i < split_at else 2
#     return device_map

# def _fix_rotary_device(model):
#     """
#     After PEFT wraps the model, inv_freq buffers in rotary embeddings may remain on CPU
#     while the rest of the layer is on CUDA. This moves each inv_freq to the same device
#     as its parent attention layer's parameters.
#     """
#     for name, buf in model.named_buffers():
#         if 'inv_freq' not in name or buf.device.type != 'cpu':
#             continue
#         parent_name = name.rsplit('.rotary_emb.inv_freq', 1)[0]
#         try:
#             parent = model.get_submodule(parent_name)
#             dev = next(p.device for p in parent.parameters() if p.device.type != 'cpu')
#             rotary = model.get_submodule(name.rsplit('.inv_freq', 1)[0])
#             rotary.inv_freq = buf.to(dev)
#         except (AttributeError, StopIteration):
#             pass

# def _batched_logprobs(model, prompt_ids_list, comp_ids_list, pad_id, device):
#     """
#     Helper function:
#     Compute mean per-token log-prob of each completion under model.
#     Returns a list of scalar tensors, preserving grad if called outside torch.no_grad().
#     """
#     BN = len(prompt_ids_list)
#     full_ids = [torch.cat([p.to(device), c.to(device)]) for p, c in zip(prompt_ids_list, comp_ids_list)]
#     max_len = max(f.shape[0] for f in full_ids)
#     padded = torch.full((BN, max_len), pad_id, dtype=torch.long, device=device)
#     attn_mask = torch.zeros(BN, max_len, dtype=torch.long, device=device)
#     for i, f in enumerate(full_ids):
#         padded[i, :f.shape[0]] = f
#         attn_mask[i, :f.shape[0]] = 1
#     logits = model(input_ids=padded, attention_mask=attn_mask, use_cache=False).logits
#     shift_logits = logits[:, :-1]
#     labels = padded[:, 1:]
#     log_probs = (
#         shift_logits.gather(2, labels.unsqueeze(2)).squeeze(2)
#         - torch.logsumexp(shift_logits, dim=-1)
#     )
#     del logits, shift_logits
#     result = []
#     for i, (p, c) in enumerate(zip(prompt_ids_list, comp_ids_list)):
#         C, P = c.shape[0], p.shape[0]
#         if C == 0:
#             result.append(torch.zeros(1, device=device).squeeze())
#         else:
#             result.append(log_probs[i, P - 1 : P - 1 + C].mean())
#     return result

# def _format_gt_completion(row):
#     """
#     Helper function: 
#     Format the groundtruth dict into a completion string matching model output format.
#     """
#     gt = row["groundtruth"]
#     return (
#         f"## Step 1: Causal Structure\n{gt.get('step1', '')}\n\n"
#         f"## Step 2: Query Classification\n{gt.get('step2', '')}\n\n"
#         f"## Step 3: Derive Estimand\n{gt.get('step3', '')}\n\n"
#         f"## Step 4: Compute\n{gt.get('step4', '')}\n\n"
#         f"## Step 5: Answer\n{gt.get('step5', '')}"
#     )

# def generate_rollouts(model, tokenized_batch, tokenizer):
#     """
#     Helper function:
#     Generate rollouts for a batch of tokenized prompts using the policy model. 
#     This is called by policy_updates() to get completions for each prompt in the batch.
#     """
#     device = next(model.parameters()).device
#     completions, prompt_token_ids, comp_token_ids, gen_logprobs = [], [], [], []

#     with torch.no_grad():
#         for row in tokenized_batch:
#             # generate responses for each tokenized prompt in the batch.
#             inp_ids = row["input_ids"].unsqueeze(0).to(device)  # [1, P]
#             p_len = inp_ids.shape[1]

#             outputs = model.generate(
#                 inp_ids.expand(N_ROLLOUTS, -1),
#                 attention_mask=torch.ones(N_ROLLOUTS, p_len, device=device),
#                 max_new_tokens=MAX_NEW_TOKENS,
#                 temperature=TEMPERATURE,
#                 top_p=TOP_P,
#                 do_sample=True,
#                 pad_token_id=tokenizer.pad_token_id,
#                 use_cache=True,
#                 return_dict_in_generate=True,
#                 output_scores=True,
#                 repetition_penalty=1.1,
#             )  # outputs.sequences: [N_ROLLOUTS, P+C]

#             sequences = outputs.sequences
#             # collect the final layer probs - i.e. pi_policy(a|s) - for KL penalty calculation later
#             # compute_transition_scores gives per-token log probs of the chosen tokens
#             transition_scores = model.compute_transition_scores(
#                 sequences, outputs.scores, normalize_logits=True
#             )  # [N_ROLLOUTS, C]
#             # normalize_logits=True can produce nan when log_softmax is applied to an
#             # all-(-inf) score vector (e.g. degenerate top-p nucleus after repetition
#             # penalty), so clamp before we store anything.
#             transition_scores = torch.nan_to_num(transition_scores, nan=0.0, posinf=0.0, neginf=-100.0)

#             for j in range(N_ROLLOUTS):
#                 c_ids = sequences[j, p_len:]
#                 completions.append(tokenizer.decode(c_ids, skip_special_tokens=True))
#                 prompt_token_ids.append(row["input_ids"].cpu())
#                 comp_token_ids.append(c_ids.cpu())
#                 gen_logprobs.append(transition_scores[j].mean().cpu())

#             del outputs, sequences, transition_scores

#     # return completions and token ids for reward scoring and logprob re-computation
#     return completions, prompt_token_ids, comp_token_ids, gen_logprobs

# def policy_updates(policy_model, reference_model, judge_input_queue, judge_output_queue, judge_process, all_rows, start_step=0,):
#     """
#     Main training loop: iterates through the training data, generates completions, gets rewards from the judge, 
#     and updates the policy model using GRPO. This function is called by train() after loading models and data.
#     """
#     # GPU setup: policy_model on GPUs 1-2, 
#     # judge_model on GPU 0, 
#     # reference_model on GPU 3 
#     # (pinned in train_init)

#     policy_model.train()
#     reference_model.eval()  # reference is fixed for KL penalty
#     optimizer = bnb.optim.AdamW(filter(lambda p: p.requires_grad, policy_model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

#     global_step = start_step
#     global_batch_idx = 0

#     policy_tok = AutoTokenizer.from_pretrained(POLICY_MODEL, trust_remote_code=True)
#     policy_tok.pad_token = policy_tok.eos_token if policy_tok.pad_token is None else policy_tok.pad_token
#     policy_tok.padding_side = "left"

#     import matplotlib
#     matplotlib.use("Agg")  # headless — no display on GPU servers
#     import matplotlib.pyplot as plt
#     PLOT_DIR.mkdir(parents=True, exist_ok=True)

#     plot_steps, plot_loss, plot_reward, plot_kl = [], [], [], []

#     all_rows = [r for r in all_rows if r["source"] != "causcibench"]

#     for epoch in tqdm(range(MAX_EPOCHS)):
#         epoch_rows = list(all_rows)
#         random.shuffle(epoch_rows)  # shuffle data each epoch
#         batches = [epoch_rows[i:i+BATCH_SIZE] for i in range(0, len(epoch_rows), BATCH_SIZE)]

#         print(f"[Data] {len(all_rows)} samples, {len(batches)} batches, {N_ROLLOUTS} rollouts each")

#         optimizer.zero_grad()
#         progress_bar = tqdm(total=len(batches), desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", dynamic_ncols=True)
#         for batch_idx_local, batch_rows in enumerate(tqdm(batches)):
#             batch_idx = global_batch_idx + batch_idx_local  # absolute batch index across epochs
#             B = len(batch_rows)

#             t0 = time.time()

#             # 1. Generate N rollouts for the batch using the current policy model
#             policy_model.eval()
#             completions, prompt_ids, comp_ids, gen_lps = generate_rollouts(policy_model, batch_rows, policy_tok)
#             policy_model.train()

#             # 2. Compute rewards for the generated rollouts using the judge model
#             flat_rows = [r for r in batch_rows for _ in range(N_ROLLOUTS)]
#             rewards = compute_rewards(completions, flat_rows, judge_input_queue, judge_output_queue, judge_process, ground_truth_key="groundtruth")

#             # 2.1 insert ground truth answer as N+1 completion and assign reward of 100 to it
#             gt_texts = [_format_gt_completion(row) for row in batch_rows]
#             gt_enc = policy_tok(gt_texts, truncation=True, max_length=MAX_NEW_TOKENS, padding=False, return_tensors=None)
#             gt_comp_ids  = [torch.tensor(ids, dtype=torch.long) for ids in gt_enc["input_ids"]]
#             gt_prompt_ids = [row["input_ids"].cpu() for row in batch_rows]
#             pad_id = policy_tok.pad_token_id
#             policy_device = "cuda:2"

#             # interleave rollouts and GT so view(B, group_size) gives correct per-prompt groups
#             group_size = N_ROLLOUTS + 1
#             ext_prompt_ids, ext_comp_ids, ext_rewards = [], [], []
#             for b in range(B):
#                 for k in range(N_ROLLOUTS):
#                     ext_prompt_ids.append(prompt_ids[b * N_ROLLOUTS + k])
#                     ext_comp_ids.append(comp_ids[b * N_ROLLOUTS + k])
#                     ext_rewards.append(rewards[b * N_ROLLOUTS + k])
#                 ext_prompt_ids.append(gt_prompt_ids[b])
#                 ext_comp_ids.append(gt_comp_ids[b])
#                 ext_rewards.append(100.0)

#             # 3. Compute KL divergence between policy and reference model for the generated N rollouts
#             ## 3.1 Get reference model probabilities for the generated rollouts
#             with torch.no_grad():
#                 ref_lps = _batched_logprobs(reference_model, ext_prompt_ids, ext_comp_ids, pad_id, "cuda:3")

#             ## 3.2 Get policy model probabilities for the generated rollouts
#             # rollouts computed with grad (for backprop); GT without grad (not trained on)
#             policy_lps_rollout = _batched_logprobs(
#                 policy_model, ext_prompt_ids[: B * N_ROLLOUTS], ext_comp_ids[: B * N_ROLLOUTS], pad_id, policy_device
#             )
#             with torch.no_grad():
#                 policy_lps_gt = _batched_logprobs(
#                     policy_model, ext_prompt_ids[B * N_ROLLOUTS :], ext_comp_ids[B * N_ROLLOUTS :], pad_id, policy_device
#                 )
#             policy_lps_all = policy_lps_rollout + [lp.detach() for lp in policy_lps_gt]

#             ## 3.3 Compute KL divergence and add to rewards with coefficient BETA
#             kl = torch.stack([
#                 p.detach().to(policy_device) - r.to(policy_device)
#                 for p, r in zip(policy_lps_all, ref_lps)
#             ])  # [B * group_size]
#             rewards_adj = torch.tensor(ext_rewards, dtype=torch.float32, device=policy_device) - BETA * kl

#             ## 3.4 Optionally clip KL penalty to stabilize training and compute final mean KLD for the loss calculation
#             mean_kld = kl.clamp(min=0).mean().item()

#             # 4. Find grouped advantage and normalize it across the batch for stable training
#             rewards_grouped = rewards_adj.view(B, group_size)
#             adv = (rewards_grouped - rewards_grouped.mean(dim=1, keepdim=True)) / (
#                 rewards_grouped.std(dim=1, keepdim=True) + 1e-8
#             )
#             adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # global normalization across batch

#             # 5. Compute GRPO loss and backpropagate
#             # IS ratio: recomputed policy lps (with grad) vs generation-time lps (no grad)
#             policy_lps_t = torch.stack(policy_lps_rollout)                    # [B*N_ROLLOUTS], has grad
#             gen_lps_t    = torch.stack(gen_lps).to(policy_device).detach()    # [B*N_ROLLOUTS], no grad
#             gen_lps_t    = torch.nan_to_num(gen_lps_t, nan=0.0, posinf=0.0, neginf=-100.0)
#             # clamp log-ratio before exp() so it can never overflow to inf (which would
#             # make ratio*adv = -inf, loss = +inf, and backward produce NaN gradients)
#             log_ratio     = (policy_lps_t - gen_lps_t).clamp(-10.0, 10.0)
#             ratio         = torch.exp(log_ratio)
#             clipped_ratio = ratio.clamp(IS_CLIP_RANGE[0], IS_CLIP_RANGE[1])
#             adv_rollout   = adv[:, :N_ROLLOUTS].contiguous().view(-1).detach()  # [B*N_ROLLOUTS]
#             loss = -(torch.min(ratio * adv_rollout, clipped_ratio * adv_rollout)).mean()
#             if not torch.isfinite(loss):
#                 tqdm.write(f"[WARN] step={global_step} non-finite loss={loss.item()}, skipping backward")
#                 optimizer.zero_grad()
#             else:
#                 (loss / GRAD_ACCUM).backward()

#             # 6. Optimizer step every GRAD_ACCUM steps
#             global_step += 1
#             if global_step % GRAD_ACCUM == 0:
#                 torch.nn.utils.clip_grad_norm_(policy_model.parameters(), MAX_GRAD_NORM)
#                 optimizer.step()
#                 optimizer.zero_grad()

#             # 7. Logging and checkpointing
#             gen_time = time.time() - t0
#             mean_reward = torch.tensor(rewards, dtype=torch.float32).mean().item()
#             if global_step % LOG_EVERY == 0:
#                 print(
#                     f"[TRAIN] step={global_step} epoch={epoch+1} | "
#                     f"loss={loss.item():.4f} reward={mean_reward:.3f} "
#                     f"kl={mean_kld:.4f} time={gen_time:.1f}s"
#                 )
#                 progress_bar.set_postfix(step=global_step, loss=f"{loss.item():.4f}", reward=f"{mean_reward:.3f}")

#             if global_step % SAVE_EVERY == 0:
#                 ckpt = CHECKPOINT_DIR / f"step_{global_step}"
#                 policy_model.save_pretrained(ckpt)
#                 policy_tok.save_pretrained(ckpt)
#                 tqdm.write(f"[CKPT] Saved → {ckpt}")

#             # 8. plot reward, loss, and KL curves for monitoring training progress
#             plot_steps.append(global_step)
#             plot_loss.append(loss.item())
#             plot_reward.append(mean_reward)
#             plot_kl.append(mean_kld)

#             if global_step % LOG_EVERY == 0:
#                 fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#                 axes[0].plot(plot_steps, plot_loss);   axes[0].set_title("Loss");        axes[0].set_xlabel("Step")
#                 axes[1].plot(plot_steps, plot_reward); axes[1].set_title("Reward Mean"); axes[1].set_xlabel("Step")
#                 axes[2].plot(plot_steps, plot_kl);     axes[2].set_title("KL Divergence"); axes[2].set_xlabel("Step")
#                 fig.tight_layout()
#                 fig.savefig(PLOT_DIR / "training_curves.png", dpi=100)
#                 plt.close(fig)

#             # 9. clean up GPU memory if needed
#             del completions, prompt_ids, comp_ids, gen_lps
#             del ext_prompt_ids, ext_comp_ids, gt_comp_ids, gt_prompt_ids
#             del ref_lps, policy_lps_rollout, policy_lps_gt, policy_lps_all
#             del kl, rewards_adj, adv, adv_rollout, ratio, clipped_ratio, loss
#             torch.cuda.empty_cache()

#             progress_bar.update(1)

#         progress_bar.close()
#         global_batch_idx += len(batches)

#         ckpt = CHECKPOINT_DIR / f"epoch_{epoch + 1}"
#         policy_model.save_pretrained(ckpt)
#         policy_tok.save_pretrained(ckpt)
#         tqdm.write(f"[CKPT] Epoch {epoch+1} complete → {ckpt}")

#         # save fig of plotted curves at end of each epoch
#         fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#         axes[0].plot(plot_steps, plot_loss);   axes[0].set_title("Loss");          axes[0].set_xlabel("Step")
#         axes[1].plot(plot_steps, plot_reward); axes[1].set_title("Reward Mean");   axes[1].set_xlabel("Step")
#         axes[2].plot(plot_steps, plot_kl);     axes[2].set_title("KL Divergence"); axes[2].set_xlabel("Step")
#         fig.tight_layout()
#         fig.savefig(PLOT_DIR / f"training_curves_epoch_{epoch + 1}.png", dpi=100)
#         plt.close(fig)


#     final = CHECKPOINT_DIR / "final"
#     policy_model.save_pretrained(final)
#     policy_tok.save_pretrained(final)
#     print(f"Training complete → {final}")

# def init_judge(model_name):
#     """
#     Spawn judge model in a separate process to avoid vllm gpu mapping issue.
#     Returns judge input queue, output_queue, and the process object.
#     """
#     import os
#     os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # clear before spawning
#     judge_input_queue  = mp.Queue(maxsize=1) # only one batch at a time to avoid OOM
#     judge_output_queue = mp.Queue(maxsize=1) # only one batch at a time to avoid OOM
#     ready_queue        = mp.Queue(maxsize=1)
#     judge_process = mp.Process(
#         target=_judge_process_fn,
#         args=(model_name, judge_input_queue, judge_output_queue, ready_queue),
#         daemon=True
#     )
#     judge_process.start()
#     ready_queue.get()  # wait for judge process to signal it's ready (model loaded and waiting for input)
#     return judge_input_queue, judge_output_queue, judge_process

# def _judge_process_fn(model_name, input_queue, output_queue, ready_queue):
#     import os
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # pin judge to GPU 2

#     from vllm import LLM, SamplingParams
#     import torch

#     engine = LLM(
#         model=model_name,
#         dtype='bfloat16',
#         tensor_parallel_size=1,  # judge doesn't need tensor parallelism
#         trust_remote_code=True,
#         enforce_eager=True,
#         disable_custom_all_reduce=True,
#         gpu_memory_utilization=0.3,
#         device="cuda:0",  # pin judge to GPU 0 (seen as cuda:0 in this subprocess)
#     )

#     tokenizer = engine.get_tokenizer()

#     ready_queue.put('ready')  # signal to main process that judge is ready for input
#     sampling = SamplingParams(max_tokens=1, temperature=0.0)

#     while True:
#         items = input_queue.get()
#         # print(items)
#         if items is None:  # shutdown signal
#             output_queue.put(None)
#             return
#         prompts = []
#         for item in items:
#             system = item[0]
#             user_message = item[1]
#             messages = [
#                 {"role": "system", "content": system},
#                 {"role": "user", "content": user_message},
#             ]
#             if hasattr(tokenizer, "apply_chat_template"):
#                 prompt = tokenizer.apply_chat_template(
#                     messages, tokenize=False, add_generation_prompt=True
#                 )
#             else:
#                 prompt = f"System: {system}\nUser: {user_message}\nAssistant:"
#             prompts.append(prompt)
#         outputs = engine.generate(prompts, sampling_params=sampling)
#         scores = []
#         for out in outputs:
#             text = out.outputs[0].text.strip()
#             try:
#                 scores.append(int(text[0]))
#             except (ValueError, IndexError):
#                 scores.append(0)
#         output_queue.put(scores)  # put scores not raw outputs

#         torch.cuda.empty_cache()


# def train_init():
#     """
#     Main training loop.
#     """
#     # load models
#     print("Loading Judge")
#     mp.set_start_method("spawn", force=True)
#     input_queue, output_queue, process = init_judge(JUDGE_MODEL)

#     # Check if CUDA is available and print the number of GPUs detected. This should be done before any CUDA initialization.
#     print(f"Detected {torch.cuda.device_count()} CUDA GPU(s)")

#     # Check if model has been partially trained. If so, resume from the latest checkpoint. 
#     # Otherwise, start fresh from the base model.
#     CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
#     resume_ckpt, start_step = _find_latest_checkpoint(CHECKPOINT_DIR)
#     if resume_ckpt:
#         print(f"Resuming from {resume_ckpt} (step {start_step})")
#     else:
#         print(f"Starting fresh from {POLICY_MODEL}")

#     # Load Training Data
#     with open(TRAIN_DATA) as f:
#         all_rows = [json.loads(l) for l in f]
#     print(f"Loaded {len(all_rows)} training rows")

#     # Tokenize data bvefore hand.

#     # check if tokenized data already exists from a previous run to speed up development iterations
#     tokenized_path = CHECKPOINT_DIR / "tokenized_data.pt"
#     if tokenized_path.exists():
#         print(f"Loading tokenized data from {tokenized_path}")
#         all_rows = torch.load(tokenized_path)
#     else:
#         _tok = AutoTokenizer.from_pretrained(POLICY_MODEL, trust_remote_code=True)
#         _tok.pad_token = _tok.eos_token if _tok.pad_token is None else _tok.pad_token
#         _tok.padding_side = "left"
#         print(f"Pre-tokenizing {len(all_rows)} rows...")
#         for row in tqdm(all_rows):
#             row["formatted_prompt"] = format_prompt(_tok, row["prompt"])
#             enc = _tok(
#                 row["formatted_prompt"],
#                 truncation=True,
#                 max_length=MAX_PROMPT_LEN,
#                 padding=False,
#             )
#             row["input_ids"] = torch.tensor(enc.input_ids, dtype=torch.long)
#         del _tok
#         # save tokenized data for faster reloads during development
#         tokenized_path = CHECKPOINT_DIR / "tokenized_data.pt"
#         torch.save(all_rows, tokenized_path)
#         print(f"Saved tokenized data to {tokenized_path}")
#     print("tokenization complete")

#     # clean cuda cache before loading large models to avoid OOM
#     torch.cuda.empty_cache()
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
#     bnb_config = BitsAndBytesConfig(load_in_8bit=True)
#     print(f"loading base model {POLICY_MODEL}")
#     base_model = AutoModelForCausalLM.from_pretrained(
#         POLICY_MODEL,
#         trust_remote_code=True,
#         device_map=get_device_map(POLICY_MODEL, 0.7),
#         quantization_config=bnb_config,
#         torch_dtype=torch.bfloat16,  # base weights in bf16
#         attn_implementation="sdpa",  # faster attention for A100 (vLLM uses this too)
#     )
#     # freeze this model because we train only lora
#     for param in base_model.parameters():
#         param.requires_grad = False
        
#     print("Loading LoRA adapters")
#     lora_config = LoraConfig(
#         r=R,
#         lora_alpha=16,
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#         # target_modules=["gate_proj", "up_proj", "down_proj"],  # FFN
#         # target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],  # both
#         lora_dropout=0.05,
#         bias="none",
#     )

#     print("Applying LoRA adapters to base model")
#     policy_model = get_peft_model(base_model, lora_config)
#     _fix_rotary_device(policy_model)

#     print("Loading reference model")
#     reference_model = AutoModelForCausalLM.from_pretrained(
#         POLICY_MODEL,
#         trust_remote_code=True,
#         device_map={"": "cuda:3"},  # pin reference model to GPU 3
#         quantization_config=bnb_config,
#         torch_dtype=torch.bfloat16,
#         attn_implementation="sdpa",
#     )

#     print("Models loaded successfully")
#     policy_updates(
#         policy_model=policy_model,
#         reference_model=reference_model,
#         judge_input_queue=input_queue,
#         judge_output_queue=output_queue,
#         judge_process=process,
#         all_rows=all_rows,
#         start_step=start_step,
#     )

#     # end of training
#     input_queue.put(None)
#     process.join()  # now it's safe, process is exiting

# if __name__ == "__main__":
#     train_init()