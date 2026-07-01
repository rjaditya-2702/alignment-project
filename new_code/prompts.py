# CauSciBench prompts, borrowed verbatim from old_code (SFT and RL differ only in the
# tail after the shared header). data_prep bakes the per-phase user prompt into each row;
# CAUSCI_SYSTEM is prepended at message-build time (rl_data / train_sft).

CAUSCI_SYSTEM = (
    "You are a causal inference expert. Analyze the study design carefully before "
    "selecting variables and methods. Think through your reasoning, then output only the JSON."
)

# ── Shared header (identical in old_code SFT and RL prompts) ─────────────
_HEADER = """
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

"""

# ── SFT tail (old_code/src/output_fine_tune_lora) ───────────────────────
_SFT_TAIL = """## Method Reference

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

# ── RL tail (old_code/src/training/verl_/data_process.py) ────────────────
_RL_TAIL = """## Method Reference

| Method | Estimand | Use when |
|--------|----------|----------|
| diff_in_means | ATE | RCT with enforced compliance. Groups comparable by design. No confounding. |
| ols | ATE | Observational. All confounders observed and included. No unobserved confounding. |
| ipw | ATE, ATT, or ATC — based on whether question asks about population, treated group, or control group | Observational. Confounders observed. Reweight by propensity score. Needs overlap: 0 < e(X) < 1. |
| matching | ATE, ATT, or ATC  | Observational. Confounders observed. Use when propensity score overlap is poor. |
| did | ATT | Panel data. Treatment introduced at one point in time to one group. Time variable must be treatment timing, not a covariate. Parallel trends must hold. |
| rdd | Local ATE at the cutoff | Treatment assigned by a running variable crossing a known cutoff. Units just above and below cutoff are comparable. |
| iv | LATE | Unobserved confounders exist. Valid instrument available — correlated with treatment, affects outcome only through treatment. |
| frontdoor | ATE | Unobserved confounders exist. Full mediator pathway T→M→Y with no unobserved T→M or M→Y confounding. |
| glm | Conditional effect (log-odds for binary, incidence rate ratio for counts) | Binary outcome (logistic) or count outcome (Poisson). Confounders observed. |

---

Think through the following before answering:
- Was treatment randomly assigned or self-selected?
- Are confounders observed or unobserved?
- Is there a time variable marking treatment timing (not just a covariate)?
- Is there a continuous running variable with a cutoff?
- Is there a variable that affects treatment but not outcome directly?
- Is the outcome binary, count, or continuous?
- Does the question ask about the full population (ATE), treated units (ATT), or local effect (LATE)?

Then output this JSON and nothing else after your thinking. Use `null` only when the field is truly not applicable.

{{
  "step1": {{
    "treatment": "<exact column name>",
    "outcome": "<exact column name>",
    "controls": ["<list of exact column names>"],
    "instrument": "<exact column name if step2 is 'iv', else null>",
    "running_variable": "<exact column name if step2 is 'rdd', else null>",
    "cutoff": "<numeric threshold value if step2 is 'rdd', else null>",
    "time_variable": "<exact column name if step2 is 'did', else null>",
    "group_variable": "<exact column name if step2 is 'did', else null>",
    "mediator": "<exact column name if step2 is 'frontdoor', else null>",
    "estimand": "<ATE, ATT, ATC, LATE, or conditional>"
  }},
  "step2": "<method name - one of: diff_in_means, ols, ipw, matching, did, rdd, iv, frontdoor, glm >"
}}
"""

CAUSCI_USER_SFT = _HEADER + _SFT_TAIL
CAUSCI_USER_RL  = _HEADER + _RL_TAIL
