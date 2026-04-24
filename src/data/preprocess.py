CLADDER_PROMPT = """You are given a scenario describing relationships between variables, along with numerical data and a question. Your task is to determine the answer by following these steps precisely.
---

Use the following reference to guide your reasoning.

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

Now solve the problem.

## Step 1: Causal Structure
Assign algebraic variables (e.g., X, Y, Z) to each entity mentioned in the scenario. Identify all directed causal edges.
Format: V1 -> V2, V2 -> V3

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
Rewrite the expression from Step 3 using only the numerical values given in the Data section. Then provide executable Python code that computes the result.

Return the code inside a ```python block. The code must:
- Define each probability value as a variable
- Compute the final estimand step by step
- Print exactly one line: result=<number> for numerical queries, or result=yes / result=no for qualitative queries (backadj, collider_bias, exp_away)

## Step 5: Answer
Based on the computed result and what the question is asking, answer either a "yes" or a "no".
- For ate/ett/nde/nie: if the result is positive and the question asks "does X increase Y", answer Yes. If negative, answer No. Vice versa if the question asks "does X decrease Y".
- For marginal: if P(Y) > 0.5 and the question asks "is Y more likely than not", answer Yes.
- For correlation: if P(Y|X=1) > P(Y|X=0) and the question asks "does observing X increase Y", answer Yes.
- For backadj/collider_bias/exp_away: answer Yes or No based on the graph analysis.
- For det-counterfactual: answer based on the computed probability and what the question asks.

## Scenario
{verbalized_story}

Answer:
"""

"""
Preprocessing pipeline for causal alignment training data.

Input:  dataset/train.jsonl  (sources: cladder_synthetic, causcibench_synthetic)
        dataset/test.jsonl   (sources: cladder, causcibench)

For CLadder: rebuild prompt with CLADDER_PROMPT (new template uses {verbalized_story}).
For CauSciBench: reload CSV, rebuild prompt with CAUSCI_PROMPT (adds shape + low_cardinality).

Output: output/train.jsonl
        output/test.jsonl
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_INPUT = Path("dataset/train.jsonl")
TEST_INPUT  = Path("dataset/test.jsonl")
OUTPUT_DIR  = Path("output")

SOURCE_MAP = {
    "cladder_synthetic": "cladder",
    "causcibench_synthetic": "causcibench",
    "cladder":              "cladder",
    "causcibench":          "causcibench",
}

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

Now solve the problem.

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

## Step 4: Implement
Write executable Python code that loads the dataset, preprocesses it, implements the method from Step 3, and computes the effect.

Return the code inside a ```python block. The code must:
- Load the data: pd.read_csv("{file_path}")
- Handle missing values and encoding as needed
- Implement the chosen method using only: pandas, numpy, statsmodels, linearmodels, dowhy, rdd, sklearn, scipy
- Print exactly one line: result=<number>

## Step 5: Answer
Report the estimated effect.
Answer:
"""


# ── Helpers ───────────────────────────────────────────────────────────

def _is_null(v):
    if v is None:
        return True
    if isinstance(v, float) and v != v:
        return True
    if isinstance(v, str) and v.strip().lower() in ("nan", "none", ""):
        return True
    return False


def _normalize_cladder_label(label):
    if isinstance(label, bool):
        return "yes" if label else "no"
    if isinstance(label, (int, float)):
        return "yes" if int(label) == 1 else "no"
    s = str(label).strip().lower()
    if s in ("yes", "true", "1"):
        return "yes"
    if s in ("no", "false", "0"):
        return "no"
    raise ValueError(f"Unrecognized CLadder label: {label!r}")


def _extract_verbalized_story(prompt):
    """Extract the scenario blob from existing CLadder prompt (built with old {problem} template)."""
    marker = "## Problem\n"
    end_marker = "\n\n---"
    start = prompt.find(marker)
    if start == -1:
        # Older format: scenario is under ## Scenario
        marker = "## Scenario\n"
        start = prompt.find(marker)
    if start == -1:
        return prompt  # fallback: use full prompt
    start += len(marker)
    end = prompt.find(end_marker, start)
    return prompt[start:end].strip() if end != -1 else prompt[start:].strip()


def _parse_causci_prompt_fields(prompt):
    """Extract description, file_path, query from existing CauSciBench prompt."""
    desc_start = prompt.find("## Study Description\n") + len("## Study Description\n")
    desc_end   = prompt.find("\n\n## Dataset")
    description = prompt[desc_start:desc_end].strip()

    path_start = prompt.find("Path: ") + len("Path: ")
    path_end   = prompt.find("\n", path_start)
    file_path  = prompt[path_start:path_end].strip()

    query_start = prompt.find("## Question\n") + len("## Question\n")
    query_end   = prompt.find("\n\n---", query_start)
    query = prompt[query_start:query_end].strip()

    return description, file_path, query


def _build_df_describe(df, step1):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) <= 25:
        return df.describe(include="all").to_string()

    key_cols = []
    if isinstance(step1, dict):
        for field in ("treatment", "outcome", "instrument", "running_variable", "time_variable"):
            val = step1.get(field)
            if val and val in df.columns and val not in key_cols:
                key_cols.append(val)
        controls = step1.get("controls") or []
        if isinstance(controls, str):
            controls = [c.strip() for c in controls.split(",") if c.strip()]
        for c in controls:
            if c in df.columns and c not in key_cols:
                key_cols.append(c)

    remaining = [c for c in numeric_cols if c not in key_cols]
    selected  = key_cols + remaining[:max(0, 25 - len(key_cols))]
    selected  = [c for c in selected if c in df.columns]
    return df[selected].describe(include="all").to_string()


def _compute_df_metadata(df, step1):
    shape = f"{df.shape[0]} rows, {df.shape[1]} columns"

    columns_and_types = "\n".join(f"  {col}: {df[col].dtype}" for col in df.columns)

    df_head = df.head(5).to_string()

    df_describe = _build_df_describe(df, step1)

    missing = df.isnull().sum()
    missing_lines = [f"  {col}: {cnt}" for col, cnt in missing.items() if cnt > 0]
    missing_str = "\n".join(missing_lines) if missing_lines else "  None"

    low_card_lines = []
    for col in df.columns:
        if df[col].nunique() <= 10:
            vals = sorted([v for v in df[col].unique().tolist() if pd.notna(v)])
            vals = [
                int(v)   if isinstance(v, np.integer)  else
                float(v) if isinstance(v, np.floating) else
                str(v)   if not isinstance(v, (int, float, str)) else v
                for v in vals
            ]
            low_card_lines.append(f"  {col}: {vals}")
    low_cardinality_str = "\n".join(low_card_lines) if low_card_lines else "  None"

    return shape, columns_and_types, df_head, df_describe, missing_str, low_cardinality_str


# ── Row processors ────────────────────────────────────────────────────

def process_cladder_row(row, split):
    label          = _normalize_cladder_label(row["label"])
    verbalized     = _extract_verbalized_story(row["prompt"])
    prompt         = CLADDER_PROMPT.replace("{verbalized_story}", verbalized)
    gt             = row.get("groundtruth", {})

    step3 = None if _is_null(gt.get("step3")) else gt.get("step3")
    step4 = None if _is_null(gt.get("step4")) else gt.get("step4")
    has_nan = (step3 is None) or (step4 is None)

    out = {
        "id":         row["id"],
        "source":     SOURCE_MAP[row["source"]],
        "split":      split,
        "prompt":     prompt,
        "label":      label,
        "label_type": "binary",
        "groundtruth": {
            "step1": gt.get("step1"),
            "step2": gt.get("step2"),
            "step3": step3,
            "step4": step4,
            "step5": label,
        },
    }
    if has_nan:
        out["has_nan_reasoning"] = True
    return out


def process_causcibench_row(row, split, csv_failures):
    description, file_path, query = _parse_causci_prompt_fields(row["prompt"])

    csv_path = Path(file_path)
    if not csv_path.exists():
        print(f"  WARNING: CSV not found for {row['id']}: {file_path}")
        csv_failures.append(row["id"])
        return None

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"  WARNING: CSV load failed for {row['id']}: {e}")
        csv_failures.append(row["id"])
        return None

    step1 = row.get("groundtruth", {}).get("step1")
    shape, columns_and_types, df_head, df_describe, missing_str, low_cardinality_str = \
        _compute_df_metadata(df, step1)

    prompt = CAUSCI_PROMPT.format(
        dataset_description=description,
        file_path=file_path,
        shape=shape,
        columns_and_types=columns_and_types,
        df_head=df_head,
        df_describe=df_describe,
        missing_values=missing_str,
        low_cardinality=low_cardinality_str,
        query=query,
    )

    label = float(row["label"])
    gt    = row.get("groundtruth", {})

    return {
        "id":         row["id"],
        "source":     SOURCE_MAP[row["source"]],
        "split":      split,
        "prompt":     prompt,
        "label":      label,
        "label_type": "continuous",
        "groundtruth": {
            "step1": step1,
            "step2": gt.get("step2"),
            "step3": None,
            "step4": None,
            "step5": label,
        },
    }


# ── Validation ────────────────────────────────────────────────────────

METHODS = ["ols", "iv", "did", "rdd", "matching", "ipw", "glm", "frontdoor", "diff_in_means"]


def _validate(train_rows, test_rows, csv_failures):
    train_cl = [r for r in train_rows if r["source"] == "cladder"]
    train_cs = [r for r in train_rows if r["source"] == "causcibench"]
    test_cl  = [r for r in test_rows  if r["source"] == "cladder"]
    test_cs  = [r for r in test_rows  if r["source"] == "causcibench"]

    print("=== Row Counts ===")
    print(f"train: cladder={len(train_cl)}  causcibench={len(train_cs)}")
    print(f"test:  cladder={len(test_cl)}  causcibench={len(test_cs)}")

    print("\n=== CLadder Label Balance ===")
    for name, rows in [("train", train_cl), ("test", test_cl)]:
        yes = sum(1 for r in rows if r["label"] == "yes")
        no  = sum(1 for r in rows if r["label"] == "no")
        print(f"{name}: yes={yes}  no={no}")

    print("\n=== CLadder NaN Reasoning ===")
    train_nan = sum(1 for r in train_cl if r.get("has_nan_reasoning"))
    test_nan  = sum(1 for r in test_cl  if r.get("has_nan_reasoning"))
    print(f"train: {train_nan}")
    print(f"test:  {test_nan}")
    if train_nan > 0:
        print(f"  NOTE: {train_nan} train rows have empty step4 — kept as-is, step4 set to null")

    for name, rows in [("Train", train_cs), ("Test", test_cs)]:
        counts  = Counter(r["groundtruth"]["step2"] for r in rows)
        line    = "  ".join(f"{m}={counts.get(m, 0)}" for m in METHODS)
        missing = [m for m in METHODS if counts.get(m, 0) == 0]
        print(f"\n=== CauSciBench Methods in {name} ===")
        print(line)
        if missing:
            print(f"  WARN: missing methods: {missing}")

    print("\n=== Prompt Length (whitespace-split word count) ===")
    for src, rows in [("cladder", train_cl + test_cl), ("causcibench", train_cs + test_cs)]:
        if not rows:
            continue
        lengths = [len(r["prompt"].split()) for r in rows]
        print(f"{src}:   min={min(lengths)}  mean={int(sum(lengths)/len(lengths))}  max={max(lengths)}")
        over = [r["id"] for r in rows if len(r["prompt"].split()) > 5000]
        if over:
            print(f"  WARN: {len(over)} prompts exceed 5000 words: {over[:5]}{'...' if len(over) > 5 else ''}")

    print("\n=== CSV Load Failures ===")
    if csv_failures:
        for r in csv_failures:
            print(f"  {r}")
    else:
        print("  None")

    print("\n=== Groundtruth Completeness ===")
    for src, rows in [("cladder", train_cl + test_cl), ("causcibench", train_cs + test_cs)]:
        steps = ["step1", "step2", "step3", "step4", "step5"] if src == "cladder" \
                else ["step1", "step2", "step5"]
        for step in steps:
            null_count = sum(1 for r in rows if r["groundtruth"].get(step) is None)
            print(f"{src} {step} null: {null_count}")

    print("\n=== Sample Outputs ===")
    for label, rows in [("CLadder", train_cl + test_cl), ("CauSciBench", train_cs + test_cs)]:
        if rows:
            print(f"--- {label} sample ---")
            print(json.dumps(rows[0], indent=2, ensure_ascii=False, default=str))


# ── Main ──────────────────────────────────────────────────────────────

def preprocess():
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_rows, test_rows, csv_failures = [], [], []

    for jsonl_path, split, out_list in [
        (TRAIN_INPUT, "train", train_rows),
        (TEST_INPUT,  "test",  test_rows),
    ]:
        with open(jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                src = SOURCE_MAP[row["source"]]
                if src == "cladder":
                    out = process_cladder_row(row, split)
                else:
                    out = process_causcibench_row(row, split, csv_failures)
                if out is not None:
                    out_list.append(out)

    with open(OUTPUT_DIR / "train.jsonl", "w") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    with open(OUTPUT_DIR / "test.jsonl", "w") as f:
        for row in test_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    print(f"Wrote {len(train_rows)} train rows → {OUTPUT_DIR / 'train.jsonl'}")
    print(f"Wrote {len(test_rows)} test rows  → {OUTPUT_DIR / 'test.jsonl'}\n")

    _validate(train_rows, test_rows, csv_failures)


if __name__ == "__main__":
    preprocess()