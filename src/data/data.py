"""
Data loading and unification for GRPO post-training.
Datasets: CLadder (HuggingFace) + CauSciBench (local JSON files)
"""

import json
import math
from pathlib import Path

import pandas as pd
from datasets import load_dataset

CAUSCIBENCH_DIR = Path("original_data/CauSciBench")
CAUSCIBENCH_JSON_DIR = CAUSCIBENCH_DIR / "data" / "metadata_json"


# ── Prompt Templates ─────────────────────────────────────────────────

CLADDER_PROMPT = """\
Solve the following causal inference problem step by step.

## Problem
{problem}

---

## Step 1: Causal Structure
Assign algebraic variables to each entity. Identify all variables and causal edges from the scenario.
Format: V1 -> V2, V2 -> V3

## Step 2: Query Classification
What type of causal query is this? Return exactly one of the following:
- marginal: the overall probability of a variable, P(Y)
- correlation: whether observing one variable changes the probability of another, P(Y|X)
- ate: the causal effect of intervening on a variable, E[Y|do(X=1)] - E[Y|do(X=0)]
- backadj: whether a set of variables is a valid adjustment set for estimating a causal effect
- det-counterfactual: what a variable would have been under a different condition, P(Y_x=y)
- ett: for those who received treatment, what would have happened without it, E[Y1-Y0|X=1]
- nde: the direct effect of a variable not through any mediator, E[Y_{{1,M0}} - Y_{{0,M0}}]
- nie: the indirect effect of a variable only through mediators, E[Y_{{0,M1}} - Y_{{0,M0}}]
- collider_bias: whether intervening on a cause of a common effect induces a spurious association with another cause
- exp_away: whether conditioning on a common effect changes the association between its causes

## Step 3: Derive Estimand
Using the causal graph and query type, derive the mathematical expression needed. Show your do-calculus or counterfactual reasoning.

## Step 4: Compute
Derive an expression equivalent to Step 3 that uses only terms available in the problem data. Write the derivation, then provide executable Python code that computes the result.

Return the code inside a ```python block. The code must:
- Define each probability value as a variable
- Compute the final estimand
- Print exactly one line: result=<number> for numerical queries, or result=yes / result=no for qualitative queries (backadj, collider_bias)

## Step 5: Answer
Given the result and the question, respond: Yes or No
Answer:"""


CAUSCIBENCH_PROMPT = """\
Solve the following causal inference problem step by step.

## Study Description
{dataset_description}

## Dataset
Path: {file_path}
Columns and types:
{columns_and_types}
First 5 rows:
{df_head}
Summary statistics:
{df_describe}
Missing values:
{missing_values}

## Question
{query}

---

## Step 1: Causal Structure
Identify the key causal variables from the dataset.
Return in this exact format:
- treatment: <column_name>
- outcome: <column_name>
- controls: [<col1>, <col2>, ...]
- instrument: <column_name> or none
- running_variable: <column_name> or none
- time_variable: <column_name> or none
- group_variable: <column_name> or none

## Step 2: Method Selection
What causal inference method is appropriate for this study design? Return exactly one of:
- diff_in_means, ols, ipw, matching, did, rdd, iv, frontdoor, glm

Select one. Justify in one sentence based on the study design.

## Step 3: Estimation Specification
Write the formal estimation setup. Specify the regression formula or estimation procedure, the estimand (ATE, ATT, LATE, etc.), and the key identification assumption.

## Step 4: Implement
Write executable Python code that loads the data, implements the method from Step 3, and computes the causal effect estimate.

Return the code inside a ```python block. The code must:
- Load the dataset from the given path
- Preprocess if needed
- Implement the chosen method using only: pandas, numpy, statsmodels, linearmodels, dowhy, rdd, sklearn, scipy
- Print exactly one line: result=<number>

## Step 5: Answer
Report the causal effect estimate.
Answer:"""


# ── CLadder ──────────────────────────────────────────────────────────

def parse_cladder_step1(reasoning: str) -> str:
    """First two lines of reasoning: variable assignments + edges."""
    lines = [l.strip() for l in reasoning.strip().split("\n") if l.strip()]
    return "\n".join(lines[:2])


def load_cladder() -> list[dict]:
    # local JSON has only 100 rows (truncated preview); use HuggingFace
    ds = load_dataset("causal-nlp/CLadder", split="full_v1.5_default")
    rows = []
    for row in ds:
        rows.append({
            "id": f"cladder_{row['id']}",
            "source": "cladder",
            "prompt": CLADDER_PROMPT.format(problem=row["prompt"]),
            "label": row["label"],
            "label_type": "binary",
            "groundtruth": {
                "step1": parse_cladder_step1(row["reasoning"]),
                "step2": row["query_type"],
                "step3": row["formal_form"],
                "step4": row["reasoning"],
                "step5": row["label"],
            },
        })
    return rows


# ── CauSciBench ──────────────────────────────────────────────────────

def _nan_to_none(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def build_causcibench_prompt(entry: dict) -> str:
    csv_path = CAUSCIBENCH_DIR / entry["dataset_path"]
    df = pd.read_csv(csv_path)

    columns_and_types = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
    df_head = df.head().to_string(index=False)
    df_describe = df.describe().to_string()
    missing = df.isnull().sum()
    missing_str = "\n".join(f"  {col}: {cnt}" for col, cnt in missing.items() if cnt > 0) or "  none"

    return CAUSCIBENCH_PROMPT.format(
        dataset_description=entry["dataset_description"],
        file_path=str(csv_path.resolve()),
        columns_and_types=columns_and_types,
        df_head=df_head,
        df_describe=df_describe,
        missing_values=missing_str,
        query=entry["query"],
    )


def load_causcibench() -> list[dict]:
    sources = [
        ("real",      "real_data.json"),
        ("synthetic", "synthetic_data.json"),
        ("qrdata",    "qrdata.json"),
    ]
    rows = []
    for source_key, json_file in sources:
        with open(CAUSCIBENCH_JSON_DIR / json_file) as f:
            entries = json.load(f)

        for i, entry in enumerate(entries):
            if not (CAUSCIBENCH_DIR / entry["dataset_path"]).exists():
                continue
            prompt = build_causcibench_prompt(entry)
            step1 = {
                "treatment":        _nan_to_none(entry.get("treatment_var")),
                "outcome":          _nan_to_none(entry.get("outcome_var")),
                "controls":         _nan_to_none(entry.get("control_variables")),
                "instrument":       _nan_to_none(entry.get("instrument_var")),
                "running_variable": _nan_to_none(entry.get("running_var")),
                "time_variable":    _nan_to_none(entry.get("temporal_var")),
                "group_variable":   _nan_to_none(entry.get("state_var")),
            }
            rows.append({
                "id":         f"causci_{source_key}_{i}",
                "source":     "causcibench",
                "prompt":     prompt,
                "label":      entry["effect"],
                "label_type": "continuous",
                "groundtruth": {
                    "step1": step1,
                    "step2": entry["method"],
                    "step3": None,
                    "step4": None,
                    "step5": entry["effect"],
                },
            })
    return rows


# ── Combine ───────────────────────────────────────────────────────────

def load_unified_dataset() -> list[dict]:
    print("Loading CLadder...")
    cladder = load_cladder()
    print(f"  {len(cladder)} examples")

    print("Loading CauSciBench...")
    causci = load_causcibench()
    print(f"  {len(causci)} examples")

    combined = cladder + causci
    print(f"  Total: {len(combined)} examples")
    return combined


if __name__ == "__main__":
    data = load_unified_dataset()

    cl = data[0]
    print("\n--- CLadder sample ---")
    print("id:", cl["id"])
    print("label:", cl["label"])
    print("gt step2:", cl["groundtruth"]["step2"])
    print("gt step3:", cl["groundtruth"]["step3"])
    print("prompt[:200]:", cl["prompt"][:200])

    cs = next(d for d in data if d["source"] == "causcibench")
    print("\n--- CauSciBench sample ---")
    print("id:", cs["id"])
    print("label:", cs["label"])
    print("gt step1:", cs["groundtruth"]["step1"])
    print("gt step2:", cs["groundtruth"]["step2"])
    print("prompt[:200]:", cs["prompt"][:200])
