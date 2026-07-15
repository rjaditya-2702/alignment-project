import json
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3] # verl_ -> training -> src -> project root
sys.path.insert(0, str(PROJECT_ROOT)) # allow imports from project root

from src.data.preprocess import preprocess
from src.config import (
    OUTPUT_DIR_RL,
    TRAIN_RL_CLADDER, TRAIN_RL_CAUSCI,
    TEST_RL_CLADDER, TEST_RL_CAUSCI,
)

# ---------------------------------------------------------------------------
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

Step 5 - Answer: Based on the above inference performed and the question asked in the scenario, answer yes or no.

Then output this JSON and nothing else:

{{
  "step1": "<variable assignments and all directed edges>",
  "step2": "<query type>",
  "step3": "<estimand expression>",
  "step4": "<full arithmetic or graph reasoning, final value at the end>",
  "step5": "<yes or no>"
}}
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
| backdoor  | ATE | Observational. All confounders observed. Use regression (ols/glm) or other adjustment to block backdoor paths. |
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
  "step2": "<method name - one of: diff_in_means, ols, propensity_score, ipw, matching, did, rdd, iv, backdoor, frontdoor, glm >"
}}
"""


# Add these imports at top

CAUSCI_SYSTEM_PROMPT = """You are a causal inference expert. Analyze the study design carefully before selecting variables and methods. Think through your reasoning, then output only the JSON.
"""

CLADDER_SYSTEM_PROMPT = """You are a causal inference expert. Analyze the study design carefully before selecting variables and methods. Think through your reasoning, then output only the JSON.
"""

def _build_messages(row: dict) -> list[dict]:
    system = CLADDER_SYSTEM_PROMPT if row["source"] == "cladder" else CAUSCI_SYSTEM_PROMPT
    return [
        {"role": "system",    "content": system},
        {"role": "user",      "content": row["prompt"]},
        # {"role": "assistant", "content": "<think>"},
    ]

def _resolve_csv_path(stored: str) -> str:
    """Re-anchor stored csv_path to current PROJECT_ROOT via known anchor segments."""
    p = Path(stored)
    for anchor in ("dataset", "original_data"):
        for i, part in enumerate(p.parts):
            if part == anchor:
                return str(PROJECT_ROOT / Path(*p.parts[i:]))
    raise ValueError(f"Cannot resolve csv_path — no anchor found: {stored}")

def _build_extra_info(row: dict, split: str) -> dict:
    if row["source"] == "cladder":
        return {"csv_path": "", "dataset_columns": [], "split": split}

    stored = row.get("csv_path")
    if not stored:
        p = row["prompt"]
        path_start = p.find("Path: ") + len("Path: ")
        stored = p[path_start : p.find("\n", path_start)].strip()

    csv_path = _resolve_csv_path(stored)
    dataset_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    return {"csv_path": csv_path, "dataset_columns": dataset_columns, "split": split}

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


def _convert_split(jsonl_paths: list[Path], parquet_path: Path, split: str, limit: int | None = None) -> None:
    # Merge the per-source files (cladder + causci) into one parquet. Drop a path
    # from the list to keep a single source.
    rows = []
    for jp in jsonl_paths:
        if not jp.exists():
            print(f"  skip (missing): {jp}")
            continue
        with open(jp) as f:
            rows.extend(json.loads(line) for line in f)
    if not rows:
        raise FileNotFoundError(f"No rows from {jsonl_paths}. Run preprocess() first.")

    if limit is not None:
        rows = _limit_cladder(rows, limit)

    records = []
    skipped = 0

    for row in tqdm(rows):
        try:
            records.append({
                "prompt":       _build_messages(row),
                "data_source":  row["source"],
                "reward_model": {"ground_truth": json.dumps(row["groundtruth"])},
                "extra_info":   _build_extra_info(row, split),
            })
        except Exception as e:
            raise FileNotFoundError(f"Error processing row with id {row.get('id', '?')}: {e}")

    df = pd.DataFrame(records)
    df.to_parquet(parquet_path, index=False)
    print(f"  {len(records)} rows written → {parquet_path}  (skipped: {skipped})")

def main():
    # preprocess raw data into train/test jsonl files for RL training
    output_dir = Path(OUTPUT_DIR_RL)
    preprocess(cladder_prompt=CLADDER_USER_PROMPT, causci_prompt=CAUSCI_USER_PROMPT, output_dir=output_dir, which="rl")
    print("Preprocessing complete.")

    # One parquet per source — no merging here. Merge at training launch by
    # listing the parquets you want in veRL's data.train_files / data.val_files.
    print("Preparing veRL parquet files...")
    _convert_split([TRAIN_RL_CLADDER], output_dir / "train_rl_cladder.parquet", split="train", limit=CLADDER_TRAIN_LIMIT)
    _convert_split([TRAIN_RL_CAUSCI],  output_dir / "train_rl_causci.parquet",  split="train")
    _convert_split([TEST_RL_CLADDER],  output_dir / "test_cladder.parquet",     split="test")
    _convert_split([TEST_RL_CAUSCI],   output_dir / "test_causci.parquet",      split="test")
    print("Done.")


if __name__ == "__main__":
    main()