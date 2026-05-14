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
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT        = Path(__file__).parent.parent.parent
TRAIN_INPUT = ROOT / "dataset" / "train.jsonl"
TEST_INPUT  = ROOT / "dataset" / "test.jsonl"

SOURCE_MAP = {
    "cladder_synthetic": "cladder",
    "causcibench_synthetic": "causcibench",
    "cladder":              "cladder",
    "causcibench":          "causcibench",
}


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

def process_cladder_row(row, split, cladder_prompt):
    label          = _normalize_cladder_label(row["label"])
    verbalized     = _extract_verbalized_story(row["prompt"])
    prompt         = cladder_prompt.replace("{verbalized_story}", verbalized)
    gt             = row.get("groundtruth", {})

    step3 = None if _is_null(gt.get("step3")) else gt.get("step3")
    step4 = None if _is_null(gt.get("step4")) else gt.get("step4")
    has_nan = step3 is None

    out = {
        "id":         row["id"],
        "source":     'cladder',
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


def process_causcibench_row(row, split, csv_failures, causci_prompt):
    description, file_path, query = _parse_causci_prompt_fields(row["prompt"])

    csv_path = Path(file_path)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    else:
        # Re-anchor absolute paths to ROOT in case they came from a different machine
        for i, part in enumerate(csv_path.parts):
            if part == "dataset":
                csv_path = ROOT / Path(*csv_path.parts[i:])
                break

    if not csv_path.exists():
        print(f"  WARNING: CSV not found for {row['id']}: {csv_path}")
        csv_failures.append(row["id"])
        raise

    df = pd.read_csv(csv_path, low_memory=False)

    step1 = row.get("groundtruth", {}).get("step1")
    shape, columns_and_types, df_head, df_describe, missing_str, low_cardinality_str = \
        _compute_df_metadata(df, step1)

    rel_path = str(csv_path.relative_to(ROOT))
    prompt = causci_prompt.format(
        dataset_description=description,
        file_path=rel_path,
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
        "source":     'causcibench',
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
        "csv_path": csv_path
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
        print(f"  NOTE: {train_nan} train rows have null step3 (estimand) — kept as-is")

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

def preprocess(cladder_prompt, causci_prompt, output_dir):

    # should not be null
    if cladder_prompt is None:
        raise ValueError("cladder_prompt is required")
    if causci_prompt is None:
        raise ValueError("causcibench_prompt is required")
    if output_dir is None:
        raise ValueError("output_dir is required")

    os.makedirs(output_dir, exist_ok=True)

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
                    out = process_cladder_row(row, split, cladder_prompt)
                else:
                    out = process_causcibench_row(row, split, csv_failures, causci_prompt)
                if out is not None:
                    out_list.append(out)

    with open(output_dir / "train.jsonl", "w") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    with open(output_dir / "test.jsonl", "w") as f:
        for row in test_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    print(f"Wrote {len(train_rows)} train rows → {output_dir / 'train.jsonl'}")
    print(f"Wrote {len(test_rows)} test rows  → {output_dir / 'test.jsonl'}\n")

    _validate(train_rows, test_rows, csv_failures)


if __name__ == "__main__":
    preprocess(cladder_prompt = None, causci_prompt = None, output_dir = None)