"""Build CauSciBench train/test jsonl from data/metadata_json + data/csv_files.

  train = qrdata + synthetic, hash-split disjoint by id:
            SFT_FRACTION  -> train_sft.jsonl
            the remainder -> train_rl.jsonl
  test  = realdata (full; shared by SFT and RL eval)

Each row: {id, source, split, prompt, label, groundtruth, csv_path}.
Run:  python data_prep.py
"""

import hashlib
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

import config
from prompts import CAUSCI_USER_SFT, CAUSCI_USER_RL


def _assign(row_id) -> str:
    """Stable per-id split so SFT and RL pick complementary rows even run separately."""
    h = int(hashlib.md5(str(row_id).encode()).hexdigest(), 16) % 100
    return "sft" if h < config.SFT_FRACTION * 100 else "rl"


def _none(v):
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, str) and v.strip().lower() in ("", "nan", "none"):
        return None
    return v


def _controls(v):
    v = _none(v)
    if not v:
        return []
    return [c.strip() for c in str(v).split(",") if c.strip()]


# ── Dataset metadata for the prompt ─────────────────────────────────────

def _describe(df, step1):
    numeric = df.select_dtypes(include="number").columns.tolist()
    if len(numeric) <= 25:
        return df.describe(include="all").to_string()
    key = [step1.get(f) for f in ("treatment", "outcome", "instrument",
                                  "running_variable", "time_variable")]
    key = [c for c in key if c and c in df.columns]
    key += [c for c in step1.get("controls", []) if c in df.columns and c not in key]
    rest = [c for c in numeric if c not in key]
    cols = [c for c in key + rest[: max(0, 25 - len(key))] if c in df.columns]
    return df[cols].describe(include="all").to_string()


def _metadata(df, step1):
    shape = f"{df.shape[0]} rows, {df.shape[1]} columns"
    cols  = "\n".join(f"  {c}: {df[c].dtype}" for c in df.columns)
    head  = df.head(5).to_string()
    desc  = _describe(df, step1)
    miss  = "\n".join(f"  {c}: {n}" for c, n in df.isnull().sum().items() if n > 0) or "  None"
    low = []
    for c in df.columns:
        if df[c].nunique() <= 10:
            vals = sorted(v for v in df[c].dropna().unique().tolist())
            vals = [int(v) if isinstance(v, np.integer) else
                    float(v) if isinstance(v, np.floating) else v for v in vals]
            low.append(f"  {c}: {vals}")
    return shape, cols, head, desc, miss, ("\n".join(low) or "  None")


# ── Row builder ─────────────────────────────────────────────────────────

def _read_csv(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def _build_base(entry, source, csv_path, idx):
    """Read csv + metadata once. Phase prompts are baked later from `fmt`."""
    df = _read_csv(csv_path)
    step1 = {
        "treatment":        _none(entry.get("treatment_var")),
        "outcome":          _none(entry.get("outcome_var")),
        "controls":         _controls(entry.get("control_variables")),
        "instrument":       _none(entry.get("instrument_var")),
        "running_variable": _none(entry.get("running_var")),
        "time_variable":    _none(entry.get("temporal_var")),
        "group_variable":   _none(entry.get("state_var")),
    }
    shape, cols, head, desc, miss, low = _metadata(df, step1)
    effect = float(entry["effect"])
    return {
        "id":     str(entry.get("id") or f"{source}_{idx}"),
        "source": source,
        "fmt": {
            "dataset_description": entry["dataset_description"],
            "file_path":           str(csv_path.relative_to(config.PROJECT)),
            "shape": shape, "columns_and_types": cols, "df_head": head,
            "df_describe": desc, "missing_values": miss, "low_cardinality": low,
            "query": entry["query"],
        },
        "label":       effect,
        "groundtruth": {"step1": step1, "step2": entry["method"],
                        "step3": None, "step4": None, "step5": effect},
        "csv_path":    str(csv_path),
    }


def _row(base, template, split):
    return {
        "id": base["id"], "source": base["source"], "split": split,
        "prompt": template.format(**base["fmt"]),
        "label": base["label"], "groundtruth": base["groundtruth"],
        "csv_path": base["csv_path"],
    }


def _load_split(key):
    json_name, subdir = config.SPLITS[key]
    entries = json.load(open(config.META_DIR / json_name))
    bases = []
    for i, e in enumerate(entries):
        csv_path = config.CSV_DIR / subdir / os.path.basename(e["dataset_path"])
        if not csv_path.exists():
            print(f"  WARNING: missing csv for {key}[{i}]: {csv_path}")
            continue
        bases.append(_build_base(e, key, csv_path, i))
    return bases


def _write(bases, template, split, path):
    rows = [_row(b, template, split) for b in bases]
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"Wrote {len(rows):>4} rows → {path}")


def main():
    config.OUT.mkdir(parents=True, exist_ok=True)

    train = [b for k in config.TRAIN_SPLITS for b in _load_split(k)]
    sft  = [b for b in train if _assign(b["id"]) == "sft"]
    rl   = [b for b in train if _assign(b["id"]) == "rl"]
    test = _load_split(config.TEST_SPLIT)

    _write(sft,  CAUSCI_USER_SFT, "train", config.TRAIN_SFT_JSONL)
    _write(test, CAUSCI_USER_SFT, "test",  config.TEST_SFT_JSONL)
    _write(rl,   CAUSCI_USER_RL,  "train", config.TRAIN_RL_JSONL)
    _write(test, CAUSCI_USER_RL,  "test",  config.TEST_RL_JSONL)

    print(f"\ntrain {len(train)} (sft {len(sft)} / rl {len(rl)})  test {len(test)}")
    for name, rows in [("sft", sft), ("rl", rl), ("test", test)]:
        m = Counter(b["groundtruth"]["step2"] for b in rows)
        print(f"  {name:4} methods: {dict(m)}")


if __name__ == "__main__":
    main()
