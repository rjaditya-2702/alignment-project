"""Convert RL train + test jsonl into veRL parquet files.

Run AFTER data_prep.py.  Reads train_rl.jsonl / test_rl.jsonl → train_rl.parquet / test.parquet.
veRL row schema: prompt(messages), data_source, reward_model.ground_truth, extra_info.
"""

import json

import pandas as pd

import config
from prompts import CAUSCI_SYSTEM


def _messages(row):
    return [{"role": "system", "content": CAUSCI_SYSTEM},
            {"role": "user",   "content": row["prompt"]}]


def _columns(csv_path):
    try:
        return pd.read_csv(csv_path, nrows=0).columns.tolist()
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, nrows=0, encoding="latin-1").columns.tolist()


def _convert(jsonl_path, parquet_path, split):
    rows = [json.loads(l) for l in open(jsonl_path)]
    records = [{
        "prompt":       _messages(r),
        "data_source":  r["source"],
        "reward_model": {"ground_truth": json.dumps(r["groundtruth"])},
        # dataset_columns is consumed by reward.extract_causci to map predicted names to real columns
        "extra_info":   {"csv_path": r["csv_path"], "dataset_columns": _columns(r["csv_path"]),
                         "split": split, "id": str(r["id"])},
    } for r in rows]
    pd.DataFrame(records).to_parquet(parquet_path, index=False)
    print(f"{len(records):>4} rows → {parquet_path}")


def main():
    _convert(config.TRAIN_RL_JSONL, config.TRAIN_RL_PARQUET, "train")
    _convert(config.TEST_RL_JSONL,  config.TEST_PARQUET,     "test")


if __name__ == "__main__":
    main()
