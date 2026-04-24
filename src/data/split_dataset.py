"""
Split unified.jsonl into train and test sets.

Train: synthetic (generated) data only
  - cladder_synthetic
  - causcibench_synthetic

Test: original benchmark data only
  - cladder        (HuggingFace)
  - causcibench    (existing real/qrdata/synthetic)

Output:
  dataset/train.jsonl
  dataset/test.jsonl
"""

import json
from collections import Counter
from pathlib import Path

INPUT  = Path("dataset/unified.jsonl")
TRAIN  = Path("dataset/train.jsonl")
TEST   = Path("dataset/test.jsonl")

TRAIN_SOURCES = {"cladder_synthetic", "causcibench_synthetic"}
TEST_SOURCES  = {"cladder", "causcibench"}


def split():
    train, test = [], []

    with open(INPUT) as f:
        for line in f:
            row = json.loads(line)
            if row["source"] in TRAIN_SOURCES:
                train.append(row)
            elif row["source"] in TEST_SOURCES:
                test.append(row)

    with open(TRAIN, "w") as f:
        for row in train:
            f.write(json.dumps(row) + "\n")

    with open(TEST, "w") as f:
        for row in test:
            f.write(json.dumps(row) + "\n")

    print(f"Train: {len(train):>7}  →  {TRAIN}")
    print(f"Test:  {len(test):>7}  →  {TEST}")

    print(f"\nTrain sources: {dict(Counter(r['source'] for r in train))}")
    print(f"Test sources:  {dict(Counter(r['source'] for r in test))}")


if __name__ == "__main__":
    split()
