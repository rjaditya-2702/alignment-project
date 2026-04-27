"""
Build the unified causal alignment training dataset.

Pipeline:
  1. CLadder HuggingFace        (~10K)  binary, formal causal reasoning
  2. CLadder pre-generated synth (~4K)  binary, commonsensical/anti/nonsensical/det
  3. CauSciBench existing        (~316) continuous, real + qrdata + existing synthetic
  4. CauSciBench new synthetic   (~450) continuous, 50 per method x 9 methods

Checkpoints saved to dataset/ckpt_*.jsonl after each step.
Output: dataset/unified.jsonl  (one JSON object per line)
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data import load_cladder, load_causcibench
from synthetic_cladder import load_cladder_synthetic
from synthetic_causci import generate_causci_synthetic

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR  = ROOT / "dataset"
OUTPUT_FILE = OUTPUT_DIR / "unified.jsonl"

CKPT = {
    1: OUTPUT_DIR / "ckpt_1_cladder_hf.jsonl",
    2: OUTPUT_DIR / "ckpt_2_cladder_synth.jsonl",
    3: OUTPUT_DIR / "ckpt_3_causci_existing.jsonl",
    4: OUTPUT_DIR / "ckpt_4_causci_synth.jsonl",
}

N_CAUSCI_SYNTHETIC_PER_METHOD = 50


def _save(rows, path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def build():
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_data = []

    print("=" * 50)

    if CKPT[1].exists():
        print("1. CLadder (HuggingFace) — loading checkpoint")
        cladder = _load(CKPT[1])
    else:
        print("1. CLadder (HuggingFace)")
        cladder = load_cladder()
        _save(cladder, CKPT[1])
    print(f"   {len(cladder)} examples")
    all_data.extend(cladder)

    if CKPT[2].exists():
        print("2. CLadder pre-generated synthetic — loading checkpoint")
        cladder_synth = _load(CKPT[2])
    else:
        print("2. CLadder pre-generated synthetic")
        cladder_synth = load_cladder_synthetic()
        _save(cladder_synth, CKPT[2])
    print(f"   {len(cladder_synth)} examples")
    all_data.extend(cladder_synth)

    if CKPT[3].exists():
        print("3. CauSciBench existing — loading checkpoint")
        causci = _load(CKPT[3])
    else:
        print("3. CauSciBench existing (real + qrdata + synthetic)")
        causci = load_causcibench()
        _save(causci, CKPT[3])
    print(f"   {len(causci)} examples")
    all_data.extend(causci)

    if CKPT[4].exists():
        print("4. CauSciBench new synthetic — loading checkpoint")
        causci_synth = _load(CKPT[4])
    else:
        print(f"4. CauSciBench new synthetic ({N_CAUSCI_SYNTHETIC_PER_METHOD}/method x 9 methods)")
        causci_synth = generate_causci_synthetic(n_per_method=N_CAUSCI_SYNTHETIC_PER_METHOD, seed=42)
        _save(causci_synth, CKPT[4])
    print(f"   {len(causci_synth)} examples")
    all_data.extend(causci_synth)

    # Deduplicate by prompt — keep first occurrence
    seen = set()
    deduped = []
    for row in all_data:
        if row["prompt"] not in seen:
            seen.add(row["prompt"])
            deduped.append(row)
    print(f"Dropped {len(all_data) - len(deduped)} duplicate prompts")
    all_data = deduped

    print("=" * 50)
    print(f"Total: {len(all_data)} examples")

    with open(OUTPUT_FILE, "w") as f:
        for row in all_data:
            f.write(json.dumps(row) + "\n")
    print(f"Saved → {OUTPUT_FILE}")

    sources     = Counter(d["source"] for d in all_data)
    label_types = Counter(d["label_type"] for d in all_data)
    print(f"\nSources:     {dict(sources)}")
    print(f"Label types: {dict(label_types)}")

    cladder_qt = Counter(
        d["groundtruth"]["step2"]
        for d in all_data
        if d["source"] in ("cladder", "cladder_synthetic")
        and d["groundtruth"]["step2"]
    )
    print(f"\nCLadder query types: {dict(cladder_qt)}")

    causci_methods = Counter(
        d["groundtruth"]["step2"]
        for d in all_data
        if d["source"] in ("causcibench", "causcibench_synthetic")
    )
    print(f"CauSciBench methods: {dict(causci_methods)}")

    return all_data


if __name__ == "__main__":
    build()
