#!/bin/bash
set -e

export OPENAI_API_KEY="your-key-here"   # only needed if ckpt_4 (causci synthetic) is missing

# CLadder synthetic is generated separately as lossless raw records (needs the
# pomegranate/numpy<2 env), if not already present:
#   conda run -n cladder python src/data/synthetic_cladder.py   → dataset/cladder_synth_raw.jsonl

# Build train/test directly (no unified intermediate).
#   TRAIN (synthetic): cladder_synth_raw.jsonl + ckpt_4_causci_synth.jsonl
#   TEST  (benchmark): ckpt_1_cladder_hf.jsonl + ckpt_3_causci_existing.jsonl
echo "Building train / test sets..."
python /Users/rjaditya/Documents/projects/causal_alignment/src/data/build_dataset.py
echo ""
wc -l dataset/train.jsonl dataset/test.jsonl
