#!/bin/bash
set -e

export OPENAI_API_KEY="your-key-here"

# export HF_HOME=/iopsstor/scratch/cscs/ajannali/.cache/huggingface
# export TOKENIZERS_PARALLELISM=false

# ── Step 1 ────────────────────────────────────────────────────────────────────
# Input:  HuggingFace CLaDDer dataset (downloaded automatically)
#         original_data/CauSciBench/** (local JSON + CSVs)
#         original_data/Cladder/** (local story YAMLs, for synthetic CLaDDer)
#         OpenAI API (generates realistic context for CauSciBench synthetic rows)
# Output: dataset/ckpt_1_cladder_hf.jsonl
#         dataset/ckpt_2_cladder_synth.jsonl
#         dataset/ckpt_3_causci_existing.jsonl
#         dataset/ckpt_4_causci_synth.jsonl   + dataset/synthetic_causci/*.csv
#         dataset/unified.jsonl
echo "Step 1/2 — building unified dataset..."
python /Users/rjaditya/Documents/projects/causal_alignment/src/data/build_dataset.py
echo "Done: dataset/unified.jsonl"

# ── Step 2 ────────────────────────────────────────────────────────────────────
# Input:  dataset/unified.jsonl
# Output: dataset/train.jsonl  (synthetic sources: cladder_synthetic, causcibench_synthetic)
#         dataset/test.jsonl   (benchmark sources: cladder, causcibench)
echo "Step 2/2 — splitting into train / test..."
python /Users/rjaditya/Documents/projects/causal_alignment/src/data/split_dataset.py
echo "Done: dataset/train.jsonl  dataset/test.jsonl"

echo ""
wc -l dataset/train.jsonl dataset/test.jsonl
