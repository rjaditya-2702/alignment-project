# Causal Alignment

Fine-tuning Qwen3-14B on causal reasoning via GRPO (Group Relative Policy Optimization). The model learns to solve causal inference problems step-by-step across two benchmarks: CLadder (binary causal queries) and CauSciBench (continuous effect estimation).

## Overview

The training pipeline teaches the model to produce structured 5-step reasoning chains:
1. Identify causal structure (graph with arrows)
2. Select query type or estimation method
3. Derive the estimand
4. Write and execute Python code
5. Report the final answer

GRPO generates N rollouts per prompt, scores each with a reward function, normalizes rewards within the group to advantages, and optimizes the policy with a KL-penalized objective.

## Benchmarks

| Benchmark | Task | Label | Query types / Methods |
|---|---|---|---|
| CLadder | Binary causal query (yes/no) | `yes` / `no` | 10 types (ate, ett, nde, nie, ...) |
| CauSciBench | Causal effect estimation | continuous float | 9 methods (ols, ipw, iv, rdd, did, ...) |

## Dataset

| Split | CLadder | CauSciBench | Total |
|---|---|---|---|
| train | 101,600 (synthetic) | 450 (synthetic) | 102,050 |
| test | 10,112 (original) | 314 (original) | 10,426 |

Synthetic CLadder examples are generated via `causalbenchmark` (47 stories × 9 query types). Synthetic CauSciBench examples are generated via `causci_bench` generators with GPT context.

## Project Structure

```
src/
  data/
    data.py               — load CLadder (HuggingFace) and CauSciBench (local JSON), prompt templates
    synthetic_cladder.py  — generate synthetic CLadder examples
    synthetic_causci.py   — generate synthetic CauSciBench examples
    build_dataset.py      — orchestrate all 4 sources → dataset/unified.jsonl
    split_dataset.py      — split into dataset/train.jsonl + dataset/test.jsonl
    preprocess.py         — rebuild prompts, normalize labels → output/train.jsonl + output/test.jsonl
  eval/
    sandbox.py            — subprocess code executor (120s timeout, ThreadPoolExecutor)
    parser.py             — parse model completions into per-step fields
    metrics.py            — per-step scoring, LLM judge for CLadder step3, aggregate metrics
    eval.py               — entry point: batched generation → parse → sandbox → score → write results
  training/
    reward.py             — reward functions for GRPO (CLadder cascading, CauSciBench independent)
    train.py              — GRPO loop with LoRA, gradient checkpointing, single-model reference

dataset/
  unified.jsonl           — 112,476 combined examples
  train.jsonl             — 102,050 synthetic examples
  test.jsonl              — 10,426 original benchmark examples

output/
  train.jsonl             — preprocessed train set (updated prompts + normalized labels)
  test.jsonl              — preprocessed test set
  eval/                   — baseline eval results (results.jsonl, metrics.json)
  eval_post_grpo/         — post-training eval results
  checkpoints/            — LoRA checkpoints (step_N/, final/)

train.ipynb               — end-to-end notebook: data → inspect → reward check → train → eval
```

## Setup

Requires Python 3.10 (`pomegranate==0.14.8` breaks on 3.11+).

```bash
conda create -n alignment python=3.10
conda activate alignment
pip install -r requirements.txt

# GPU node additionally needs:
pip install torch transformers peft accelerate
```

Register CauSciBench as a local package (no setup.py):
```bash
echo "/path/to/causal_alignment/original_data/CauSciBench" \
  > $(python -c "import site; print(site.getsitepackages()[0])")/causci_bench.pth
```

Set your OpenAI API key (used for synthetic CauSciBench generation and the CLadder step3 LLM judge):
```bash
export OPENAI_API_KEY=sk-...
```

## Run Order

```bash
# 1. Build dataset (all 4 sources, checkpointed)
python src/data/build_dataset.py

# 2. Split into train/test
python src/data/split_dataset.py

# 3. Preprocess (rebuild prompts, normalize labels)
python src/data/preprocess.py

# 4. Baseline eval (requires GPU, ~10k rows)
python src/eval/eval.py --model Qwen/Qwen3-14B --workers 8

# 5. GRPO training
python src/training/train.py --model Qwen/Qwen3-14B

# Resume from checkpoint
python src/training/train.py --resume output/checkpoints/step_500

# 6. Post-training eval
python src/eval/eval.py --model output/checkpoints/final --output-dir output/eval_post_grpo
```

For a guided run with inspection at each stage, use `train.ipynb`.

## Reward Functions

**CLadder** (max 100, cascading −100 penalty per failed step):

| Step | Points | Criterion |
|---|---|---|
| 1: Graph | 11 | at least one `->` arrow |
| 2: Query type | 15 | exact match to known type |
| 3: Estimand | 24 | non-empty with math notation; cascades from step 2 |
| 4: Code | 30 | executes and produces `result=` |
| 5: Answer | 20 | `yes`/`no` exact match |

**CauSciBench** (max 100, independent −50 penalty per failed step):

| Step | Points | Criterion |
|---|---|---|
| 1: Variables | 30 | treatment(5) + outcome(5) + controls Jaccard(15) + special var(5) |
| 2: Method | 30 | exact match to known method |
| 3: Spec | — | no ground truth, skipped |
| 4: Code | 10 | executes and produces `result=` |
| 5: Numeric | 30 | relative error ≤10%→30, ≤25%→20, ≤50%→10, else −50 |

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen3-14B |
| LoRA rank | 16 |
| LoRA α | 32 |
| KL coefficient β | 0.01 |
| Rollouts per prompt (N) | 8 |
| Learning rate | 2e-5 |
| Gradient accumulation | 8 |
| Generation temperature | 0.8 |
| Max new tokens | 2048 |

LoRA adapters are applied to all attention and MLP projection layers (~20M trainable params out of 14B). Reference logprobs are computed by temporarily disabling adapters (`model.disable_adapter_layers()`), avoiding a second 28GB model copy.
