# Causal Alignment

## What is the task

Fine-tune a language model to solve causal inference problems by producing structured, step-by-step reasoning chains. Two benchmarks:

- **CLadder** — binary causal queries (yes/no). Covers 10 query types: marginal, correlation, ATE, ETT, NDE, NIE, counterfactual, backdoor adjustment, collider bias, explaining away.
- **CauSciBench** — continuous causal effect estimation. Covers 9 estimation methods: OLS, IPW, matching, DiD, RDD, IV, frontdoor, GLM, difference-in-means.

For each problem the model must produce 5 steps: (1) identify causal structure, (2) select query type or method, (3) derive the estimand or estimation spec, (4) implement and compute, (5) report the answer.

---

## Training methods

Two training approaches are implemented, each with its own prompt format and output directory.

### RL-based (GRPO)

**Script:** `src/training/train.py`

Uses TRL's `GRPOTrainer` with vLLM in colocate mode for rollout generation. For each prompt, N completions are sampled, scored by a reward function, and the policy is updated via Group Relative Policy Optimization with a KL penalty against the base model.

The model reasons inside a `<think>` block (Qwen3 thinking mode enabled via `enable_thinking=True`) and returns a structured JSON object with all 5 steps. The `<think>\n` prefill is injected into every prompt before generation.

**Reward functions:**

- *CLadder*: cascade scoring — step1 (graph, judge), step2 (query type, exact match), step3 (estimand, judge), step5 (answer, exact match). Wrong graph → −1.0 immediately; wrong query type → −0.5 immediately; wrong estimand applies a −0.25 penalty to the final score.
- *CauSciBench*: cascade scoring — method (exact match), treatment/outcome (exact match), controls overlap (≥0.75 threshold), effect accuracy (within 5% of reference via estimation library). Wrong method → −1.0 immediately; wrong treatment or outcome → −0.5 immediately.
- Parse failure (malformed JSON) → −1.0 for both benchmarks.

**Judge:** `Qwen/Qwen2.5-72B-Instruct` served as a local vLLM API on port 8001 (GPU 2-3). Scores step1 and step3 for CLadder (binary 0/1).

Output: `src/output_RL/`

### SFT-based (LoRA fine-tuning)

**Scripts:** `src_finetune/train_sft.py` (single GPU), `src_finetune/train_sft_ddp.py` (multi-GPU DDP)

QLoRA fine-tuning with a custom loss:
- CE over thinking tokens (steps 1–4 inside `<think>`)
- λ × CE over the answer token (Yes/No) immediately after `</think>`

The prompt asks the model to reason inside a `<think>` block and output a single word answer.

Output: `src/output_fine_tune_lora/`

### Prompt differences

The two training methods use different prompt templates defined in their respective training scripts. Both are passed to `preprocess()` at startup, which rebuilds the dataset with the correct template for that run.

---

## Hardware

4 × GH200 GPUs (96 GB HBM3 each, 384 GB total).

| GPUs | Role |
|------|------|
| 0–1 | Qwen3-8B policy — TRL training + vLLM rollout (colocate mode) |
| 2–3 | Qwen2.5-72B-Instruct judge — frozen vLLM inference server |

---

## Policy model

`Qwen/Qwen3-8B` with thinking mode enabled.

Judge for reward scoring: `Qwen/Qwen2.5-72B-Instruct` (RL only, local vLLM server).

---

## Dataset

102,050 synthetic training examples (CLadder + CauSciBench). Test set: original benchmark examples held out entirely.

---

## Codebase structure

```
src/
  config.py                    — single source of truth: model names, paths, all hyperparameters

  data/
    preprocess.py              — rebuild prompts with caller-supplied templates, normalize labels
                                 called at the start of each training script with that script's prompts

  eval/
    parser.py                  — extract per-step fields from completions via regex
    metrics.py                 — per-step scoring + judge calls + aggregate metrics
    eval.py                    — entry point: load model → generate → parse → score → write results

  training/
    train.py                   — GRPO training loop (TRL + vLLM)
    tool_calling.py            — estimation library: loads CSV, runs the correct estimator
                                 (OLS, IPW, matching, DiD, RDD, IV, frontdoor, GLM),
                                 returns float effect estimate for CauSciBench reward scoring

  output_RL/                   — written by train.py
    train.jsonl / test.jsonl   — preprocessed data (RL prompt format)
    checkpoints/               — step_N/, epoch_N/, final/
    plots/

src_finetune/
  train_sft.py                 — SFT QLoRA, single GPU
  train_sft_ddp.py             — SFT QLoRA, multi-GPU DDP

  output_fine_tune_lora/       — written by train_sft*.py
    train.jsonl / test.jsonl   — preprocessed data (SFT prompt format)
    checkpoints/
    plots/

dataset/
  train.jsonl                  — raw synthetic training examples
  test.jsonl                   — raw original benchmark examples
```

---

## How to run

### RL training (GRPO)

**Step 1 — Launch the judge server on GPU 2-3:**

```bash
CUDA_VISIBLE_DEVICES=2,3 vllm serve Qwen/Qwen2.5-72B-Instruct \
    --port 8001 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16
```

**Step 2 — Once the server is ready, start training on GPU 0-1:**

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 src/training/train.py
```

This runs preprocessing with the RL prompt first, then starts GRPO training. Everything is configured in `src/config.py`.

**Or submit as a SLURM job (handles both steps automatically):**

```bash
sbatch run_rl_script.sh
```

Checkpoints saved every `SAVE_EVERY` steps and at end of each epoch. Final weights: `src/output_RL/checkpoints/final/`.

### SFT training — single GPU

```bash
python src_finetune/train_sft.py
```

### SFT training — multi-GPU DDP

```bash
torchrun --nproc_per_node=NUM_GPUS src_finetune/train_sft_ddp.py
```

Both SFT scripts run preprocessing with the SFT prompt before training. Final weights: `src/output_fine_tune_lora/final/`.

---

## How to evaluate

```bash
python src/eval/eval.py
```

Loads from `src/output_RL/checkpoints/final` (set via `FINAL_MODEL` in `src/config.py`). Runs greedy generation over the test set, scores each row, writes results to `src/output_RL/eval/`.

Output: `results.jsonl` (per-row) and `metrics.json` (aggregated), plus a summary table broken down by query type (CLadder) and method (CauSciBench).
