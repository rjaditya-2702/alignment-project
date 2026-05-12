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

Uses TRL's `GRPOTrainer` with vLLM for rollout generation. For each prompt, N completions are sampled, scored by a reward function, and the policy is updated via Group Relative Policy Optimization with a KL penalty against the base model.

The prompt asks the model to reason inside a `<think>` block and return a structured JSON object with all 5 steps.

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

## Policy model

`Qwen/Qwen3-8B` with QLoRA adapters (r=32, all attention + MLP projections).

Judge for reward scoring: `deepseek-ai/deepseek-math-7b-instruct` (RL only).

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

```bash
python src/training/train.py
```

Runs preprocessing with the RL prompt, then starts GRPO training. Everything is configured in `src/config.py`.

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
