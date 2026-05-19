# Causal Alignment

## Task

Fine-tune a language model to solve causal inference problems via structured step-by-step reasoning. Two benchmarks:

- **CLadder** — binary causal queries (yes/no). 10 query types: marginal, correlation, ATE, ETT, NDE, NIE, counterfactual, backdoor adjustment, collider bias, explaining away.
- **CauSciBench** — continuous causal effect estimation. 9 methods: OLS, IPW, matching, DiD, RDD, IV, frontdoor, GLM, difference-in-means.

For each problem the model produces 5 steps: (1) identify causal structure, (2) select query type or method, (3) derive the estimand, (4) compute, (5) report the answer.

---

## Models

- **Policy:** `Qwen/Qwen3-8B` with extended thinking (`enable_thinking=True`)
- **Judge:** `Qwen/Qwen3-8B` served locally on port 8001 — scores CLadder step1 (graph) and step3 (estimand) as 0/1

---

## Hardware

4 × GH200 GPUs (96 GB HBM3 each).

| GPUs | Role |
|------|------|
| 0–2  | Policy model (training + vLLM rollout) |
| 3    | Judge server (frozen vLLM/sglang inference) |

---

## Training Methods

Three training approaches, each with its own prompt format, output directory, and built-in eval.

---

### 1. SFT — QLoRA (DDP)

**Script:** `src/training/train_sft_ddp.py`
**Run:** `sbatch run_sft_script.sh`

QLoRA fine-tuning with a weighted cross-entropy loss:
- Weight 1.0 on thinking tokens (steps 1–4 inside `<think>`)
- Weight 5.0 (`ANSWER_LAMBDA`) on the answer token (Yes/No after `</think>`)

**CLadder only** — CauSciBench rows are filtered out during data loading.

Output format: single word `Yes` or `No` after `</think>`.

**Eval during training:** one logit-based accuracy pass on the CLadder test set at the end of training (rank 0 only). No generation — model logits at the answer position are compared directly.

**Outputs:**
```
src/output_fine_tune_lora/
  train.jsonl / test.jsonl     — preprocessed data (SFT prompt format)
  checkpoints/step_{N}/        — periodic LoRA adapter checkpoints
  final/                       — final LoRA adapter
  plots/training_curves.png
  tokenized_data.pt            — cached tokenized sequences
```

---

### 2. RL — GRPO via TRL

**Script:** `src/training/train_trl.py`
**Run:** `sbatch run_rl_script.sh`

GRPO with TRL's `GRPOTrainer`. vLLM runs in colocate mode on GPUs 0–2 for rollout generation (6 completions per prompt). Judge server runs on GPU 3.

The model reasons inside `<think>` and returns a JSON object with all 5 steps. Both CLadder and CauSciBench are trained jointly.

**Reward functions:**
- *CLadder*: cascade — step1 graph (judge, −1.0 gate) → step2 query type (exact match, −0.5 gate) → step3 estimand (judge, −0.25 penalty) → step5 answer (exact match, ±1.0)
- *CauSciBench*: cascade — method (exact match, −1.0 gate) → treatment/outcome (exact match, −0.5 gate) → controls overlap (≥0.75) + effect within 5% of reference (±1.0 / 0.5 / −0.25)
- Parse failure → −1.0 for both

**Eval during training:** reward function runs on the test set every 1000 steps (`evaluation_strategy="steps"`, `eval_steps=1000`). Logged by `MetricsCallback`.

**Outputs:**
```
src/output_RL/
  train.jsonl / test.jsonl     — preprocessed data (TRL prompt format)
  checkpoints/step_{N}/        — periodic LoRA adapter checkpoints (every 100 steps)
  checkpoints/final/           — final LoRA adapter
  plots/
```

---

### 3. RL — GRPO via veRL (FSDP - InProgress)

**Script:** `src/training/verl_/data_process.py` (data prep) + `src/training/verl_/reward.py` (reward)
**Run:** `sbatch run_verl.sh`

Same GRPO algorithm as TRL but using veRL's FSDP backend instead of DDP, which gives higher throughput. Ray cluster on GPUs 0–2, judge server (sglang) on GPU 3.

Data must be converted to Parquet first — `run_verl.sh` calls `data_process.py` automatically before starting training.

**Eval during training:** reward-based validation every 100 steps (`trainer.test_freq=100`) using `compute_score` from `verl_/reward.py`.

**Outputs:**
```
src/output_RL/
  train.parquet / test.parquet — veRL parquet input files
  verl_checkpoints/            — checkpoints saved every 500 steps
```

---

## Dataset

| Split | Source | Rows |
|-------|--------|------|
| Train | CLadder synthetic + CauSciBench synthetic | ~87K |
| Test  | CLadder (causal-nlp/CLadder on HF) + CauSciBench original | ~9K |

Raw files: `dataset/train.jsonl`, `dataset/test.jsonl`

Each training script calls `preprocess()` at startup, which rebuilds prompts using that script's own templates and writes `train.jsonl` / `test.jsonl` to the script's output directory. The raw `dataset/` files are never modified.

---

## Codebase Structure

```
src/
  config.py                        — all model names, paths, hyperparameters

  data/
    data.py                        — load CLadder (HuggingFace) + CauSciBench (local JSON)
    preprocess.py                  — rebuild prompts with caller's template, normalize labels
    build_dataset.py               — one-time pipeline: assemble dataset/ from all sources
    split_dataset.py               — train/test split
    synthetic_cladder.py           — generate CLadder synthetic examples
    synthetic_causci.py            — generate CauSciBench synthetic examples (via OpenAI)

  training/
    train_sft_ddp.py               — SFT QLoRA, multi-GPU DDP
    train_trl.py                   — GRPO training (TRL + vLLM colocate)
    tool_calling.py                — causal estimation library: loads CSV, runs OLS/IPW/
                                     matching/DiD/RDD/IV/frontdoor/GLM, returns effect float

    verl_/
      data_process.py              — convert train/test JSONL to Parquet for veRL
      reward.py                    — reward function + extraction logic (veRL interface)

dataset/
  train.jsonl                      — raw synthetic training examples
  test.jsonl                       — raw original benchmark examples (held out)
  synthetic_causci/                — 450 generated CauSciBench CSV files

original_data/
  CauSciBench/                     — source benchmark data + causci_bench package
  Cladder/                         — CLadder RandomBuilder (synthetic generation)

logs/                              — SLURM job logs
run_sft_script.sh                  — SLURM: SFT training (4 GPUs, DDP)
run_rl_script.sh                   — SLURM: TRL GRPO training (judge on GPU 3, policy on 0–2)
run_verl.sh                        — SLURM: veRL GRPO training (judge sglang on GPU 3, Ray on 0–2)
```

---

## How to Run

All three scripts expect the cluster environment with the appropriate venv activated. Edit the `source` and `cd` lines in the SLURM scripts to match your paths before submitting.

### SFT

```bash
sbatch run_sft_script.sh
```

Runs `torchrun --nproc_per_node=4 src/training/train_sft_ddp.py` across all 4 GPUs.
Preprocessing (SFT prompt format) runs on rank 0 before training starts.

Final model → `src/output_fine_tune_lora/final/`
Eval metric → CLadder test accuracy (logit-based), printed at end of training.

### TRL GRPO

```bash
sbatch run_rl_script.sh
```

Starts judge server on GPU 3, waits for it to be healthy, then runs:
`torchrun --nproc_per_node=3 src/training/train_trl.py` on GPUs 0–2.

Preprocessing (TRL prompt format) runs on rank 0 before training starts.

Final model → `src/output_RL/checkpoints/final/`
Eval metric → mean reward on test set, logged every 1000 steps.

### veRL GRPO

```bash
sbatch run_verl.sh
```

1. Runs `data_process.py` to convert JSONL → Parquet (includes preprocessing with veRL prompts)
2. Starts judge server (sglang) on GPU 3
3. Starts Ray cluster on GPUs 0–2
4. Runs `verl.trainer.main_ppo` with GRPO config

Checkpoints → `src/output_RL/verl_checkpoints/`
Eval metric → mean reward on validation parquet, logged every 100 steps.

---

## Tests

```bash
python -m pytest tests/test_reward_extraction.py -v
```

76 tests, no GPU or network required — all ML deps are stubbed with `unittest.mock`.

| Suite | Modules tested | What's covered |
|-------|---------------|----------------|
| `TestExtractCladder` | TRL, veRL | Clean JSON, `</think>` stripping, trailing-comma recovery, missing fields, unknown query type, invalid step5, case normalisation, all 10 query types |
| `TestExtractCausci` | TRL, SFT, veRL | OLS, `</think>`, unknown method, treatment/outcome not in columns, control filtering, IV / RDD / DiD / frontdoor validation |
| `TestTRLRewardFn` | TRL | Perfect score, wrong estimand penalty (−0.25), wrong answer, wrong step1/step2 early-exit, unparseable output, CauSciBench correct/wrong, mixed batch |
| `TestVeRLRewardFn` | veRL | Same reward paths via veRL's `(solution_strs, ground_truths, extra_infos)` interface |

---

## Configuration

All hyperparameters and paths are in `src/config.py`. Change them there and every script picks them up.

Key settings:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `POLICY_MODEL` | `Qwen/Qwen3-8B` | Base model |
| `JUDGE_MODEL` | `Qwen/Qwen3-8B` | Judge for CLadder scoring |
| `N_ROLLOUTS` | 6 | Completions per prompt (RL) |
| `TRAIN_BATCH_SIZE` | 2 | Prompts per training step |
| `LORA_R` | 16 | LoRA rank (RL); 32 for SFT |
| `BETA` | 0.01 | KL penalty coefficient |
| `LR` | 2e-5 | Learning rate |
| `EVAL_MAX_TOKENS` | 4096 | Max tokens during eval generation |
