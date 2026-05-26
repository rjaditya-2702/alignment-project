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
| 3    | Judge server (frozen vLLM inference) |

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

### 3. RL — GRPO via veRL (FSDP)

**Script:** `src/training/verl_/data_process.py` (data prep) + `src/training/verl_/reward.py` (reward)
**Run:** `sbatch run_verl.sh`

Same GRPO algorithm as TRL but using veRL's FSDP backend instead of DDP, which gives higher throughput. Ray cluster on GPUs 0–2, judge server (vLLM) on GPU 3.

Data must be converted to Parquet first — `run_verl.sh` calls `data_process.py` automatically before starting training.

**Eval during training:** reward-based validation at an auto-computed interval (`TEST_FREQ = total_steps / 100`, minimum 150) using `compute_score` from `verl_/reward.py`.

After training, `parse_verl_logs.py` converts `verl_training.log` → `verl_metrics.csv`. Run `python plot_verl.py verl_metrics.csv` locally to visualize.

**Outputs:**
```
src/output_RL/
  train.parquet / test.parquet — veRL parquet input files
  verl_checkpoints/            — checkpoints saved every 150 steps
verl_training.log              — raw training log (tee'd output)
verl_metrics.csv               — parsed metrics CSV for plotting
verl_train_metrics.png         — reward/loss curves (local plotting)
verl_eval_cladder.png          — CLadder eval breakdown
verl_eval_causci.png           — CauSciBench eval breakdown
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
    eval_metrics.py                — compute CLadder & CauSciBench metrics, save plots

    verl_/
      data_process.py              — convert train/test JSONL to Parquet for veRL
      reward.py                    — reward function + extraction logic (veRL interface)
      parse_verl_logs.py           — parse verl_training.log → verl_metrics.csv

  eval/
    eval_sft.py                    — offline evaluation for SFT checkpoint
    eval_rl.py                     — offline evaluation for RL checkpoint

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
run_verl.sh                        — SLURM: veRL GRPO training (judge vLLM on GPU 3, Ray on 0–2)
plot_verl.py                       — local: plot verl_metrics.csv → PNG charts
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
2. Starts judge server (vLLM) on GPU 3
3. Starts Ray cluster on GPUs 0–2
4. Runs `verl.trainer.main_ppo` with GRPO config

Checkpoints → `src/output_RL/verl_checkpoints/`
Eval metric → mean reward on validation parquet, logged at auto-computed interval.

After the job completes, copy `verl_metrics.csv` locally and run:
```bash
python plot_verl.py verl_metrics.csv
```

---

## Reward Function

Both TRL and veRL use the same scoring logic. Parse failure always returns **−1.0** for either benchmark.

### CLadder — cascade gates

Each gate short-circuits scoring if the model gets that step wrong.

| Step | Check | Pass | Fail |
|------|-------|------|------|
| step1 | Judge scores causal graph (0/1) | continue | **−1.0** (stop) |
| step2 | Exact match on query type | continue | **−0.5** (stop) |
| step3 | Judge scores estimand (0/1) | 0 penalty | **−0.25** penalty (continue) |
| step5 | Exact match on yes/no answer | **+1.0** + step3 penalty | **−0.75** + step3 penalty |

Range: [−1.25, +1.0]. The step3 judge is only called when step2 matches.

### CauSciBench — weighted combination

No gates. All components scored independently, then combined:

| Component | Weight | Scoring |
|-----------|--------|---------|
| method (step2) | 30% | exact match → 0 or 1 |
| treatment | 15% | exact match → 0 or 1 |
| outcome | 10% | exact match → 0 or 1 |
| controls | 15% | Jaccard recall vs. ground truth → [0, 1] |
| effect | 30% | MRE ≤ 5% vs. reference → 0 or 1 |

`reward = (weighted_sum) × 2 − 1` — rescales [0, 1] → [−1, +1].

`library_fn` (the causal estimator) is only called when the predicted method matches the ground truth. If method is wrong, effect score is 0.

---

## Metric Logging and Parsing (veRL)

### During training

veRL's native console output emits one line per training step:
```
step: 42  actor/loss: 0.312  kl: 0.008  reward/mean: -0.241 ...
```

`reward_fn` also prints a line per batch call:
```
[verl] call   123  reward=-0.241  src=cladder
```

### During eval (test split)

`reward_fn` detects `split == "test"` from `extra_info`. Rewards are still computed and returned to veRL (for its own validation reporting), but each batch's parsed outputs are also accumulated in an in-process buffer. An intermediate line is printed per eval batch:
```
[verl_eval] eval_pass:0 call:  456 reward=+0.312 src=causcibench
```

When the next **training** batch arrives (split switches back to "train"), the buffer is flushed: `compute_eval_metrics()` runs over all accumulated test samples and prints one summary line:
```
[verl_eval] eval_pass:1 cladder/overall_acc:0.5100 cladder/rung1:0.6200 causci/method_acc:0.4800 ...
```
The buffer is then cleared. An `atexit` handler fires the same flush if training ends immediately after an eval pass.

### parse_verl_logs.py

Reads `verl_training.log` and separates two line types:

| Line type | Pattern | Output row |
|-----------|---------|------------|
| veRL native step | `step: N …key:value…` | `step` + metric columns |
| Eval summary | `[verl_eval] eval_pass:N …key:value…` | `eval_pass` + metric columns |

Both are written to one CSV (missing columns are NaN). Run locally:
```bash
python3 src/training/verl_/parse_verl_logs.py verl_training.log verl_metrics.csv
python plot_verl.py verl_metrics.csv
```

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
| `N_ROLLOUTS` | 3 | Completions per prompt (RL) |
| `TRAIN_BATCH_SIZE` | 1 | Prompts per training step |
| `LORA_R` | 16 | LoRA rank (RL); 32 for SFT |
| `BETA` | 0.01 | KL penalty coefficient |
| `LR` | 2e-5 | Learning rate |
| `EVAL_MAX_TOKENS` | 4096 | Max tokens during eval generation |
