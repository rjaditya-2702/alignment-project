# Causal Alignment

Fine-tune a language model to solve causal inference problems via structured 5-step reasoning: (1) identify causal structure, (2) select query type / method, (3) derive the estimand, (4) compute, (5) report the answer.

Two benchmarks:
- **CLadder** — binary causal queries (yes/no). 10 query types: marginal, correlation, ATE, ETT, NDE, NIE, counterfactual, backdoor adjustment, collider bias, explaining away.
- **CauSciBench** — continuous causal-effect estimation. 9 methods: OLS, IPW, matching, DiD, RDD, IV, frontdoor, GLM, difference-in-means.

## Models

- **Policy:** `Qwen/Qwen3-8B` with extended thinking (`enable_thinking=True`)
- **Judge:** `Qwen/Qwen3-8B` served locally on port 8001 — scores CLadder step1 (graph) and step3 (estimand) as 0/1

## Hardware

4 × GH200 (96 GB each). For RL: GPUs 0–2 run the policy (training + vLLM/Ray rollout), GPU 3 runs the frozen judge server.

## Data

Raw splits (never modified): `dataset/train.jsonl` (CLadder + CauSciBench synthetic, ~87K) and `dataset/test.jsonl` (original benchmark, ~9K).

`preprocess(which=...)` rebuilds prompts with the caller's templates and partitions the data:

- **Train is split disjointly between SFT and RL** by a deterministic per-`id` hash — `which="sft"` keeps 30% of each source, `which="rl"` keeps the remaining 70%. The split is identical across the two runs (same id → same side), so SFT and RL never train on the same example.
- **Test is not split** — both methods evaluate on the full benchmark.

Output is per-source so a trainer can merge CLadder + CauSciBench or load just one:

```
<output_dir>/
  train_{sft|rl}_cladder.jsonl   train_{sft|rl}_causci.jsonl   # disjoint 30/70 train
  test_cladder.jsonl             test_causci.jsonl             # full benchmark, shared
```

Each trainer picks which to merge — `TRAIN_FILES`/`TEST_FILES` in `train_sft_ddp.py`, the `_convert_split([...])` lists in `data_process.py`.

## Training Methods

### 1. SFT — QLoRA (DDP)

**Script:** `src/training/train_sft_ddp.py` · **Run:** `sbatch run_sft_script.sh`

QLoRA with a weighted cross-entropy loss over the full assistant span (1.0), with the CLadder Yes/No answer token up-weighted to `ANSWER_LAMBDA = 5.0`. Trains **both** CLadder (thinking + one-word answer) and CauSciBench (JSON answer). Generation-based eval on the test set every `EVAL_EVERY = 50` steps via `eval_metrics.py`.

```
src/output_fine_tune_lora/
  train_sft_*.jsonl / test_*.jsonl   # preprocessed (SFT prompt format)
  checkpoints/step_{N}/  final/  plots/
  tokenized_data_v4.pt               # cached tokenized sequences
```

### 2. RL — GRPO via veRL (FSDP)

**Scripts:** `src/training/verl_/data_process.py` (data prep) + `verl_/reward.py` (reward) · **Run:** `sbatch run_verl.sh`

GRPO on veRL's FSDP backend. `run_verl.sh` calls `data_process.py` first to preprocess (veRL prompts) and convert JSONL → Parquet, starts the judge (vLLM) on GPU 3 and a Ray cluster on GPUs 0–2, then runs `verl.trainer.main_ppo`. Reward-based validation at an auto-computed interval (`TEST_FREQ = total_steps / 100`, min 150).

```
src/output_RL/
  train_rl_*.jsonl / test_*.jsonl    # preprocessed (veRL prompt format)
  train.parquet / test.parquet       # veRL input
  verl_checkpoints/
verl_training.log → verl_metrics.csv (via parse_verl_logs.py) → plot_verl.py
```

## Reward Function

Used by veRL. Parse failure → **−1.0** for either benchmark.

**CLadder — cascade gates** (each wrong step short-circuits):

| Step | Check | Pass | Fail |
|------|-------|------|------|
| step1 | Judge scores graph (0/1) | continue | **−1.0** (stop) |
| step2 | Exact match query type | continue | **−0.5** (stop) |
| step3 | Judge scores estimand (0/1) | 0 penalty | **−0.25** penalty |
| step5 | Exact match yes/no | **+1.0** + step3 penalty | **−0.75** + step3 penalty |

Range [−1.25, +1.0]; step3 judge only runs when step2 matches.

**CauSciBench — weighted combination** (no gates), then `reward = weighted_sum × 2 − 1` rescaling [0,1]→[−1,+1]:

| Component | Weight | Scoring |
|-----------|--------|---------|
| method (step2) | 30% | exact match |
| treatment | 15% | exact match |
| outcome | 10% | exact match |
| controls | 15% | Jaccard recall vs. ground truth |
| effect | 30% | MRE ≤ 5% vs. reference |

`library_fn` (the causal estimator: loads CSV, runs the chosen method) is only called when the predicted method matches; otherwise effect = 0.

## Metric logging (veRL)

veRL emits one console line per training step. `reward_fn` detects `split == "test"` from `extra_info`, accumulates parsed eval outputs in an in-process buffer, and flushes a summary (`compute_eval_metrics`) when training resumes (an `atexit` handler covers ending on an eval pass). `parse_verl_logs.py` turns `verl_training.log` into a CSV of native step rows + `[verl_eval]` summary rows:

```bash
python3 parse_verl_logs.py verl_training.log verl_metrics.csv
python plot_verl.py verl_metrics.csv
```

## Codebase Structure

```
src/
  config.py                  — model names, paths, hyperparameters, data file lists
  data/
    data.py                  — load CLadder (HF) + CauSciBench (local JSON)
    preprocess.py            — rebuild prompts, normalize labels, 30/70 SFT/RL train split
    build_dataset.py         — assemble dataset/train.jsonl (synthetic) + test.jsonl (benchmark)
    synthetic_cladder.py     — generate CLadder synthetic as lossless raw records
    synthetic_causci.py      — generate CauSciBench synthetic examples (via OpenAI)
  training/
    train_sft_ddp.py         — SFT QLoRA, multi-GPU DDP
    tool_calling.py          — causal estimation library (OLS/IPW/matching/DiD/RDD/IV/frontdoor/GLM → effect)
    eval_metrics.py          — CLadder & CauSciBench metrics + plots
    verl_/
      data_process.py        — preprocess + JSONL → Parquet for veRL
      reward.py              — reward + extraction (veRL interface)
  eval/
    eval_sft.py / eval_rl.py — offline checkpoint evaluation
dataset/
  train.jsonl / test.jsonl   — raw splits
  synthetic_causci/          — generated CauSciBench CSVs
original_data/               — source benchmark data + generators
parse_verl_logs.py           — verl_training.log → verl_metrics.csv
plot_verl.py                 — plot verl_metrics.csv → PNGs
run_sft_script.sh            — SLURM: SFT (4 GPUs, DDP)
run_verl.sh                  — SLURM: veRL GRPO (judge on GPU 3, Ray on 0–2)
```

## Tests

```bash
python -m pytest tests/test_reward_extraction.py -v
```

53 tests, no GPU/network — all ML deps stubbed with `unittest.mock`.

| Suite | What's covered |
|-------|----------------|
| `TestExtractCladder` (veRL) | clean JSON, `</think>` stripping, trailing-comma recovery, missing fields, unknown query type, invalid step5, case normalisation, all 10 query types |
| `TestExtractCausci` (SFT, veRL) | OLS, `</think>`, unknown method, treatment/outcome not in columns, control filtering, IV/RDD/DiD/frontdoor validation |
| `TestVeRLRewardFn` (veRL) | perfect score, wrong-estimand penalty, wrong answer, step1/step2 early-exit, unparseable, CauSciBench correct/wrong, mixed batch |

## Configuration

All hyperparameters and paths in `src/config.py`.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `POLICY_MODEL` / `JUDGE_MODEL` | `Qwen/Qwen3-8B` | Policy / judge |
| `N_ROLLOUTS` | 3 | Completions per prompt (RL) |
| `LORA_R` | 16 (RL) / 32 (SFT) | LoRA rank |
| `BETA` | 0.01 | KL penalty coefficient |
| `LR` | 2e-5 | Learning rate |
| `EVAL_MAX_TOKENS` | 4096 | Max tokens during eval generation |

## Results

Plots are in the `iteration_*` directories.

---

## Legacy

**RL — GRPO via TRL** (`train_trl.py`, `run_rl_script.sh`) — removed. The first RL attempt used TRL's `GRPOTrainer` with vLLM in colocate mode on GPUs 0–2. It worked, but the DDP backend capped throughput. I pivoted to **veRL (FSDP)**, which shards the policy for higher rollout/training throughput at this model size — the reward logic carried over unchanged, so veRL is now the only RL path.
