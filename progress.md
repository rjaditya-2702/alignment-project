# Progress

## Lifecycle Overview

```
src/data/build_dataset.py      →  dataset/unified.jsonl        (112,476 rows)
src/data/split_dataset.py      →  dataset/train.jsonl           (102,050)
                                   dataset/test.jsonl             (10,426)
src/data/preprocess.py         →  output/train.jsonl            (102,050)
                                   output/test.jsonl              (10,426)

src/eval/eval.py (baseline)    →  output/eval/metrics.json       (floor)
src/training/train.py          →  output/checkpoints/final/
src/eval/eval.py (post)        →  output/eval_post_grpo/metrics.json

train.ipynb                    — notebook covering all phases top-to-bottom
```

All phases complete. Run order: build → split → preprocess → baseline eval → train → post eval.

---

## ✅ Phase 1: Data Creation — COMPLETE

### Files Created

**`logs/log_config.ini`**
Minimal Python logging config required by `causci_bench/synthetic/generation/synthetic_generators.py` at import time. Without it, importing any `causci_bench.synthetic.generation` submodule crashes with `KeyError: 'formatters'`.

**`/opt/homebrew/anaconda3/envs/alignment/lib/python3.10/site-packages/causci_bench.pth`**
Registers `original_data/CauSciBench` on `sys.path` so `import causci_bench` resolves. Required because CauSciBench has no `setup.py` and uses absolute internal imports.

**`dataset/unified.jsonl`** — 112,476 examples (all 4 sources combined)

**`dataset/train.jsonl`** — 102,050 examples (synthetic only)

**`dataset/test.jsonl`** — 10,426 examples (original benchmarks only)

**`dataset/ckpt_*.jsonl`** — Per-step checkpoints; re-runs skip completed steps.

### src/data/ scripts

| File | What it does |
|---|---|
| `data.py` | Loads CLadder (HuggingFace) and CauSciBench (local JSON). Exports prompt templates. |
| `synthetic_cladder.py` | Generates CLadder synthetic examples via `causalbenchmark` + `RandomBuilder`. 47 stories × 9 query types = 101,600 examples. |
| `synthetic_causci.py` | Generates CauSciBench synthetic examples via `causci_bench` generators + `gpt-5.4-mini` for context. 9 methods × 50 = 450 examples. |
| `build_dataset.py` | Orchestrates all 4 sources, checkpoints each step, writes `unified.jsonl`. |
| `split_dataset.py` | Splits `unified.jsonl` → `train.jsonl` (synthetic) and `test.jsonl` (original benchmarks). |

### Final dataset counts

| Split | Count | Sources |
|---|---|---|
| train | 102,050 | cladder_synthetic (101,600) + causcibench_synthetic (450) |
| test | 10,426 | cladder (10,112) + causcibench (314) |
| **total** | **112,476** | |

### Conda environment
Rebuilt from Python 3.11 → Python 3.10. `pomegranate==0.14.8` requires old Cython API and a CPython header that moved in 3.11 arm64.

---

## ✅ Phase 2: Preprocessing — COMPLETE

**Script:** `src/data/preprocess.py`

Reads `dataset/train.jsonl` and `dataset/test.jsonl`, rebuilds prompts with updated templates, normalizes labels, and writes `output/train.jsonl` and `output/test.jsonl`.

### What it does

- **CLadder**: rebuilds prompt using `CLADDER_PROMPT` (new template with `{verbalized_story}`, detailed query type reference, and reasoning guidance). Extracts the scenario blob from the existing prompt. Normalizes label to lowercase `"yes"`/`"no"`. Flags rows with null step3/step4 with `has_nan_reasoning: true`.
- **CauSciBench**: reloads each CSV, computes enhanced metadata (`shape`, `low_cardinality`), rebuilds prompt using `CAUSCI_PROMPT` (includes full method reference guide). Parses `dataset_description`, `file_path`, `query` from existing prompt text.
- Normalizes source names: `cladder_synthetic` → `cladder`, `causcibench_synthetic` → `causcibench`.
- Adds `split` field to every row.

### Output counts

| Split | CLadder | CauSciBench | Total |
|---|---|---|---|
| train | 101,600 | 450 | 102,050 |
| test | 10,112 | 314 | 10,426 |

### Validation results

- CLadder label balance — train: yes=48,861 / no=52,739 | test: yes=5,056 / no=5,056
- CauSciBench methods in test: `ipw` and `diff_in_means` missing (not in original benchmark test set — expected)
- 4 CauSciBench prompts exceed 5,000 words: `causci_real_44`, `causci_real_115`, `causci_real_116`, `causci_real_117`
- No CSV load failures

### ⚠️ Known issue: 13,700 train CLadder rows have null step4

The `causalbenchmark` generator produces some entries (from mediation/chain/fork/collision/arrowhead graph types) where the reasoning dictionary is empty — resulting in `step4 = ""`. These rows are kept in train with `step4 = null` and `has_nan_reasoning: true`. They still have valid `step1`, `step2`, `step3`, and labels, so they contribute to method classification training but not to full reasoning chain training. **Decision: kept as-is, not filtered.**

---

## ✅ Phase 3: Baseline Eval — COMPLETE

**Scripts:** `src/eval/`

Generates completions from Qwen3-14B (greedy, temp=0), parses per-step outputs, executes code in subprocess sandbox, scores each row, and writes results + aggregate metrics to `output/eval/`.

### src/eval/ scripts

| File | What it does |
|---|---|
| `sandbox.py` | Runs model-generated Python in isolated subprocesses. `execute_code` → single run; `execute_batch` → ThreadPoolExecutor parallel. 120s timeout. Returns `{ok, stdout, stderr, result}` where `result` is the `result=...` line. |
| `parser.py` | Extracts step1–5 from completions via regex. CLadder: normalizes step2 to known query type, extracts ```python block from step4, normalizes step5 to yes/no. CauSciBench: normalizes step2 to known method, extracts code, parses step5 numeric. |
| `metrics.py` | Per-step scoring for CLadder (11+15+24+30+20) and CauSciBench (5+5+15+30+10+30+5). CLadder step3 uses gpt-4o-mini LLM judge (0/1/2 → 0/12/24 pts). CauSciBench step5 scored by relative error (≤50% → 30pts, ≤1% → +5pts exact bonus). Aggregates accuracy, avg score, per-query-type and per-method breakdowns. |
| `eval.py` | Entry point. Loads `output/test.jsonl`, batched generation (BATCH_SIZE=4), calls parser + sandbox + metrics, writes `output/eval/results.jsonl` and `output/eval/metrics.json`, prints summary table. |

### Scoring rubrics

**CLadder (100 pts):**
- Step 1 (structure): 11 — has at least one `->` arrow
- Step 2 (query type): 15 — exact match to one of 10 known types
- Step 3 (estimand): 24 — LLM judge (0/12/24); fallback: 12 if non-empty when judge disabled
- Step 4 (code): 30 — code runs and produces `result=`; 10 if code present but fails
- Step 5 (answer): 20 — yes/no exact match

**CauSciBench (100 pts):**
- Step 1 (variable ID): 5 — treatment + outcome identified
- Step 2 (method): 5 — exact match to one of 9 methods
- Step 3 (spec): 15 — non-empty
- Step 4 (code): 30 — runs and produces `result=`; 10 if code present but fails
- Bonus: 10 — if method correct AND code ran
- Step 5 (numeric): 30 — relative error ≤50%; 0 if ≥100%
- Step 5 exact: 5 — relative error ≤1%

### Usage

```bash
# Full eval (10,426 rows) — run on GPU node
python src/eval/eval.py --model Qwen/Qwen3-14B --workers 8

# Quick smoke test
python src/eval/eval.py --limit 50 --no-llm-judge

# After GRPO training
python src/eval/eval.py --model output/checkpoints/final --output-dir output/eval_post_grpo
```

---

## ✅ Phase 4: GRPO Training — COMPLETE

**Scripts:** `src/training/`

### src/training/ scripts

| File | What it does |
|---|---|
| `reward.py` | Per-step reward functions for both sources. `compute_rewards(completions, rows)` runs parse + sandbox in parallel, returns list of scalar rewards. |
| `train.py` | GRPO training loop. Loads policy with LoRA, generates N rollouts per prompt, scores them, computes GRPO loss, steps optimizer. Saves checkpoints to `output/checkpoints/`. |

### Reward rubric (training — differs from eval)

**CLadder (max 100, cascading -100 per failed step):**
- Step 1: 11 (arrows) or -100
- Step 2: 15 (query type exact) or -100; cascade: wrong type → step 3 also -100
- Step 3: 24 (non-empty + math notation proxy) or -100  ← no LLM judge during training
- Step 4: 30 (code runs + result) or -100
- Step 5: 20 (yes/no match) or -100

**CauSciBench (max 100, independent -50 per failed step, no cascade):**
- Step 1 (30 pts total): treatment=5, outcome=5, control Jaccard×15, special var=5
  - Special var (instrument/running_variable/time_variable/group_variable): +5 if model correctly identifies active variable, or correctly predicts "none" for all when gt has none. 0 if hallucinated or missed.
- Step 2: 30 (method exact) or -50
- Step 3: 0 (no ground truth, skipped)
- Step 4: 10 (code runs + result) or -50
- Step 5: 30/20/10 (relative error ≤10%/25%/50%) or -50

### Memory strategy

Single model (Qwen3-14B bfloat16 ~28GB) with LoRA adapters. Reference logprobs computed by temporarily disabling adapters (`model.disable_adapter_layers()`), avoiding a second 28GB model copy. Gradient checkpointing enabled.

### Usage

```bash
# Baseline eval (run first to establish floor)
python src/eval/eval.py --model Qwen/Qwen3-14B --workers 8

# Train
python src/training/train.py --model Qwen/Qwen3-14B

# Resume from checkpoint
python src/training/train.py --resume output/checkpoints/step_500

# Post-training eval
python src/eval/eval.py --model output/checkpoints/final --output-dir output/eval_post_grpo
```

### Key hyperparameters

| Param | Value |
|---|---|
| N rollouts | 8 |
| LoRA r | 16 |
| β (KL coeff) | 0.01 |
| LR | 2e-5 |
| Grad accum | 8 |
| Temperature | 0.8 |
| Max new tokens | 2048 |
