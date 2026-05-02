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

Generates completions from Qwen3-14B (greedy, temp=0), parses per-step outputs, scores each row using heuristics + DeepSeek-Math-7B judge, and writes results + aggregate metrics to `output/eval/`.

**Judge model:** `deepseek-ai/deepseek-math-7b-instruct` — loaded at 4-bit, frozen. Used for CLadder step 3 (estimand semantic equivalence) and CauSciBench step 3 (spec appropriateness). All other steps use heuristic or exact matching.

### src/eval/ scripts

| File | What it does |
|---|---|
| `parser.py` | Extracts step1–5 from completions via regex. CLadder: normalizes step2 to known query type, normalizes step5 to yes/no. CauSciBench: normalizes step2 to known method, parses step5 numeric. |
| `metrics.py` | Per-step scoring for CLadder (11+15+24+20=70 pts) and CauSciBench (5+5+15+30+5=60 pts). Step 3 on both sources uses DeepSeek-Math judge (0/1/2 → mapped pts). Aggregates accuracy, avg score, per-query-type and per-method breakdowns. |
| `eval.py` | Entry point. Loads `output/test.jsonl` and both models (policy + judge), batched generation (BATCH_SIZE=4), calls parser + metrics, writes `output/eval/results.jsonl` and `output/eval/metrics.json`, prints summary table. |

### Scoring rubrics

**CLadder (70 pts):**
- Step 1 (structure): 11 — has at least one `->` arrow
- Step 2 (query type): 15 — exact match to one of 10 known types
- Step 3 (estimand): 24 — DeepSeek-Math judge (0/12/24)
- Step 5 (answer): 20 — yes/no exact match

**CauSciBench (60 pts):**
- Step 1 (variable ID): 5 — treatment + outcome identified
- Step 2 (method): 5 — exact match to one of 9 methods
- Step 3 (spec): 15 — DeepSeek-Math judge (0/7/15)
- Step 5 (numeric): 30 — relative error ≤50%; 0 if ≥100%
- Step 5 exact: 5 — relative error ≤1%

### Usage

```bash
# Full eval (10,426 rows) — run on GPU node
python src/eval/eval.py --model Qwen/Qwen3-14B

# Quick smoke test
python src/eval/eval.py --limit 50

# After GRPO training
python src/eval/eval.py --model output/checkpoints/final --output-dir output/eval_post_grpo
```

---

## ✅ Phase 4: GRPO Training — COMPLETE

**Scripts:** `src/training/`

**Judge model:** `deepseek-ai/deepseek-math-7b-instruct` — loaded at 4-bit alongside the policy, frozen throughout training. Provides reward signal for semantic steps (step 3 on both sources). Heuristics cover all other steps.

### src/training/ scripts

| File | What it does |
|---|---|
| `reward.py` | Per-step reward functions for both sources. `compute_rewards(completions, rows, judge_model, judge_tokenizer)` parses completions, batches step-3 judge calls by source, returns list of scalar rewards. |
| `train.py` | GRPO training loop. Loads policy (Qwen3-14B + LoRA) and judge (DeepSeek-Math-7B 4-bit), generates N rollouts per prompt, scores them, computes GRPO loss, steps optimizer. Saves checkpoints to `output/checkpoints/`. |

### Reward rubric

**CLadder (max 70, cascading -100 per failed step):**
- Step 1: 11 (arrows) or -100
- Step 2: 15 (query type exact) or -100; cascade: wrong type → step 3 also -100
- Step 3: 24 — DeepSeek-Math judge (0/12/24) or -100
- Step 5: 20 (yes/no match) or -100

**CauSciBench (max 105, independent -50 per failed step, no cascade):**
- Step 1 (30 pts total): treatment=5, outcome=5, control Jaccard×15, special var=5
  - Special var (instrument/running_variable/time_variable/group_variable): +5 if model correctly identifies active variable, or correctly predicts "none" for all when gt has none. 0 if hallucinated or missed.
- Step 2: 30 (method exact) or -50
- Step 3: 15 — DeepSeek-Math judge (0/7/15)
- Step 5: 30/20/10 (relative error ≤10%/25%/50%) or -50

### Memory strategy

Two models in process: Qwen3-14B bfloat16 (~28GB) with LoRA adapters as policy; reference logprobs computed by temporarily disabling adapters (`model.disable_adapter_layers()`), avoiding a second 28GB model copy. DeepSeek-Math-7B at 4-bit (~4GB) as frozen judge. Gradient checkpointing enabled. Total ~47GB on A100-80GB.

### Usage

```bash
# Baseline eval (run first to establish floor)
python src/eval/eval.py --model Qwen/Qwen3-14B

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
| LoRA r | 32 |
| β (KL coeff) | 0.01 |
| LR | 2e-5 |
| Grad accum | 8 |
| Temperature | 0.8 |
| Max new tokens | 2048 |
