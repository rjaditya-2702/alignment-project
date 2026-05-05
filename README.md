# Causal Alignment

## What is the task

Fine-tune a language model to solve causal inference problems by producing structured, step-by-step reasoning chains. Two benchmarks:

- **CLadder** — binary causal queries (yes/no). Covers 10 query types: marginal, correlation, ATE, ETT, NDE, NIE, counterfactual, backdoor adjustment, collider bias, explaining away.
- **CauSciBench** — continuous causal effect estimation. Covers 9 estimation methods: OLS, IPW, matching, DiD, RDD, IV, frontdoor, GLM, difference-in-means.

For each problem the model must produce 5 steps: (1) identify causal structure, (2) select query type or method, (3) derive the estimand or estimation spec, (4) implement and compute, (5) report the answer.

---

## High-level design

**Algorithm: GRPO (Group Relative Policy Optimization)**

For each training prompt, generate N rollouts, score each with a reward function, normalize rewards within the group to advantages, and minimize the KL-penalized policy gradient loss:

```
loss = -mean(advantage * log_prob_policy) + β * KL(policy || reference)
```

**Policy:** Qwen3-14B with LoRA adapters (r=32, all attention + MLP projections). Reference logprobs come from the same base weights with adapters temporarily disabled — no second model copy needed.

**Judge:** DeepSeek-Math-7B-Instruct at 4-bit, frozen. Used to score step 3 (estimand / estimation spec) where exact matching is impossible. All other steps use heuristics.

**Reward design:**
- CLadder: cascading −100 penalty per failed step (wrong query type → step 3 also penalized)
- CauSciBench: independent −50 per failed step; numeric answer scored by relative error tiers (≤10% → 30 pts, ≤25% → 20, ≤50% → 10, else −50)

**Dataset:** 102,050 synthetic training examples (101,600 CLadder via `causalbenchmark`, 450 CauSciBench via `causci_bench` + GPT context). Test set: 10,426 original benchmark examples held out entirely.

**Multi-GPU:** On 2 GPUs, generation (GPU 0) and reference logprob + reward computation (GPU 1) are pipelined to overlap.

---

## Codebase structure

```
src/
  config.py               — single source of truth: model names, paths, all hyperparameters

  data/
    data.py               — load CLadder (HuggingFace) + CauSciBench (local JSON), prompt templates
    synthetic_cladder.py  — generate synthetic CLadder via causalbenchmark + RandomBuilder
    synthetic_causci.py   — generate synthetic CauSciBench via causci_bench generators + GPT context
    build_dataset.py      — orchestrate all 4 sources → dataset/unified.jsonl (checkpointed)
    split_dataset.py      — split unified.jsonl → dataset/train.jsonl + dataset/test.jsonl
    preprocess.py         — rebuild prompts with updated templates, normalize labels → output/

  eval/
    parser.py             — extract per-step fields from completions via regex
    metrics.py            — per-step scoring + DeepSeek-Math judge + aggregate metrics
    eval.py               — entry point: load model → generate → parse → score → write results

  training/
    reward.py             — reward functions (heuristics + batched judge calls)
    train.py              — GRPO loop: LoRA policy, gradient checkpointing, single/multi-GPU

dataset/
  unified.jsonl           — 112,476 combined examples
  train.jsonl             — 102,050 synthetic
  test.jsonl              — 10,426 original benchmarks

output/
  train.jsonl             — preprocessed train (rebuilt prompts, normalized labels)
  test.jsonl              — preprocessed test
  checkpoints/            — LoRA checkpoints: step_N/, epoch_N/, final/
  eval/                   — baseline eval: results.jsonl, metrics.json
  eval_post_grpo/         — post-training eval
```

---

## How to run training

```bash
python src/training/train.py
```

No arguments. Everything is configured in `src/config.py` — model, hyperparameters, paths.

Auto-resume: on startup, scans `output/checkpoints/` for the latest `step_N/` checkpoint and resumes from there. If none exists, starts fresh from `POLICY_MODEL`. Checkpoints are saved every `SAVE_EVERY` steps and at the end of each epoch. Final weights land in `output/checkpoints/final/`.

---

## How to do eval

```bash
python src/eval/eval.py
```

Loads the trained model from `output/checkpoints/final` (set via `EVAL_MODEL` in `src/config.py`). Runs greedy generation over `output/test.jsonl`, scores each row, and writes results to `output/eval/`.

```bash
# Smoke test on first N rows
python src/eval/eval.py --limit 50
```

Output: `output/eval/results.jsonl` (per-row) and `output/eval/metrics.json` (aggregated). The terminal prints a summary table broken down by query type (CLadder) and method (CauSciBench).

**CLadder scoring (70 pts max):** step 1 structure (11) + step 2 query type (15) + step 3 estimand via judge (24) + step 5 answer (20)

**CauSciBench scoring (60 pts max):** step 1 variables (5) + step 2 method (5) + step 3 spec via judge (15) + step 5 numeric answer (30) + exact within 1% (5)
