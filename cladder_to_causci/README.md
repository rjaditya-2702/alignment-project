# cladder_to_causci

RLVR (RL with **verifiable** rewards) for CLadder **rung-2** causal reasoning, using the
DoVerifier symbolic verifier (He et al., EACL 2026) as the reward signal — building toward
transfer to CauSciBench. The model emits **six reasoning steps as tagged blocks** (regex-parsed,
not whole-response JSON — JSON hurts free-running reasoning during RL).

## Six-step tagged schema — Variant B (`prompts.py`)
Model-generated mapping (O2=B): the model assigns canonical symbols (X=treatment, Y=outcome,
V1,V2,…=rest) from the raw description, so it learns the CauSci-relevant variable-identification
skill. Expressions use Do-Verifier surface syntax. CLadder's `reasoning` field maps 1:1 onto these
blocks and is the process supervision (rendered into the SFT target by `format_target`).
```
<mapping>    symbol = meaning        X = pexu; Y = rukz; V1 = hwax
<query_type> query type              ate | collider_bias
<graph>      directed edges          V1->X,V1->Y,X->Y
<estimand>   symbolic do-form        E[Y | do(X = 1)] - E[Y | do(X = 0)]
<data>       probabilities           P(Y=1 | V1=0, X=0) = 0.96   (one per line)
<derivation> estimable/identified    Σ_{V1} P(V1)[P(Y=1|V1,X=1) - P(Y=1|V1,X=0)]
<arithmetic> substitute & compute    0.18*(0.85-0.96) + 0.82*(0.37-0.72) = -0.31
<answer>     Yes / No                (de-emphasized: a consequence, not a target)
```

**Background synthesis:** CLadder's vendored dataset ships no edge-description prose, so
`verbalize_background` reconstructs CLadder's canonical background ("A has a direct effect on
B and C.") from the gold graph + mapping in real-world terms — this is what forces the model to
do the mapping/graph-extraction work rather than being handed it.

## Gated cascade reward (`reward.py`, RL only) — ordered gates, accumulated credit
```
Step 0 parse (format gate) → 0 if required blocks missing
graph g ─HARD gate─▶ query q ─gate─▶ expression e ─gate─▶ arithmetic c (checksum)   answer (≈0)
reward = W_G + [q]·W_Q + [q][e]·(W_E + W_C·c) + ... ,  accumulated ∈ [0,1], stops at first failed gate
W_G,W_Q,W_E,W_C = 0.20,0.15,0.55,0.10   (expression load-bearing);  W_A = 0 (don't reward the bit)
```
Credit accrues only as far up as the model stays correct, then stops. A wrong graph zeroes
everything (the graph *is* the model's understanding). This is the RL gradient only — SFT never
sees it (see the grader/reward split below).

## Expression verification — the common numeric verifier (`verify.py`)
The rung-agnostic core, **replacing the symbolic DoVerifier**. Step 3 checks the model's do-free
`<derivation>` is FUNCTIONALLY equivalent to the gold estimable form: build N random binary
parameterizations (CBNs) of the item's graph, evaluate both expressions observationally on each,
require agreement on all N. A correct identification equals the gold on *every* parameterization; a
merely-lucky expression breaks on a re-draw. Works identically for rungs 1/2/3 (the gold estimable
form is do-free), so it subsumes backdoor/frontdoor/IV/mediation without special-casing.
- Do-free gate: residual `do(` or a counterfactual subscript `Y_{...}` ⇒ identification not completed ⇒ stop.
- `collider_bias` / `det-counterfactual` have prose (non-expression) derivations → expression credit
  falls back to matching the gold numeric value.

**Mapping gating (deferred):** the `<mapping>` block is trained (SFT) but not yet scored
explicitly. Because `<graph>` is edge-F1'd in canonical symbols against the gold DAG, the model is
already forced to adopt CLadder's canonical symbol assignment to pass the graph gate — so mapping
correctness is implicitly enforced. An explicit mapping-alignment scorer (O2's stated cost) can be
added later if free-running RL shows symbol drift.

## Data scope & balancing (all 3 rungs, real + synth)
`data.py` builds from two pools — **real** (`train.jsonl`+`test.jsonl`, mixed stories) and **synth**
(`cladder_synth_raw.jsonl`, nonsensical) — across **all 3 rungs** (9 (rung, query_type) cells):
- **SFT = 8K real + 2K synth** (real-anchored so natural-language variation dominates),
  **RL = 2K real + 8K synth** (synth-heavy ceiling), **eval = real held-out**.
- Each portion balanced: ~equal per (rung,query_type) cell, balancing graph + story-type within cell.
- Splits **story-disjoint** (partition over the union; synth stories ⊂ real's). Graphs
  balanced-not-disjoint (query types are coupled to graphs, e.g. `collider_bias` ⟺ `collision`).
- Verifiability filter (all rungs): keep records whose gold estimable form is numerically evaluable,
  or a value-fallback cell — the numeric verifier now **admits IV and frontdoor** (they were dropped
  under DoVerifier). Shortfalls logged, never silently capped.

## Files
- `doverifier/` — vendored DoVerifier (unmodified; retained but no longer used by the reward)
- `schema.py` — Variant B prompts + `verbalize_background` + `format_target` + rollout `parse`
- `verify.py` — `adjustment_verified` (ate backdoor) / `expr_equiv` (collider structural)
- `reward.py` — six-step gated cascade + `compute_eval_metrics` + veRL batch manager + `compute_score`;
  `python3 reward.py` runs the no-GPU sanity check
- `data.py` — one pass → `output/{train,test}_rl.jsonl` (verifiability-filtered) + veRL parquet + SFT jsonl
- `train_sft.py` — LoRA SFT (Qwen3-8B, DDP); periodic single-pass eval scored by the RL reward
- `causci_eval.py` — CauSciBench transfer eval (`--validate` = no-GPU oracle/transfer-ceiling)
- `plot.py` — parses the veRL log → training + eval dashboard
- `run_sft.sh` — SFT Phase A → Phase B1 (warm-started); `run_rl.sh` — veRL GRPO (no judge server)

## Training sequence
SFT Phase A (turn-by-turn, teacher-forced) → Phase B1 (single-pass, warm-started; collapses the
scaffold to the deployable one-rollout shape) → **handoff gate** → RL (GRPO, free-running
single-pass, gated reward) → CauSci transfer eval. The turn-by-turn scaffold is a training
convenience only — Phase B1 is non-optional or the train/test single-pass mismatch kills transfer.

**SFT has no scoring** — pure token cross-entropy against the gold 8-block target, loss masked on
the prompt (and, in turn-by-turn, on the gold prefix — only the target step gets loss). Two strictly
separate instruments: the **grader** (`reward.grade`, continuous per-segment) drives the diagnostic
table and never the gradient; the **reward** (`reward.score_one`, gated cascade) drives only the RL
gradient. **Handoff gate:** RL does not start until the held-out **graph-extraction F1 ≥ 0.95**
(`train_sft.py --grade` writes `output/sft/grade.json`; `run_rl.sh` refuses to start below it) —
if graph extraction isn't near-solved, the RL graph gate starves the policy.

## CauSci transfer eval (`causci_eval.py`, TEST ONLY — never train on CauSci)
Maps the CLadder six-step output → CauSci answer: treatment=`mapping[X]`, outcome=`mapping[Y]`,
controls=adjustment set from `<derivation>`, method inferred (backdoor→`ols` bucket; collider→none).
Headline = **Method Correctness + Variable Selection** (expected transfer); Effect Accuracy is a
best-effort plan-execution measure (OLS on the CSV), de-emphasized since CLadder trains no
implementation. `--validate` (no GPU) scores an oracle backdoor reasoner = the transfer ceiling
(≈21% method on real — only `ols` golds; iv/did/rdd/frontdoor are out of a backdoor reasoner's reach).

## Run
```
conda run -n alignment python3 data.py              # build rl jsonl + parquet + sft jsonl
conda run -n alignment python3 reward.py            # sanity check (no GPU)
conda run -n alignment python3 causci_eval.py --validate   # transfer ceiling (no GPU)
sbatch run_sft.sh                                   # SFT A→B1 (cluster) → output/sft/final
sbatch run_rl.sh                                    # RL GRPO, warm-started from SFT (cluster)
python3 causci_eval.py --model output/sft/final     # transfer eval (GPU)
```
