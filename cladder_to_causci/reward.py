"""reward.py — veRL RLVR reward for CLadder rung-2, six-step gated cascade.

Ordered gates (each step presupposes the one before; reward accrues only as far as the
model stays correct — NOT a weighted sum):

    graph  g ─gate─▶  query-type q ─gate─▶  estimand e ─gate─▶  arithmetic c   (answer a: logged, weight 0)

    g : edge-set F1, predicted vs gold DAG                              [0,1], HARD gate
        (wrong/missing edge = the scenario was misunderstood → reward floor)
    q : query-type exact match                                         {0,1}, gate
    e : DoVerifier equivalence of the model's IDENTIFIED expression vs the gold
        interventional target, under the model's OWN DAG               {0,1}  ← load-bearing
        (string-match fallback only when the verifier can't run; logged)
    c : arithmetic checksum — final number vs gold value, light        {0,1}
    a : yes/no match — measured for the diagnostic table, NOT rewarded in RL

    score = W_G*g + [g>=G_GATE]( W_Q*q + [q=1]( W_E*e + [e=1]*W_C*c ) )
    reward = 2*score - 1                                               [-1,1]

veRL entry: reward_fn(solution_strs, ground_truths, extra_infos) -> list[float].
ground_truth is a JSON string {graph, query_type, formal, identify, value, answer}.
"""

import atexit
import collections
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schema import parse, n_blocks                          # noqa: E402
from verify import expr_equiv, is_estimable                 # noqa: E402
from causci_eval import score_causci, compute_causci_metrics  # noqa: E402  (CauSci val scoring)

# ── credits (accumulated, NOT a weighted sum; reward ∈ [0,1], floor 0 on gate fail) ──
W_G, W_Q, W_E, W_C = 0.20, 0.15, 0.55, 0.10   # graph, query, expression (load-bearing), arithmetic
W_A = 0.0                                      # final answer ≈ 0 by design (don't reward the bit)
CALC_TOL = 0.03
FALLBACK_QT = {"collider_bias", "det-counterfactual"}   # prose/structural derivations → value-match
REQUIRED = ["graph", "query_type", "derivation"]   # format gate: load-bearing blocks only
#   (arithmetic is a trailing checksum — a missing <arithmetic> forfeits its small credit, not all)

_call_count = [0]
_eval_buffer = []
_eval_pass = [0]
_causci_buffer = []                        # CauSci val items (parsed, comp) → [causci_eval] line
_causci_pass = [0]
_causci_samples = []                       # sampled CauSci responses, dumped per eval pass
_verify_stats = collections.Counter()
_eval_samples = []                         # sampled test responses, dumped per eval pass for debugging
SAMPLE_N = 40
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "samples")


def _zero_comp(gt, status="format"):
    return {"graph": 0.0, "query": 0.0, "estimand": 0.0, "calc": 0.0, "answer": 0.0,
            "status": status, "query_type": gt.get("query_type", ""), "rung": gt.get("rung")}


# ── component scores ──────────────────────────────────────────────────────

def _edge_set(graph_str: str):
    return {tuple(t.strip() for t in e.split("->", 1))
            for e in re.split(r"[,;\n]", str(graph_str).replace("→", "->")) if "->" in e}


def graph_f1(pred_graph: str, gold_graph: str) -> float:
    p, g = _edge_set(pred_graph), _edge_set(gold_graph)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def _norm_expr(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


_NUM = re.compile(r"[-+]?\d*\.?\d+")


def estimand_score(pred: dict, gt: dict):
    """(score in {0,1}, status). Step 3 — the load-bearing identification check, rung-agnostic.

    Probability-expression cells (marginal/correlation/exp_away/ate/ett/nie/nde): the model's
    <derivation> must be do-free (else `has_do`) and FUNCTIONALLY equivalent to the gold estimable
    form across random parameterizations of the graph (numeric CBN certificate).

    Prose/structural cells (collider_bias, det-counterfactual) have no probability-expression
    derivation, so expression credit falls back to matching the gold numeric value.
    """
    qt = str(gt.get("query_type", "")).lower()
    if qt in FALLBACK_QT:
        ok = calc_score(pred["arithmetic"], gt.get("value")) >= 1.0
        return (1.0 if ok else 0.0), ("verified_value" if ok else "refuted_value")
    model = pred["derivation"] or pred["estimand"]
    if not is_estimable(model):
        return 0.0, "has_do"                       # do-free gate: never completed identification
    if expr_equiv(model, gt.get("derivation", ""), gt["graph"]):
        return 1.0, "verified"
    return 0.0, "refuted"


def calc_score(calc_str: str, gold_value) -> float:
    """Checksum: last number in <calc> within CALC_TOL of the gold numeric effect."""
    if gold_value in (None, ""):
        return 0.0
    nums = _NUM.findall(str(calc_str))
    if not nums:
        return 0.0
    try:
        return 1.0 if abs(float(nums[-1]) - float(gold_value)) <= CALC_TOL else 0.0
    except ValueError:
        return 0.0


def grade(pred: dict, gt: dict) -> dict:
    """GRADER (SFT eval + RL diagnostics; NEVER a gradient): continuous per-segment scores.
    This is measurement only — the two code paths stay separate: SFT touches `grade`, the RL
    gradient comes from `score_one`."""
    e, status = estimand_score(pred, gt)
    return {
        "graph": graph_f1(pred["graph"], gt["graph"]),
        "query": 1.0 if pred["query_type"] and pred["query_type"].strip().lower()
                 == str(gt.get("query_type", "")).lower() else 0.0,
        "estimand": e, "status": status,
        "calc": calc_score(pred["arithmetic"], gt.get("value")),
        "answer": 1.0 if pred["answer"] and pred["answer"] == gt.get("answer") else 0.0,
        "query_type": gt.get("query_type", ""), "rung": gt.get("rung"),
    }


def score_one(pred: dict, gt: dict):
    """REWARD (RL only): the gated cascade — credit accrues only as far up as the model stays
    correct, then stops. reward ∈ [0,1]. Returns (reward, components).
      graph (HARD gate) → query (gate) → expression (gate) → arithmetic (checksum) → answer (≈0).
    """
    comp = grade(pred, gt)
    _verify_stats[comp["status"]] += 1
    g, q, e, c, a = comp["graph"], comp["query"], comp["estimand"], comp["calc"], comp["answer"]
    r = 0.0
    if g >= 1.0:                       # graph hard gate: wrong graph = wrong understanding → 0
        r += W_G
        if q >= 1.0:                   # query-type gate
            r += W_Q
            if e >= 1.0:               # expression gate (the transfer-relevant identification)
                r += W_E
                r += W_C * c           # arithmetic checksum (not a gate)
                r += W_A * a           # final answer (≈0 by design)
    return r, comp


# ── diagnostic metrics (continuous; separate from the strict gated reward) ──

_STEPS = ("graph", "query", "estimand", "calc", "answer")


def _block(comps, denom):
    """Per-step means + full-correct over a list of component dicts, normalized by `denom`."""
    d = denom or 1
    m = {s: sum(c[s] for c in comps) / d for s in _STEPS}
    m["full"] = sum(c["graph"] >= 1 and c["query"] >= 1 and c["estimand"] >= 1 for c in comps) / d
    return m


def compute_eval_metrics(items):
    """Per-step CLadder diagnostic from (parsed, components) items — overall, per-rung, per-query.
    comp is None only on a hard exception; format-gate fails carry a zero comp (so they count)."""
    comps = [c for _, c in items if c is not None]
    n = len(items) or 1
    ov = _block(comps, n)
    m = {"graph_f1": ov["graph"], "query_acc": ov["query"], "estimand_acc": ov["estimand"],
         "calc_acc": ov["calc"], "answer_acc": ov["answer"], "full_correct": ov["full"],
         "verified_rate": sum(c["status"].startswith("verified") for c in comps) / n,
         "parse_fail_rate": (len(items) - len(comps)
                             + sum(c["status"] == "format" for c in comps)) / n,
         "n": len(items)}
    by_rung = collections.defaultdict(list)
    by_qt = collections.defaultdict(list)
    for c in comps:
        by_rung[c.get("rung")].append(c)
        by_qt[c.get("query_type") or "unknown"].append(c)
    for r, cs in by_rung.items():
        b = _block(cs, len(cs))
        m[f"r{r}/answer_acc"] = b["answer"]; m[f"r{r}/graph_f1"] = b["graph"]
        m[f"r{r}/estimand_acc"] = b["estimand"]; m[f"r{r}/full"] = b["full"]
    for qt, cs in by_qt.items():
        m[f"acc/{qt}"] = sum(c["answer"] for c in cs) / len(cs)
    return m


# ── eval flush (CLadder held-out + CauSci transfer, both during RL validation) ──

def _flush_eval():
    if not _eval_buffer:
        return
    _eval_pass[0] += 1
    m = compute_eval_metrics(_eval_buffer)
    print(f"[verl_eval] eval_pass:{_eval_pass[0]} "
          + " ".join(f"{k}:{v:.4f}" for k, v in sorted(m.items())), flush=True)
    if _eval_samples:                                       # dump sampled responses for review
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        with open(os.path.join(SAMPLE_DIR, f"rl_eval_pass{_eval_pass[0]}.jsonl"), "w") as f:
            for s in _eval_samples:
                f.write(json.dumps(s) + "\n")
        _eval_samples.clear()
    _eval_buffer.clear()


def _flush_causci():
    if not _causci_buffer:
        return
    _causci_pass[0] += 1
    m = compute_causci_metrics(_causci_buffer)
    print(f"[causci_eval] eval_pass:{_causci_pass[0]} "
          + " ".join(f"{k}:{v:.4f}" for k, v in sorted(m.items()) if isinstance(v, float)), flush=True)
    if _causci_samples:
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        with open(os.path.join(SAMPLE_DIR, f"causci_rl_pass{_causci_pass[0]}.jsonl"), "w") as f:
            for s in _causci_samples:
                f.write(json.dumps(s) + "\n")
        _causci_samples.clear()
    _causci_buffer.clear()


atexit.register(_flush_eval)
atexit.register(_flush_causci)


# ── veRL interface ────────────────────────────────────────────────────────

def reward_fn(solution_strs, ground_truths, extra_infos):
    _call_count[0] += 1
    call = _call_count[0]
    t0 = time.time()

    rewards = []
    split_tag = "train"
    for sol, gt_str, ei in zip(solution_strs, ground_truths, extra_infos):
        ei = json.loads(ei) if isinstance(ei, str) else (ei or {})
        split_tag = ei.get("split", "train")
        try:
            gt = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
            if split_tag == "causci":                       # CauSci transfer val (PO spec, no format gate)
                reward, comp = score_causci(sol, ei.get("columns") or [], gt, ei.get("csplit"))
                _causci_buffer.append((None, comp))
                if len(_causci_samples) < SAMPLE_N:
                    _causci_samples.append({"id": ei.get("id", ""), "split": ei.get("csplit"),
                                            "comp": comp, "completion": sol[:2500]})
            else:                                           # CLadder train or held-out val
                parsed = parse(sol)
                if not all(parsed.get(b) for b in REQUIRED):
                    reward, comp = 0.0, _zero_comp(gt)      # format gate → 0 (comp keeps rung so it counts)
                else:
                    reward, comp = score_one(parsed, gt)
                if split_tag == "test":
                    _eval_buffer.append((parsed, comp))
                    if len(_eval_samples) < SAMPLE_N:
                        _eval_samples.append({"id": ei.get("id", ""), "reward": round(reward, 3),
                                              "comp": comp, "completion": sol[:3000]})
        except Exception as ex:
            print(f"[reward] item error → 0.0: {type(ex).__name__}: {ex}", flush=True)
            reward = 0.0
        rewards.append(reward)

    mean_r = sum(rewards) / len(rewards) if rewards else 0.0
    if split_tag == "train":                                # a train batch → flush any buffered val
        _flush_eval()
        _flush_causci()
        vs = _verify_stats
        print(f"[verl] call {call:5d} reward={mean_r:+.3f} "
              f"verify(ok/refuted/has_do)={vs['verified']+vs['verified_value']}/"
              f"{vs['refuted']+vs['refuted_value']}/{vs['has_do']}", flush=True)
    elif split_tag == "test":
        print(f"[verl_eval] eval_pass:{_eval_pass[0]} call:{call:5d} reward={mean_r:+.3f}", flush=True)
    elif split_tag == "causci":
        print(f"[causci_eval] pass:{_causci_pass[0]} call:{call:5d} reward={mean_r:+.3f}", flush=True)
    print(f"[reward] n={len(rewards)} dt={time.time()-t0:.2f}s", flush=True)
    return rewards


# ── batch reward manager (score whole batch in one call) ──────────────────

def _install_batch_reward_manager():
    try:
        import torch
        import verl.workers.reward_manager.naive as _naive

        def _batch_call(self_rm, data, return_dict=False):
            responses = data.batch["responses"]
            n = responses.shape[0]
            if "response_length" in data.batch:
                lens = [int(data.batch["response_length"][i]) for i in range(n)]
                sols = [self_rm.tokenizer.decode(responses[i, :lens[i]], skip_special_tokens=True)
                        for i in range(n)]
            else:
                pad = self_rm.tokenizer.pad_token_id
                sols, lens = [], []
                for i in range(n):
                    valid = responses[i][responses[i] != pad]
                    lens.append(len(valid))
                    sols.append(self_rm.tokenizer.decode(valid, skip_special_tokens=True))

            rm = data.non_tensor_batch["reward_model"]
            gts = [rm[i]["ground_truth"] for i in range(n)]
            ei = data.non_tensor_batch.get("extra_info")
            eis = [ei[i] for i in range(n)] if ei is not None else [{} for _ in range(n)]

            scores = reward_fn(sols, gts, eis)
            rt = torch.zeros_like(responses, dtype=torch.float32)
            for i, (sc, ln) in enumerate(zip(scores, lens)):
                if ln > 0:
                    rt[i, ln - 1] = float(sc)
            return {"reward_tensor": rt} if return_dict else rt

        _naive.NaiveRewardManager.__call__ = _batch_call
        print("[reward] BatchRewardManager installed", flush=True)
    except Exception as e:
        print(f"[reward] BatchRewardManager install skipped ({e})", flush=True)


_install_batch_reward_manager()


def compute_score(data_source, solution_str, ground_truth, extra_info) -> float:
    return reward_fn([solution_str], [ground_truth], [extra_info])[0]


# ── sanity check (no GPU): gated reward on real records ─────────────────────
# Run:  conda run -n alignment python3 reward.py

def _selftest():
    from pathlib import Path
    OUT = Path(__file__).resolve().parent / "output"
    rows = [json.loads(l) for l in open(OUT / "train_rl.jsonl")]
    ate = next(r for r in rows if r["groundtruth"]["query_type"] == "ate")
    col = next(r for r in rows if r["groundtruth"]["query_type"] == "collider_bias")
    g, gp = ate["groundtruth"], parse(ate["target"])

    def blk(**kw):
        return "\n".join(f"<{t}>{kw[t]}</{t}>" for t in ["mapping", "query_type", "graph", "estimand",
                                                          "data", "derivation", "arithmetic", "answer"] if t in kw)

    p = parse(ate["target"])
    assert n_blocks(p) == 7, f"gold target parsed {n_blocks(p)} blocks: {p}"
    print("parser round-trip OK — gold target → 7 blocks\n")

    scenarios = [
        ("perfect (gold target)",       ate, ate["target"]),
        ("identified obs form",         ate, blk(mapping=gp["mapping"], query_type=g["query_type"], graph=g["graph"],
                                                 estimand="E[Y|do(X=1)]-E[Y|do(X=0)]", data=gp["data"],
                                                 derivation="P(Y=1|X=1) - P(Y=1|X=0)", arithmetic=gp["arithmetic"],
                                                 answer=g["answer"])),
        ("wrong query type",            ate, blk(query_type="marginal", graph=g["graph"],
                                                 derivation=g["derivation"], answer=g["answer"])),
        ("wrong estimand (dropped do)", ate, blk(query_type=g["query_type"], graph=g["graph"],
                                                 derivation="P(Y)", answer=g["answer"])),
        ("wrong graph, right rest",     ate, blk(query_type=g["query_type"], graph="A->B",
                                                 derivation=g["derivation"], answer=g["answer"])),
        ("garbage",                     ate, "I think the answer is yes."),
        ("collider perfect",            col, col["target"]),
    ]
    for name, row, sol in scenarios:
        r = reward_fn([sol], [json.dumps(row["groundtruth"])], [json.dumps({"split": "test"})])[0]
        print(f"reward={r:+.3f}  {name}")


if __name__ == "__main__":
    _selftest()
