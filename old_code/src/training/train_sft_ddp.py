"""
Package raw CLaDDer JSONL -> tokenized SFT examples for Qwen3-8B.
Implements the locked packager spec exactly. The only trusted numeric is
meta["groundtruth"]; reasoning.step5 / reasoning.end are never reproduced.
"""

import argparse
import json
import re
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path

from transformers import AutoTokenizer

# ── LOCKED CONSTANTS ──────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen3-8B"
MAX_SEQ_LEN  = 4096
VAL_FRACTION = 0.05
SPLIT_SALT   = "cladder_sft_v1"
EPS          = 1e-9
INCLUDE_STRUCTURAL = False
SIGNED_TYPES     = {"ate", "ett", "nde", "nie"}
STRUCTURAL_TYPES = {"backadj", "collider_bias", "exp_away"}

USER_PROMPT_TEMPLATE = '''You are given a scenario with variables, numerical data, and a yes/no question. Reason step by step, then answer.

Step 1 — Causal structure: assign variables and list the directed edges.
Step 2 — Query type: state what causal quantity the question asks for.
Step 3 — Estimand: write the expression that identifies it from the available data.
Step 4 — Substitute the available data.
Step 5 — Answer: conclude from the computed quantity.

After your reasoning, write the final answer as exactly "Yes" or "No" on its own line.

Scenario:
{given_info}

Question: {question}'''


class Reject(Exception):
    def __init__(self, reason, detail=None):
        self.reason = reason
        self.detail = detail


def _label(qt, gt, pol, ans_oracle):
    """Return the derived Yes/No label or raise Reject."""
    if qt == "ett":                                   # inverted sign vs ate/nde/nie
        if abs(gt) <= EPS:
            raise Reject("zero_effect_ambiguous")
        pos = gt > EPS; neg = gt < -EPS
        derived = "Yes" if ((pol and neg) or ((not pol) and pos)) else "No"
    elif qt in SIGNED_TYPES or qt == "correlation":   # ate/nde/nie + correlation
        if abs(gt) <= EPS:
            raise Reject("zero_effect_ambiguous")
        pos = gt > EPS
        derived = ("Yes" if pol else "No") if pos else ("No" if pol else "Yes")
    elif qt == "marginal":
        if abs(gt - 0.5) <= EPS:
            raise Reject("marginal_tie")
        more = gt > 0.5
        derived = ("Yes" if pol else "No") if more else ("No" if pol else "Yes")
    elif qt == "det-counterfactual":
        if gt not in (0, 1):
            raise Reject("det_nonbinary_gt")
        derived = "Yes" if gt == 1 else "No"
    else:
        raise Reject("unknown_query_type", qt)

    if derived != ans_oracle:
        raise Reject("label_mismatch", f"{qt} gt={gt} pol={pol} derived={derived} oracle={ans_oracle}")
    return derived


def _recompute(qt, step4):
    """Recompute the estimand value from the component probabilities listed in
    (sanitized) step4 + the per-type estimand. Raises arithmetic_recompute_conflict
    if the needed values can't be parsed/evaluated."""
    pY, joint, y, pm = {}, {}, {}, {}
    pX1 = None
    for line in step4.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        lhs, _, rhs = line.rpartition("=")
        try:
            v = float(rhs.strip())
        except ValueError:
            continue
        k = lhs.replace(" ", "")
        if (m := re.match(r"^P\(Y=1\|X=([01])\)$", k)):                pY[int(m.group(1))] = v
        elif k == "P(X=1)":                                            pX1 = v
        elif (m := re.match(r"^P\(Y=1,X=([01])\)$", k)):               joint[int(m.group(1))] = v
        elif (m := re.match(r"^P\(Y=1\|X=([01]),\w+=([01])\)$", k)):   y[(int(m.group(1)), int(m.group(2)))] = v
        elif (m := re.match(r"^P\((\w+)=1\|X=([01])\)$", k)) and m.group(1) != "Y": pm[int(m.group(2))] = v
    try:
        if qt in ("ate", "ett"):  return pY[1] - pY[0]
        if qt == "marginal":      return pY[1] * pX1 + pY[0] * (1 - pX1)
        if qt == "correlation":   return joint[1] / pX1 - joint[0] / (1 - pX1)
        if qt == "nde":           return (1 - pm[0]) * (y[(1, 0)] - y[(0, 0)]) + pm[0] * (y[(1, 1)] - y[(0, 1)])
        if qt == "nie":           return (pm[1] - pm[0]) * (y[(0, 1)] - y[(0, 0)])
    except (KeyError, TypeError, ZeroDivisionError):
        raise Reject("arithmetic_recompute_conflict", f"recompute_parse_failed:{qt}")
    raise Reject("arithmetic_recompute_conflict", f"no_recompute_formula:{qt}")


def _parse_edges(step1):
    """reasoning.step1 'X->V2,X->Y,V2->Y' -> set of directed (parent, child) edges."""
    edges = set()
    for part in step1.replace(" ", "").split(","):
        if "->" in part:
            a, b = part.split("->", 1)
            if a and b:
                edges.add((a, b))
    return edges


def _step3_justification(graph_id, qt, step1, meta):
    """Identification reasoning derived from the record's ACTUAL edges + role fields.
    Names the real variables; rejects if edges don't match the graph's topology.
    Only the mediation branch is implemented; others await confirmed records."""
    X, Y = meta["treatment"], meta["outcome"]
    edges = _parse_edges(step1)

    if graph_id == "mediation":
        # mediator M = the intermediate node on X -> M -> Y (read from the real edges)
        mids = [m for (a, m) in edges if a == X and (m, Y) in edges and m not in (X, Y)]
        if len(mids) != 1 or edges != {(X, mids[0]), (X, Y), (mids[0], Y)}:
            raise Reject("graph_edge_mismatch", f"{graph_id}: {step1}")
        M = mids[0]
        med = meta.get("mediators")                       # cross-check when populated (nde/nie)
        if med and list(med) != [M]:
            raise Reject("graph_edge_mismatch", f"{graph_id}: mediators={med} edges_M={M}")
        if qt == "nde":
            return (f"{X} affects {Y} directly ({X}->{Y}) and indirectly through {M} "
                    f"({X}->{M}->{Y}). The natural direct effect isolates the direct path, "
                    f"holding {M} at its value under no treatment. There is no confounding "
                    f"to adjust for. The estimand is:")
        if qt == "nie":
            return (f"{X} affects {Y} directly and through the mediator {M} ({X}->{M}->{Y}). "
                    f"The natural indirect effect isolates the path through {M}, via the shift "
                    f"in {M}'s distribution under treatment. No confounding to adjust for. "
                    f"The estimand is:")
        return (f"{X} influences {Y} both directly and through {M}; the total effect "
                f"combines both paths. No backdoor path exists, so no adjustment is needed. "
                f"The estimand is:")

    if graph_id == "confounding":
        raise NotImplementedError("await confounding record")
    if graph_id == "diamond":
        raise NotImplementedError("await diamond record")
    if graph_id == "fork":
        raise NotImplementedError("await fork record")
    if graph_id == "chain":
        raise NotImplementedError("await chain record")
    if graph_id == "collider":
        raise NotImplementedError("await collider record")
    raise Reject("unknown_graph_id", graph_id)


def process_one(record, tok, im_end_id, max_seq_len, val_fraction):
    """Return ('accept', example) or ('reject', reject_dict)."""
    desc_id = record.get("desc_id")
    try:
        meta = record["meta"]
        qt   = meta["query_type"]
        gt   = meta["groundtruth"]
        pol  = bool(meta.get("polarity"))
        ans_oracle = "Yes" if record["answer"] == "yes" else "No"

        if qt in STRUCTURAL_TYPES:
            raise Reject("structural_excluded")

        ANS = _label(qt, gt, pol, ans_oracle)

        # ── body assembly (steps 0-4 echoed; step4 sanitized; step5 synthesized) ──
        r = record["reasoning"]
        step4 = r["step4"].replace("=1=1", "=1").replace("X=0=1", "X=0").replace("X=1=1", "X=1")
        step4 = re.sub(r"=([01])=1", r"=\1", step4)   # generic VAR=0=1 / VAR=1=1 double collapse
        if re.search(r"=\d=1", step4):
            raise Reject("step4_artifact_unhandled", step4)

        # ── synthesize step5: value recomputed from listed probabilities (not gt echo),
        #    gated to agree in sign with the trusted meta.groundtruth ──
        if qt == "det-counterfactual":
            step5 = (f"Step 5 — Answer: Resolving the structural equations under the stated action, "
                     f"the proposition {meta.get('formal_form', '').strip()} is "
                     f"{'true' if gt==1 else 'false'}. Therefore, the answer is: {ANS}")
        else:
            V = _recompute(qt, step4)
            if qt == "marginal":
                if (V > 0.5) != (gt > 0.5):
                    raise Reject("arithmetic_recompute_conflict", f"V={V:.4f} gt={gt:.4f}")
                rel = "greater than 0.5 (more likely than not)" if gt > 0.5 else "less than 0.5 (less likely than not)"
                step5 = f"Step 5 — Answer: P(Y=1) = {V:.2f}, which is {rel}. Therefore, the answer is: {ANS}"
            else:
                if (V > 0) != (gt > 0):
                    raise Reject("arithmetic_recompute_conflict", f"V={V:.4f} gt={gt:.4f}")
                verb = "positive" if gt > EPS else "negative"
                step5 = (f"Step 5 — Answer: The estimand evaluates to {V:.2f}, a {verb} value. "
                         f"{'This matches' if ANS=='Yes' else 'This does not match'} what the question asks. "
                         f"Therefore, the answer is: {ANS}")

        just = _step3_justification(meta.get("graph_id"), qt, r.get("step1", ""), meta)

        body  = f"Step 1 — Causal structure: {r['step0']} {r['step1']}\n"
        body += f"Step 2 — Query type: {qt} — formal form: {r['step2']}\n"
        body += f"Step 3 — Estimand: {just} {r['step3']}\n"
        body += f"Step 4 — Substitute the available data:\n{step4}\n"
        body += step5 + "\n"
        body += f"\n{ANS}"

        # ── user prompt + /no_think ──
        user_content = USER_PROMPT_TEMPLATE.format(given_info=record["given_info"],
                                                   question=record["question"]) + "\n/no_think"

        # ── tokenize + mask ──
        prompt_ids = tok.apply_chat_template([{"role": "user", "content": user_content}],
                                             add_generation_prompt=True, enable_thinking=False, tokenize=True)
        if not isinstance(prompt_ids, list):      # transformers>=5 returns BatchEncoding
            prompt_ids = prompt_ids["input_ids"]
        has_think = "<think>" in tok.decode(prompt_ids)
        completion_text = ("" if has_think else "<think>\n\n</think>\n\n") + body
        completion_ids  = tok(completion_text, add_special_tokens=False)["input_ids"] + [im_end_id]
        input_ids = prompt_ids + completion_ids
        labels    = [-100] * len(prompt_ids) + completion_ids
        attention_mask = [1] * len(input_ids)
        if len(input_ids) > max_seq_len:
            raise Reject("length_exceeded", len(input_ids))

        # ── per-example assertions ──
        full = tok.decode(input_ids)
        if full.count("<think>") != 1 or full.count("</think>") != 1:
            raise Reject("assertion_failed", "think_count")
        between = full.split("<think>", 1)[1].split("</think>", 1)[0]
        if not re.match(r"^\s*$", between):
            raise Reject("assertion_failed", "think_nonempty")
        first_label = next(i for i, l in enumerate(labels) if l != -100)
        if first_label != len(prompt_ids):
            raise Reject("assertion_failed", "label_start")
        if input_ids[-1] != im_end_id:
            raise Reject("assertion_failed", "im_end")
        final_line = [ln for ln in body.split("\n") if ln.strip()][-1]
        if final_line != ANS:
            raise Reject("assertion_failed", "final_line")

        # ── split by story_id ──
        h = int(sha256((SPLIT_SALT + meta["story_id"]).encode()).hexdigest(), 16) % 10000
        split = "val" if h / 10000.0 < val_fraction else "train"

        example = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "meta": {
                "desc_id": desc_id,
                "graph_id": meta.get("graph_id"),
                "query_type": qt,
                "rung": meta.get("rung"),
                "story_id": meta["story_id"],
                "split": split,
            },
        }
        return "accept", example
    except Reject as e:
        return "reject", {"desc_id": desc_id, "reason": e.reason, "detail": e.detail}


def _read(paths):
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_name", default=MODEL_NAME)
    ap.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--val_fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--include_structural", action="store_true")
    args = ap.parse_args()

    if args.include_structural or INCLUDE_STRUCTURAL:
        sys.exit("structural-type packaging: not implemented")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    assert im_end_id and im_end_id >= 0

    total_in = total_train = total_val = total_rejected = 0
    rejects_by_reason = Counter()
    counts_by_query_type = Counter()
    length_hist = Counter()   # 0..15 = 256-buckets to 4096, 16 = overflow

    f_train = open(out_dir / "train.jsonl", "w")
    f_val   = open(out_dir / "val.jsonl", "w")
    f_rej   = open(out_dir / "rejects.jsonl", "w")

    for record in _read(args.input):
        total_in += 1
        status, payload = process_one(record, tok, im_end_id, args.max_seq_len, args.val_fraction)
        if status == "reject":
            total_rejected += 1
            rejects_by_reason[payload["reason"]] += 1
            f_rej.write(json.dumps(payload) + "\n")
            continue
        qt = payload["meta"]["query_type"]
        counts_by_query_type[qt] += 1
        n = len(payload["input_ids"])
        length_hist[min(n // 256, 16)] += 1
        if payload["meta"]["split"] == "val":
            total_val += 1
            f_val.write(json.dumps(payload) + "\n")
        else:
            total_train += 1
            f_train.write(json.dumps(payload) + "\n")

    f_train.close(); f_val.close(); f_rej.close()

    stats = {
        "total_in": total_in,
        "total_train": total_train,
        "total_val": total_val,
        "total_rejected": total_rejected,
        "rejects_by_reason": dict(rejects_by_reason),
        "counts_by_query_type": dict(counts_by_query_type),
        "length_histogram": {("overflow" if k == 16 else f"{k*256}-{k*256+255}"): length_hist[k]
                             for k in sorted(length_hist)},
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
