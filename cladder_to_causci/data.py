"""data.py — build all CLadder training data, balanced across all 3 rungs, in one pass.

STRATIFICATION (priority: rung → query_type → graph → story_type). Cells = (rung, query_type);
each cell gets ~equal quota per split, and within a cell we balance the ~10 graph structures
(CGTEs) and secondarily the story type (nonsensical vs sensible — CLadder showed nonsensical
stories most test real reasoning, so they're kept, just not allowed to dominate).

SPLITS = SFT / RL / EVAL, story-DISJOINT (each story_id lands in exactly one split, so eval
surface forms are never trained on). Graphs are NOT made disjoint: query types are coupled to
graphs (e.g. collider_bias ⟺ collision graph), so disjoint graphs would drop whole query-type
cells from a split, breaking the within-stratum holdout. Both nonsensical and sensible stories
are present in every split. Sampling to quota gives equal coverage at build time (no post-hoc
resampling); shortfalls (a cell/graph with too few records) are logged, never silently capped.

VERIFIABILITY FILTER is rung-2 only (that's all the symbolic reward covers today): drop rung-2
records whose gold identification the verifier can't confirm (IV, frontdoor). Rung-1/3 pass
through unfiltered — their reward/verification is not built yet, so RL should not train on them
until it is; SFT (teacher-forced) is fine on all rungs now.

Outputs (output/):
  sft.jsonl, train_rl.jsonl, test_rl.jsonl   — the three story-disjoint splits (six-step rows)
  train_rl.parquet, test.parquet             — veRL (RL train + eval)
  sft_single.jsonl, sft_turns.jsonl, sft_test.jsonl  — SFT Phase B1 / Phase A / periodic-eval

Run:  conda run -n alignment python3 data.py
"""

import collections
import json
import random
import re
from pathlib import Path

import pandas as pd

from schema import (CLADDER_SYSTEM, CLADDER_USER, CAUSCI_SYSTEM, format_target, parse,
                    parse_mapping, verbalize_background)
from reward import FALLBACK_QT
from verify import evaluable
from causci_eval import build_user as causci_user, csv_columns, load_bench

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "old_code" / "dataset"
OUT = ROOT / "output"
SEED = 0

REAL_SRC = ["train.jsonl", "test.jsonl"]          # official CLadder (mixed stories)
SYNTH_SRC = ["cladder_synth_raw.jsonl"]           # generator-minted (nonsensical stories)
# composition (real-anchored SFT so natural-language variation dominates; synth-heavy RL ceiling)
SFT_REAL, SFT_SYNTH = 8000, 2000                  # SFT = 10K
RL_REAL, RL_SYNTH = 2000, 8000                    # RL  = 10K
EVAL_N = 1080                                     # held-out CLadder diagnostic (real only)
STORY_FRAC = {"sft": 0.55, "rl": 0.30, "eval": 0.15}     # story-disjoint partition
SPLIT_FILE = {"sft": "sft.jsonl", "rl": "train_rl.jsonl", "eval": "test_rl.jsonl"}

_NUM = re.compile(r"[-+]?\d*\.?\d+")


# ── row building ────────────────────────────────────────────────────────────

def _final_value(step5):
    """Gold numeric effect = last number in step5. None if absent."""
    nums = _NUM.findall(str(step5))
    return float(nums[-1]) if nums else None


def _row(r):
    """CLadder record → six-step row, or None if reasoning is incomplete (drops backadj)."""
    rs = r.get("reasoning")
    if not isinstance(rs, dict) or not all(rs.get(k) for k in ("step0", "step1", "step2", "step3")):
        return None
    meta = r.get("meta", {})
    mapping = parse_mapping(rs.get("step0", ""))
    return {
        "id": r.get("desc_id", ""),
        "source": "cladder",
        "prompt": CLADDER_USER.format(background=verbalize_background(rs.get("step1", ""), mapping),
                                      unobserved_clause="",
                                      given_info=r.get("given_info", ""),
                                      question=r.get("question", "")),
        "target": format_target(rs, meta, r.get("answer", "")),
        "groundtruth": {
            "rung": meta.get("rung"),
            "graph": rs.get("step1"),
            "query_type": meta.get("query_type", ""),
            "formal": meta.get("formal_form") or rs.get("step2"),
            "derivation": rs.get("step3"),
            "value": _final_value(rs.get("step5")),
            "answer": str(r.get("answer", "")).strip().lower(),
        },
    }


def _entry(r):
    """Balancing dims for a complete record (row built lazily for selected only), else None."""
    rs = r.get("reasoning")
    if not isinstance(rs, dict) or not all(rs.get(k) for k in ("step0", "step1", "step2", "step3")):
        return None
    m = r.get("meta", {})
    sid = m.get("story_id", "")
    return {"r": r, "rung": m.get("rung"), "qt": m.get("query_type"), "graph": m.get("graph_id"),
            "story": sid, "stype": "nonsense" if str(sid).startswith("nonsense") else "sensible",
            "deriv": rs.get("step3"), "edges": rs.get("step1")}


# ── balanced sampling ───────────────────────────────────────────────────────

def _interleave(items, key, rng):
    """Round-robin across groups keyed by `key(item)` (balances that dimension)."""
    groups = collections.defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    for g in groups.values():
        rng.shuffle(g)
    out, ks = [], sorted(groups)
    while any(groups[k] for k in ks):
        for k in ks:
            if groups[k]:
                out.append(groups[k].pop())
    return out


def _order_cell(entries, rng):
    """Order a (rung,qt) cell: story-type interleave within each graph, then round-robin graphs."""
    by_graph = collections.defaultdict(list)
    for e in entries:
        by_graph[e["graph"]].append(e)
    for g in by_graph:
        by_graph[g] = _interleave(by_graph[g], lambda e: e["stype"], rng)
    out, gs = [], sorted(by_graph)
    while any(by_graph[g] for g in gs):
        for g in gs:
            if by_graph[g]:
                out.append(by_graph[g].pop(0))
    return out


def _split_stories(story_ids, rng):
    """Assign each story to sft/rl/eval, splitting nonsensical and sensible separately so both
    appear in every split."""
    assign = {}
    for bucket in (sorted(s for s in story_ids if str(s).startswith("nonsense")),
                   sorted(s for s in story_ids if not str(s).startswith("nonsense"))):
        rng.shuffle(bucket)
        n = len(bucket)
        n_sft = round(n * STORY_FRAC["sft"])
        n_rl = round(n * STORY_FRAC["rl"])
        for i, s in enumerate(bucket):
            assign[s] = "sft" if i < n_sft else "rl" if i < n_sft + n_rl else "eval"
    return assign


# ── build ───────────────────────────────────────────────────────────────────

def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _load_pool(files):
    return [e for f in files for e in map(_entry, map(json.loads, open(SRC / f))) if e]


def _ok(e):
    """Gold estimable form is numerically checkable by the reward (verified on demand)."""
    return e["qt"] in FALLBACK_QT or evaluable(e["deriv"], e["edges"])


def _sample_portion(entries, total, rng, label, seen):
    """Stratify `total` across (rung, query_type) cells (~equal), balancing graph + story-type
    within each cell, keeping only reward-verifiable records. `seen` = desc_ids already used
    anywhere (DEDUP: real/synth overlap ~90K by desc_id, so skip repeats). Shortfalls logged."""
    cells = collections.defaultdict(list)
    for e in entries:
        cells[(e["rung"], e["qt"])].append(e)
    if not cells:
        return []
    per = max(1, total // len(cells))
    selected, short = [], []
    for cell in sorted(cells, key=lambda c: (c[0] or 0, str(c[1]))):
        picked = []
        for e in _order_cell(cells[cell], rng):
            eid = e["r"].get("desc_id", "")
            if eid in seen:                          # dedup: already used (this or another portion)
                continue
            if _ok(e):
                picked.append(e); seen.add(eid)
            if len(picked) >= per:
                break
        if len(picked) < per:
            short.append(f"r{cell[0]}/{cell[1]}={len(picked)}")
        selected += picked
    print(f"[{label}] {len(selected)} rows / {len(cells)} cells" + (f"  SHORT:{short}" if short else ""))
    return selected


def build_splits():
    OUT.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    real, synth = _load_pool(REAL_SRC), _load_pool(SYNTH_SRC)

    # story-disjoint partition over the union (synth = nonsensical only ⊂ real's stories)
    story_split = _split_stories({e["story"] for e in real + synth}, rng)

    def bucket(entries):
        b = collections.defaultdict(list)
        for e in entries:
            b[story_split[e["story"]]].append(e)
        return b
    R, S = bucket(real), bucket(synth)

    # SFT = 8K real + 2K synth ; RL = 2K real + 8K synth ; eval = real held-out.
    # `seen` dedups by desc_id GLOBALLY (real/synth overlap heavily), so no record is used twice.
    seen = set()
    splits = {
        "sft":  _sample_portion(R["sft"], SFT_REAL, rng, "sft/real", seen)
                + _sample_portion(S["sft"], SFT_SYNTH, rng, "sft/synth", seen),
        "rl":   _sample_portion(R["rl"], RL_REAL, rng, "rl/real", seen)
                + _sample_portion(S["rl"], RL_SYNTH, rng, "rl/synth", seen),
        "eval": _sample_portion(R["eval"], EVAL_N, rng, "eval/real", seen),
    }
    for split, rows in splits.items():
        rng.shuffle(rows)
        _write(OUT / SPLIT_FILE[split], [_row(e["r"]) for e in rows])
        cov = collections.Counter(e["rung"] for e in rows)
        print(f"→ {SPLIT_FILE[split]}: {len(rows)} rows  rungs={dict(sorted(cov.items()))}")


# ── veRL parquet ─────────────────────────────────────────────────────────────

def build_parquet():
    for src, dst, split in (("train_rl.jsonl", "train_rl.parquet", "train"),
                            ("test_rl.jsonl", "test.parquet", "test")):
        rows = [json.loads(l) for l in open(OUT / src)]
        records = [{
            "prompt": [{"role": "system", "content": CLADDER_SYSTEM},
                       {"role": "user", "content": r["prompt"]}],
            "data_source": r["source"],
            "reward_model": {"ground_truth": json.dumps(r["groundtruth"])},
            "extra_info": {"split": split, "id": str(r["id"])},
        } for r in rows]
        pd.DataFrame(records).to_parquet(OUT / dst, index=False)
        print(f"{len(records):>5} rows → {dst}")


def build_causci_val():
    """CauSci (canonical data/merged.jsonl benchmark: real/synthetic/qr) as a second veRL validation
    set — rolled out with the live policy during RL and scored by reward.score_causci
    (data_source=causci). TEST ONLY (never in train_files). Columns come from each study's CSV header."""
    records = []
    for r in load_bench():
        cols = csv_columns(r["csv_path"])
        records.append({
            "prompt": [{"role": "system", "content": CAUSCI_SYSTEM},
                       {"role": "user", "content": causci_user(r["description"], cols, r["query"])}],
            "data_source": "causci",
            "reward_model": {"ground_truth": json.dumps({"step1": r["step1"], "step2": r["method"]})},
            "extra_info": {"split": "causci", "csplit": r["source"], "columns": cols,
                           "csv_path": r["csv_path"], "id": str(r["id"])},
        })
    if records:
        pd.DataFrame(records).to_parquet(OUT / "causci_val.parquet", index=False)
        cov = collections.Counter(r["extra_info"]["csplit"] for r in records)
        print(f"{len(records):>5} rows → causci_val.parquet  {dict(cov)}")


# ── SFT jsonl (Phase A turn-by-turn + Phase B1 single-pass) ──────────────────
# Turns elicited in deployment order so the two phases never teach conflicting block orders.

TURNS = [
    (["mapping"],    "Assign a canonical symbol (X=treatment, Y=outcome, V1,V2,...=rest) to each "
                     "variable in the world. Output only:\n<mapping>...</mapping>"),
    (["query_type"], "Classify what the question asks for. Output only:\n<query_type>...</query_type>"),
    (["graph"],      "Using the description and mapping, write the causal graph as directed edges. "
                     "Output only:\n<graph>...</graph>"),
    (["estimand"],   "Write the symbolic estimand for this query. Output only:\n<estimand>...</estimand>"),
    (["data"],       "Extract the probabilities stated in the given information. Output only:\n<data>...</data>"),
    (["derivation"], "Using the graph, rewrite the estimand into an estimable form. "
                     "Output only:\n<derivation>...</derivation>"),
    (["arithmetic", "answer"], "Substitute the data into the derivation, compute, and give the final "
                               "answer. Output only:\n<arithmetic>...</arithmetic>\n<answer>Yes or No</answer>"),
]


def _render(blocks, keys):
    return "\n".join(f"<{k}>{blocks[k]}</{k}>" for k in keys)


def _turn_examples(prompt, target):
    blocks = parse(target)
    blocks["answer"] = {"yes": "Yes", "no": "No"}.get(blocks["answer"], blocks["answer"])
    out, prior = [], []
    for keys, instr in TURNS:
        ctx = prompt if not prior else f"{prompt}\n\n{_render(blocks, prior)}"
        out.append({"system": CLADDER_SYSTEM, "prompt": f"{ctx}\n\n{instr}", "completion": _render(blocks, keys)})
        prior += keys
    return out


def build_sft_jsonl():
    sft_rows = [json.loads(l) for l in open(OUT / "sft.jsonl")]
    eval_rows = [json.loads(l) for l in open(OUT / "test_rl.jsonl")]
    single = [{"system": CLADDER_SYSTEM, "prompt": r["prompt"], "completion": r["target"]} for r in sft_rows]
    turns = [ex for r in sft_rows for ex in _turn_examples(r["prompt"], r["target"])]
    test = [{"system": CLADDER_SYSTEM, "prompt": r["prompt"], "completion": r["target"],
             "groundtruth": r["groundtruth"]} for r in eval_rows]
    _write(OUT / "sft_single.jsonl", single)
    _write(OUT / "sft_turns.jsonl", turns)
    _write(OUT / "sft_test.jsonl", test)
    print(f"SFT: single={len(single)}  turns={len(turns)}  test={len(test)}")


def main():
    build_splits()
    build_parquet()
    build_causci_val()
    build_sft_jsonl()


if __name__ == "__main__":
    main()
