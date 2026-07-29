"""build_data.py — assemble reasoning-augmented SFT data for CauSci synth→{real,qr} transfer.

Experiment: instead of training on just the label, train on (reasoning trace + label). The label
(the gold <method>/<variables> block) is built DETERMINISTICALLY from ground truth; a bigger LLM only
writes the reasoning explaining WHY that label is correct. The model then learns to reason, then answer.

Emits the contract train_sft.py reads:
  data/train.jsonl  {"system","prompt","completion"}  completion = reasoning + gold  (SYNTH split)
  data/eval.jsonl   {"system","prompt","columns","gt","split"}                        (REAL + QR)

The LLM API call is the ONLY placeholder — see generate_reasoning(). Everything around it is built:
record loading, prompt/gold assembly, a resumable on-disk cache (reruns never re-bill the API), a
DRY-RUN stub (REASON_DRY_RUN=1) to test the pipeline without the API, and the jsonl writers.

Run on a node WITH internet (login node) — the API needs it; compute nodes don't have it.
  REASON_DRY_RUN=1 python3 build_data.py      # plumbing test, stub reasoning
  OPENROUTER_API_KEY=... python3 build_data.py # real (Fable → Opus 4.8 via llm_reason.py)
"""

import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

# reuse the CauSci prompt + gold-answer builders + LLM client from the sibling task (single source of truth)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cladder_to_causci"))
from causci_eval import load_bench, build_user, csv_columns, gold_output
from schema import CAUSCI_SYSTEM
from llm_reason import call_llm

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
DATA = ROOT / "data"
CACHE = DATA / "reasoning_cache.jsonl"           # {id: reasoning}, persisted as we go → resumable
DRY_RUN = bool(os.environ.get("REASON_DRY_RUN"))
WORKERS = int(os.environ.get("REASON_WORKERS", 8))       # concurrent API calls (I/O-bound)
_LOCK = threading.Lock()
SYNTH_SRC = "cladder_to_causci/causci_synthetic_share"   # the 449-dataset synth pool (train source)

# framed as "why these are the correct deductions" (objective), NOT "justify the given answer" (sycophantic)
TEACHER_SYS = "You are a causal inference expert."
CAUSCI_TEACHER = """As a causal inference expert, look at the dataset and the question below and write a \
short explanation of the answer. The explanation must tell why the variable roles and the method are the \
correct deductions from the dataset and question.

## Dataset
{description}
## Columns
{columns}
## Question
{question}

## The correct deductions
{answer}

Rules:
- Explain why these roles and this method are correct, reasoning only from the dataset/columns/question.
- Write it as objective reasoning about what the data implies — not as defending a given answer, and \
never mention that an answer was provided.
- Ground each claim in a specific column or sentence; invent nothing.
- Do not restate the <method>/<variables> tags — stop before them.
- A few sentences.
Output only the explanation."""


# ── reasoning generation (OpenRouter, Fable → Opus 4.8) ─────────────────────

def _answer_lines(gold, roles):
    menu = re.search(r"<method>(.*?)</method>", gold)
    lines = [f"method: {menu.group(1).strip() if menu else ''}",
             f"treatment: {roles.get('treatment')}", f"outcome: {roles.get('outcome')}",
             f"confounders: {', '.join(roles.get('controls') or []) or 'NA'}"]
    for k in ("instrument", "running_variable", "time_variable", "group_variable", "mediator"):
        if roles.get(k):
            lines.append(f"{k}: {roles[k]}")
    return "\n".join(lines)


def generate_reasoning(description, columns, question, gold, roles):
    user = CAUSCI_TEACHER.format(description=description, columns="\n".join(columns),
                                 question=question, answer=_answer_lines(gold, roles))
    txt = call_llm(TEACHER_SYS, user, max_tokens=400)
    return re.split(r"<method>|<variables>|<cues>", txt)[0].strip()   # drop any restated answer block


def _stub(*a):
    return "[DRY-RUN placeholder: why these roles and this method are the correct deductions]"


# ── reasoning cache (so reruns / crashes never re-bill the API) ──────────────

def load_cache():
    if not CACHE.exists():
        return {}
    out = {}
    for l in open(CACHE):
        try:
            r = json.loads(l); out[r["id"]] = r["reasoning"]
        except Exception:
            pass
    return out


def reasoning_for(rid, description, columns, question, gold, roles, cache):
    with _LOCK:
        if rid in cache:
            return cache[rid]
    r = _stub() if DRY_RUN else generate_reasoning(description, columns, question, gold, roles)
    with _LOCK:                                  # thread-safe cache + append
        cache[rid] = r
        DATA.mkdir(parents=True, exist_ok=True)
        with open(CACHE, "a") as f:              # persist immediately (resumable)
            f.write(json.dumps({"id": rid, "reasoning": r}) + "\n")
    return r


# ── record loading ───────────────────────────────────────────────────────────

def load_synth():
    """449-dataset synth pool (causci_synthetic_share). ground_truth == step1 role dict; method == step2."""
    meta = json.load(open(PROJ / SYNTH_SRC / "metadata.json"))
    return [{"id": m["data_file"][:-4], "description": m["description"], "query": m["query"],
             "csv_path": f"{SYNTH_SRC}/data/{m['data_file']}",
             "step1": m["ground_truth"], "method": m["method"]} for m in meta]


def load_eval():
    """real + qr benchmark rows (merged.jsonl) — the transfer targets. Same shape as load_synth."""
    return [{"id": str(r["id"]), "description": r["description"], "query": r["query"],
             "csv_path": r["csv_path"], "step1": r["step1"], "method": r["method"], "split": r["source"]}
            for r in load_bench() if r["source"] in ("real", "qr")]


# ── completion assembly (format lives here — tweak freely) ───────────────────

def build_completion(reasoning, gold):
    return f"<reasoning>{reasoning}</reasoning>\n{gold}"


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    cache = load_cache()

    synth = load_synth()

    def build_row(r):
        cols = csv_columns(r["csv_path"])
        prompt = build_user(r["description"], cols, r["query"])
        gold = gold_output({"step1": r["step1"], "step2": r["method"]})
        reasoning = reasoning_for(r["id"], r["description"], cols, r["query"], gold, r["step1"], cache)
        return {"system": CAUSCI_SYSTEM, "prompt": prompt, "completion": build_completion(reasoning, gold)}

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:      # I/O-bound API calls, order preserved
        rows = list(tqdm(ex.map(build_row, synth), total=len(synth),
                         desc=f"train synth (reasoning{' DRY' if DRY_RUN else ''}) x{WORKERS}"))
    with open(DATA / "train.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")

    evalr = load_eval()
    with open(DATA / "eval.jsonl", "w") as f:
        for r in evalr:
            cols = csv_columns(r["csv_path"])
            prompt = build_user(r["description"], cols, r["query"])
            f.write(json.dumps({"system": CAUSCI_SYSTEM, "prompt": prompt, "columns": cols,
                                "gt": {"step1": r["step1"], "step2": r["method"]},
                                "split": r["split"]}) + "\n")

    ev = {}
    for r in evalr:
        ev[r["split"]] = ev.get(r["split"], 0) + 1
    print(f"train: {len(synth)} synth rows → data/train.jsonl | eval: {ev} → data/eval.jsonl", flush=True)


if __name__ == "__main__":
    main()
