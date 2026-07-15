"""causci_thinking_eval.py — CauSci transfer eval, thinking-OFF vs thinking-ON, across checkpoints.

For each model — base (Qwen/Qwen3-8B), SFT Phase A (output/sft/turnsA), Phase B (output/sft/final),
and the latest RL checkpoint if present — this loads it once and runs the CauSci PO/method-menu eval
(causci_eval.score_causci) TWICE: `enable_thinking=False` then `enable_thinking=True`, with identical
sampling both times so the ONLY difference is Qwen3's thinking. Answers "does thinking change transfer?"

Decoding: Qwen3 thinking-mode recommended sampling (temp 0.6 / top_p 0.95 / top_k 20), fixed seed.
The think-ON pass uses BUDGET FORCING — reasoning is capped at THINK_BUDGET tokens, `</think>` is then
force-injected, and the model gets a fresh ANSWER_BUDGET to emit the structured reply — so long CoT
can't starve the answer (the failure mode of a single large max_new_tokens). `truncated_think_rate` =
fraction whose reasoning hit the cap (had to be force-closed).

Run under the cladder venv (needs torch + transformers → GPU node):
  python3 causci_thinking_eval.py                       # auto-discover base/A/B/RL
  python3 causci_thinking_eval.py base=Qwen/Qwen3-8B B=output/sft/final rl=/path/to/merged_hf
Writes output/sft/causci_thinking.jsonl (one row per model×mode) + causci_samples_thinking_*.jsonl.
"""

import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import causci_eval as ce
from schema import CAUSCI_SYSTEM

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
SFT = OUT / "sft"
CKPT_DIR = OUT / "rl" / "verl_checkpoints"
BASE = os.environ.get("CAUSCI_BASE", "Qwen/Qwen3-8B")
THINK_BUDGET = int(os.environ.get("CAUSCI_THINK_BUDGET", 700))    # cap reasoning, then force the answer
ANSWER_BUDGET = int(os.environ.get("CAUSCI_ANSWER_BUDGET", 512))  # tokens for the final structured reply
BATCH = int(os.environ.get("CAUSCI_BATCH", 8))
HEAD = ("method_correctness", "treatment_acc", "outcome_acc", "confounder_f1")
# Qwen3 thinking-mode recommended sampling — held constant across passes; max_new_tokens set per call.
GEN = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)


def _gen(model, tok, texts, max_new):
    enc = tok(texts, padding=True, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**enc, pad_token_id=tok.pad_token_id, max_new_tokens=max_new, **GEN)
    return [tok.decode(g, skip_special_tokens=True) for g in out[:, enc["input_ids"].shape[1]:]]


def run_pass(model, tok, recs, think, label):
    """One CauSci eval pass over `recs` with enable_thinking=think. Returns (metrics, samples)."""
    torch.manual_seed(0)                                   # same RNG start for off vs on → fair A/B
    items, samples, truncated = [], [], 0
    with torch.no_grad():
        for i in tqdm(range(0, len(recs), BATCH), desc=f"{label} think={think}", leave=False):
            chunk = recs[i:i + BATCH]
            cols = [ce.csv_columns(r["csv_path"]) for r in chunk]
            msgs = [[{"role": "system", "content": CAUSCI_SYSTEM},
                     {"role": "user", "content": ce.build_user(r["description"], c, r["query"])}]
                    for r, c in zip(chunk, cols)]
            texts = [tok.apply_chat_template(p, add_generation_prompt=True, enable_thinking=think,
                                             tokenize=False) for p in msgs]
            if not think:
                sols = _gen(model, tok, texts, ANSWER_BUDGET)          # direct structured answer
            else:                                                      # budget forcing: cap think, then answer
                raw = _gen(model, tok, texts, THINK_BUDGET)
                truncated += sum("</think>" not in t for t in raw)     # hit the cap → force-closed
                forced = [t.split("</think>")[0].rstrip() + "\n</think>\n" for t in raw]
                ans = _gen(model, tok, [tx + fo for tx, fo in zip(texts, forced)], ANSWER_BUDGET)
                sols = [fo + a for fo, a in zip(forced, ans)]
            for r, c, sol in zip(chunk, cols, sols):
                _, comp = ce.score_causci(sol, c, {"step1": r["step1"], "step2": r["method"]}, r["source"])
                items.append((None, comp))
                if len(samples) < 30:
                    samples.append({"id": r.get("id"), "split": r["source"], "comp": comp,
                                    "gold_method": r["method"], "completion": sol[:3000]})
    m = ce.compute_causci_metrics(items)
    m["truncated_think_rate"] = truncated / (len(recs) or 1)
    return m, samples


def eval_model(label, path):
    """Load `path` once, run thinking-off then thinking-on. Returns [off_metrics, on_metrics]."""
    print(f"\n=== {label}: {path} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True)
    model.eval()
    recs = ce.load_bench()
    rows = []
    for think in (False, True):
        m, samples = run_pass(model, tok, recs, think, label)
        m.update({"label": label, "think": think})
        rows.append(m)
        with open(SFT / f"causci_samples_thinking_{label}_{'on' if think else 'off'}.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        tr = f" trunc_think={m['truncated_think_rate']:.2f}" if think else ""
        print(f"  think={think!s:5} " + " ".join(f"{k}={m.get(k,0):.3f}" for k in HEAD) + tr, flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def merge_rl(ckpt):
    """veRL FSDP checkpoint → HF dir via verl.model_merger (best effort; returns dir or None).
    LoRA runs may need extra merger flags — on failure the real error is printed and RL is skipped."""
    actor, target = ckpt / "actor", OUT / "rl" / "merged" / ckpt.name
    if (target / "config.json").exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
           "--local_dir", str(actor), "--target_dir", str(target)]
    print("merging RL ckpt:", " ".join(cmd), flush=True)
    subprocess.run(cmd)
    return target if (target / "config.json").exists() else None


def discover():
    targets = [("base", BASE)]
    for lbl, d in (("A", SFT / "turnsA"), ("B", SFT / "final")):
        if (Path(d) / "config.json").exists():
            targets.append((lbl, str(d)))
    steps = sorted(CKPT_DIR.glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1])) \
        if CKPT_DIR.exists() else []
    if steps:
        merged = merge_rl(steps[-1])
        if merged:
            targets.append((f"rl_{steps[-1].name}", str(merged)))
        else:
            print(f"RL merge failed for {steps[-1]} — skipping RL checkpoint", flush=True)
    return targets


def main():
    argv = [a for a in sys.argv[1:] if "=" in a]
    targets = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in argv] if argv else discover()
    print("targets:", [t[0] for t in targets], flush=True)
    all_rows = []
    for label, path in targets:
        try:
            all_rows += eval_model(label, path)
        except Exception as e:                              # one bad ckpt shouldn't sink the sweep
            print(f"  {label} FAILED: {e}", flush=True)
    SFT.mkdir(parents=True, exist_ok=True)
    with open(SFT / "causci_thinking.jsonl", "w") as f:
        for m in all_rows:
            f.write(json.dumps({k: v for k, v in m.items() if isinstance(v, (int, float, str, bool))}) + "\n")

    # ── comparison table: off → on (Δ) per model, headline metrics ──
    by_label = {}
    for m in all_rows:
        by_label.setdefault(m["label"], {})[m["think"]] = m
    print("\n=== CauSci transfer: thinking OFF → ON (Δ) ===")
    print(f"{'model':<16} " + " ".join(f"{k.split('_')[0]:>10}" for k in HEAD) + f"{'trunc_on':>10}")
    for label, modes in by_label.items():
        off, on = modes.get(False, {}), modes.get(True, {})
        cells = " ".join(f"{off.get(k,0):.2f}->{on.get(k,0):.2f}" for k in HEAD)
        print(f"{label:<16} {cells}  {on.get('truncated_think_rate',0):>8.2f}")
    print(f"\nwrote {SFT / 'causci_thinking.jsonl'}")
    try:                                                    # plotting is optional (no matplotlib on cluster venv)
        import plot
        plot.plot_causci_thinking()
    except Exception as e:
        print(f"skip plot ({e}) — run `python plot.py` where matplotlib is installed")


if __name__ == "__main__":
    main()
