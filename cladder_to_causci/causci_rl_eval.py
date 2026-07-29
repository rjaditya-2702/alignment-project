"""causci_rl_eval.py — CauSci transfer eval for the RL LoRA checkpoints (thinking OFF vs ON).

Sweeps output/rl/verl_checkpoints/global_step_* (the full RL curve). Each checkpoint's
actor/lora_adapter is a standard PEFT adapter over the SFT model (base recorded in its
adapter_config), so we load base + attach the adapter for inference — NO weight merge, no
merged HF dirs on disk. Reuses causci_thinking_eval.run_pass (same prompts / budget-forcing /
sampling) so RL numbers are directly comparable to the SFT eval.

SEPARATE from the SFT eval: writes output/sft/causci_thinking_rl.jsonl (+ causci_samples_rl_*.jsonl).
plot.plot_causci_thinking() reads both files and puts SFT and RL on one axis. Resume-safe: rows are
appended per checkpoint×mode; a re-run skips finished ones (CAUSCI_FRESH=1 to start over).

Run under the cladder venv (torch + transformers + peft → GPU node):
  python3 causci_rl_eval.py                              # all steps
  python3 causci_rl_eval.py 200 400                      # only these global steps
"""

import gc
import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import causci_eval as ce
import causci_thinking_eval as te

CKPT_DIR = te.OUT / "rl" / "verl_checkpoints"
RL_JSONL = te.SFT / "causci_thinking_rl.jsonl"


def _append_row(m):
    te.SFT.mkdir(parents=True, exist_ok=True)
    with open(RL_JSONL, "a") as f:
        f.write(json.dumps({k: v for k, v in m.items() if isinstance(v, (int, float, str, bool))}) + "\n")


def eval_ckpt(label, step, ckpt, done):
    """Load base + LoRA adapter once, run the thinking modes not already in `done`, append each."""
    modes = [t for t in (False, True) if (label, t) not in done]
    if not modes:
        print(f"\n=== {label}: both modes already done — skipping ===", flush=True)
        return
    adapter = ckpt / "actor" / "lora_adapter"
    if not (adapter / "adapter_config.json").exists():
        print(f"  {label}: no PEFT adapter at {adapter} — skipping", flush=True)
        return
    base = json.load(open(adapter / "adapter_config.json"))["base_model_name_or_path"]
    print(f"\n=== {label}: base={base} + LoRA {adapter}  (running {['on' if t else 'off' for t in modes]}) ===",
          flush=True)
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter))     # attach adapter (active for inference)
    model.eval()
    recs = ce.load_bench()
    for think in modes:
        m, samples = te.run_pass(model, tok, recs, think, label)
        m.update({"label": label, "think": think, "step": step})
        with open(te.SFT / f"causci_samples_rl_{label}_{'on' if think else 'off'}.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        _append_row(m)
        tr = f" trunc_think={m['truncated_think_rate']:.2f}" if think else ""
        print(f"  think={think!s:5} " + " ".join(f"{k}={m.get(k,0):.3f}" for k in te.HEAD) + tr, flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    if os.environ.get("CAUSCI_FRESH") and RL_JSONL.exists():
        RL_JSONL.unlink()
    done = set()
    if RL_JSONL.exists():
        for l in open(RL_JSONL):
            try:
                r = json.loads(l); done.add((r["label"], r["think"]))
            except Exception:
                pass
    steps = sorted(CKPT_DIR.glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1])) \
        if CKPT_DIR.exists() else []
    want = {int(a) for a in sys.argv[1:] if a.isdigit()}                # optional: only these steps
    targets = [(f"rl_{int(s.name.split('_')[-1])}", int(s.name.split('_')[-1]), s)
               for s in steps if not want or int(s.name.split('_')[-1]) in want]
    print("RL targets:", [t[0] for t in targets], "| already done:", sorted(done), flush=True)
    for label, step, ckpt in targets:
        try:
            eval_ckpt(label, step, ckpt, done)
        except Exception as e:                              # one bad ckpt shouldn't sink the sweep
            print(f"  {label} FAILED: {e}", flush=True)

    # ── curve table: off → on per step, from all rows on disk ──
    rows = [json.loads(l) for l in open(RL_JSONL)] if RL_JSONL.exists() else []
    by_label = {}
    for m in rows:
        by_label.setdefault(m["label"], {})[m["think"]] = m
    print("\n=== CauSci transfer (RL curve): thinking OFF → ON ===")
    print(f"{'ckpt':<10} " + " ".join(f"{k.split('_')[0]:>10}" for k in te.HEAD) + f"{'trunc_on':>10}")
    for label in sorted(by_label, key=lambda x: int(x.split('_')[-1])):
        off, on = by_label[label].get(False, {}), by_label[label].get(True, {})
        cells = " ".join(f"{off.get(k,0):.2f}->{on.get(k,0):.2f}" for k in te.HEAD)
        print(f"{label:<10} {cells}  {on.get('truncated_think_rate',0):>8.2f}")
    print(f"\nwrote {RL_JSONL}")
    try:
        import plot
        plot.plot_causci_thinking()
    except Exception as e:
        print(f"skip plot ({e}) — run `python plot.py` where matplotlib is installed")


if __name__ == "__main__":
    main()
