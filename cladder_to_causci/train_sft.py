"""train_sft.py — LoRA SFT for the CLadder six-step schema. Base=Qwen3-8B, DDP via torchrun.

Teacher-forced on (system, prompt, completion) jsonl, loss masked on the prompt. Works for both
phases (data differs, trainer doesn't):
  Phase A  --train output/sft_turns.jsonl                       (turn-by-turn: learn each step)
  Phase B1 --train output/sft_single.jsonl --init <A ckpt>      (single-pass: collapse to one rollout)

Every EVAL_EVERY steps (rank 0) it greedily generates the single-pass completion on sft_test.jsonl,
parses + scores with the RL reward's per-step metrics, and appends to output/sft/metrics.csv —
watch graph_f1 → ~1.0 before starting RL (the gate starves the policy otherwise). Saves the
LoRA-merged model to --out (default output/sft/final), which the next phase / RL loads as base.

Run:  torchrun --nproc_per_node=<gpus> train_sft.py --train output/sft_turns.jsonl --out output/sft/turnsA
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainerCallback, TrainingArguments)
from peft import LoraConfig, get_peft_model

from schema import parse
from reward import compute_eval_metrics, grade   # GRADER only — SFT never touches the reward

BASE_MODEL = os.environ.get("SFT_BASE", "Qwen/Qwen3-8B")
MAX_SEQ_LEN = 2048
EVAL_EVERY = int(os.environ.get("SFT_EVAL_EVERY", 100))   # eval frequency (steps)
EVAL_N = int(os.environ.get("SFT_EVAL_N", 256))           # test rows per periodic eval (rank 0)
GEN_MAX_NEW = 512
GRAPH_GATE = 0.95                 # SFT→RL handoff: held-out graph-extraction F1 must clear this (near-ceiling)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"

METRIC_KEYS = ["graph_f1", "query_acc", "estimand_acc", "calc_acc", "answer_acc", "full_correct"]


# ── tokenization ────────────────────────────────────────────────────────────

def _prompt_ids(tok, system, prompt):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)


def _tokenize(row, tok):
    p = _prompt_ids(tok, row["system"], row["prompt"])
    t = tok(row["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(p) > MAX_SEQ_LEN - len(t):
        p = p[-(MAX_SEQ_LEN - len(t)):]         # left-truncate: keep the instruction + prior blocks
    ids = p + t
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(p) + t}


class JsonlDS(Dataset):
    def __init__(self, path, tok):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return _tokenize(self.rows[i], self.tok)


# ── held-out single-pass GRADER (measurement only — never a training signal) ──

def grade_rows(model, tok, rows, batch=16):
    """Batched single-pass generation on held-out rows → parse → GRADE (continuous per-segment)
    → diagnostic metrics dict. Uses the grader, not the reward."""
    model.eval()
    tok.padding_side = "left"                                   # decoder-only batched generation
    items = []
    with torch.no_grad():
        for i in tqdm(range(0, len(rows), batch), desc="sft-eval gen", leave=False):
            chunk = rows[i:i + batch]
            ids = [_prompt_ids(tok, r["system"], r["prompt"]) for r in chunk]
            enc = tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(model.device)
            out = model.generate(**enc, max_new_tokens=GEN_MAX_NEW, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
            for r, g in zip(chunk, out[:, enc["input_ids"].shape[1]:]):
                parsed = parse(tok.decode(g, skip_special_tokens=True))
                items.append((parsed, grade(parsed, r["groundtruth"]) if any(parsed.values()) else None))
    model.train()
    return compute_eval_metrics(items)


def grade_causci(model, tok, batch=16, phase=None, outdir=None):
    """CauSci transfer eval on the SAME loaded model (reuses batched generation, PO prompt,
    thinking-off; scores via causci_eval.score_causci). Saves a sample of responses to
    <outdir>/causci_samples_<phase>.jsonl for review. Returns metrics dict or None."""
    import causci_eval as ce
    from schema import CAUSCI_SYSTEM
    recs = ce.load_bench()
    if not recs:
        return None
    tok.padding_side = "left"
    model.eval()
    items, samples = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(recs), batch), desc="causci-eval gen", leave=False):
            chunk = recs[i:i + batch]
            cols = [ce.csv_columns(r["csv_path"]) for r in chunk]
            msgs = [[{"role": "system", "content": CAUSCI_SYSTEM},
                     {"role": "user", "content": ce.build_user(r["description"], c, r["query"])}]
                    for r, c in zip(chunk, cols)]
            ids = [tok.apply_chat_template(p, add_generation_prompt=True, enable_thinking=False,
                                           tokenize=True) for p in msgs]
            enc = tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(model.device)
            out = model.generate(**enc, max_new_tokens=512, do_sample=False, pad_token_id=tok.pad_token_id)
            for r, c, g in zip(chunk, cols, out[:, enc["input_ids"].shape[1]:]):
                sol = tok.decode(g, skip_special_tokens=True)
                _, comp = ce.score_causci(sol, c, {"step1": r["step1"], "step2": r["method"]}, r["source"])
                items.append((None, comp))
                if len(samples) < 30:
                    samples.append({"id": r.get("id"), "split": r["source"], "comp": comp,
                                    "gold_method": r["method"], "gold_step1": r["step1"],
                                    "completion": sol[:2500]})
    model.train()
    if outdir and samples:
        with open(Path(outdir) / f"causci_samples_{phase or 'x'}.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
    return ce.compute_causci_metrics(items)


class EvalCallback(TrainerCallback):
    def __init__(self, model, tok, test_rows, csv_path):
        self.model, self.tok, self.rows, self.csv = model, tok, test_rows, csv_path

    def _run(self, step):
        m = grade_rows(self.model, self.tok, self.rows)
        row = {"step": step, **{k: round(v, 4) for k, v in m.items() if isinstance(v, (int, float))}}
        print("[sft_eval] step %d | overall: %s | per-rung answer: %s" % (
            step, {k: round(m.get(k, 0.0), 3) for k in METRIC_KEYS},
            {f"r{r}": round(m.get(f"r{r}/answer_acc", 0.0), 3) for r in (1, 2, 3)}), flush=True)
        new = not self.csv.exists()
        with open(self.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def on_step_end(self, args, state, control, **kw):
        if state.is_world_process_zero and state.global_step % EVAL_EVERY == 0:
            self._run(state.global_step)


# ── post-SFT diagnostic + SFT→RL handoff gate ───────────────────────────────

def grade_checkpoint(model_path, gate, phase=None):
    """Run the continuous grader over the held-out CLadder eval split → diagnostic table +
    write output/sft/grade.json. If `phase` is given (base/A/B), also append metrics to
    output/sft/phase_metrics.jsonl (for the base-vs-phase plot). Handoff gate on graph_f1.
    Exits nonzero on FAIL."""
    outdir = OUT / "sft"
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True)
    rows = [json.loads(l) for l in open(OUT / "sft_test.jsonl")]
    m = grade_rows(model, tok, rows)
    print(f"\n=== held-out CLadder diagnostic{f' [{phase}]' if phase else ''} (grader) ===")
    for k in METRIC_KEYS:
        print(f"  {k}: {m.get(k, 0.0):.4f}")
    for r in (1, 2, 3):
        print(f"  rung{r}: answer={m.get(f'r{r}/answer_acc',0):.3f} graph={m.get(f'r{r}/graph_f1',0):.3f} "
              f"estimand={m.get(f'r{r}/estimand_acc',0):.3f} full={m.get(f'r{r}/full',0):.3f}")
    cm = grade_causci(model, tok, phase=phase, outdir=outdir)   # CauSci transfer on the same checkpoint
    if cm:
        print("  [causci] " + " ".join(f"{k}={cm.get(k,0):.3f}" for k in
              ("method_correctness", "treatment_acc", "outcome_acc", "confounder_f1")), flush=True)
        m.update({f"causci_{k}": v for k, v in cm.items() if isinstance(v, (int, float))})
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(outdir / "grade.json", "w"))
    if phase:
        with open(outdir / "phase_metrics.jsonl", "a") as f:
            f.write(json.dumps({"phase": phase, **m}) + "\n")
    ok = m.get("graph_f1", 0.0) >= gate
    print(f"\nHANDOFF GATE: graph_f1={m.get('graph_f1', 0.0):.4f} vs {gate} → "
          + ("PASS — RL may start" if ok else "FAIL — do NOT start RL (graph gate would starve)"), flush=True)
    sys.exit(0 if ok else 1)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=None)
    ap.add_argument("--grade", default=None, help="grade a merged ckpt on the held-out split + handoff gate")
    ap.add_argument("--gate", type=float, default=GRAPH_GATE)
    ap.add_argument("--phase", default=None, help="label (base/A/B) → append to phase_metrics.jsonl")
    ap.add_argument("--init", default=None, help="LoRA-merged ckpt to continue from (Phase B ← Phase A)")
    ap.add_argument("--out", default=str(OUT / "sft" / "final"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    if args.grade:
        grade_checkpoint(args.grade, args.gate, args.phase)
        return
    if not args.train:
        ap.error("need --train (SFT) or --grade (diagnostic)")

    base = args.init or BASE_MODEL
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.config.use_cache = False
    model.enable_input_require_grads()   # required for gradient checkpointing + LoRA (frozen base)

    train_ds = JsonlDS(args.train, tok)
    test_rows = [json.loads(l) for l in open(OUT / "sft_test.jsonl")][:EVAL_N]
    (OUT / "sft").mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=args.out + "_hf", per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum, num_train_epochs=args.epochs, learning_rate=args.lr,
        bf16=True, logging_steps=20, save_strategy="no", report_to=[], warmup_ratio=0.03,
        lr_scheduler_type="cosine", gradient_checkpointing=True, ddp_find_unused_parameters=False)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                      callbacks=[EvalCallback(model, tok, test_rows, OUT / "sft" / "metrics.csv")])
    trainer.train()

    if trainer.is_world_process_zero():
        merged = model.merge_and_unload()
        merged.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        print(f"[sft] merged model → {args.out}", flush=True)


if __name__ == "__main__":
    main()
