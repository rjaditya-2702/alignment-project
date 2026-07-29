"""train_sft.py — LoRA SFT for CauSci synth→{real,qr} transfer. Base=Qwen3-8B, DDP via torchrun.

Teacher-forced on (system, prompt, completion) jsonl from the SYNTH split, loss masked on the prompt
(same recipe as cladder_to_causci/train_sft.py). Every EVAL_EVERY steps (rank 0) it greedily generates
on the held-out real+qr eval rows and scores with the CauSci PO/method-menu scorer
(causci_eval.score_causci), logging overall + PER-SPLIT method/treatment/outcome/confounder_f1 to
output/sft/metrics.csv — that per-split line IS the transfer signal (train on synth, watch real & qr
climb). Saves the LoRA-merged model to --out.

Expected data (produced by your data-processing step — NOT written here). One json object per line:
  data/train.jsonl  {"system":..., "prompt":..., "completion":...}                    # synth split
  data/eval.jsonl   {"system":..., "prompt":..., "columns":[...],                     # real + qr held-out
                     "gt":{"step1":{...}, "step2":"<gold method name>"}, "split":"real"|"qr"}
The `completion` is whatever the model should emit verbatim (the <method>/<variables> block, optionally
with reasoning in front — the trainer just teacher-forces it). `columns`/`gt`/`split` feed the scorer.

Run:  python -m torch.distributed.run --nproc_per_node=<gpus> train_sft.py --train data/train.jsonl --out output/sft/final
Grade a checkpoint only:  python3 train_sft.py --grade <model_dir> --phase base
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

# reuse the CauSci scorer from the sibling task (same PO/method-menu format) — no duplication
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cladder_to_causci"))
from causci_eval import score_causci, compute_causci_metrics

BASE_MODEL = os.environ.get("SFT_BASE", "Qwen/Qwen3-8B")
MAX_SEQ_LEN = 2048
EVAL_EVERY = int(os.environ.get("SFT_EVAL_EVERY", 100))    # transfer-eval frequency (steps)
EVAL_N = int(os.environ.get("SFT_EVAL_N", 0)) or None      # cap eval rows (0/unset → all real+qr)
GEN_MAX_NEW = 512
ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SFT_DATA_DIR") or ROOT / "data")   # override to reuse another task's data
OUT = Path(os.environ.get("SFT_OUT_DIR") or ROOT / "output")   # override to write elsewhere (e.g. a transfer variant)

# monitored metrics — treatment, outcome, method, control (recall AND f1) — overall + per split
LOG_KEYS = ["method_correctness", "treatment_acc", "outcome_acc", "control_recall", "confounder_f1", "valid_rate",
            "real/method", "real/treatment", "real/outcome", "real/control_recall", "real/conf_f1",
            "qr/method", "qr/treatment", "qr/outcome", "qr/control_recall", "qr/conf_f1"]


# ── tokenization (prompt masked, teacher-forced on completion) ───────────────

def _prompt_ids(tok, system, prompt):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)


def _tokenize(row, tok):
    p = _prompt_ids(tok, row["system"], row["prompt"])
    t = tok(row["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(p) > MAX_SEQ_LEN - len(t):
        p = p[-(MAX_SEQ_LEN - len(t)):]          # left-truncate: keep the instruction closest to the answer
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


# ── held-out transfer eval (measurement only) ────────────────────────────────

def eval_transfer(model, tok, rows, batch=16, samples_out=None, preds_out=None):
    """Batched greedy generation on real+qr eval rows → CauSci scorer → per-split metrics dict.
    preds_out (optional) → dump EVERY example's {split, gold_method, pred_method} for confusion matrices."""
    model.eval()
    tok.padding_side = "left"                                    # decoder-only batched generation
    items, samples, preds = [], [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(rows), batch), desc="transfer-eval gen", leave=False):
            chunk = rows[i:i + batch]
            ids = [_prompt_ids(tok, r["system"], r["prompt"]) for r in chunk]
            enc = tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(model.device)
            out = model.generate(**enc, max_new_tokens=GEN_MAX_NEW, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
            for r, g in zip(chunk, out[:, enc["input_ids"].shape[1]:]):
                sol = tok.decode(g, skip_special_tokens=True)
                _, comp = score_causci(sol, r["columns"], r["gt"], r["split"])
                items.append((None, comp))
                preds.append({"split": r["split"], "gold_method": comp.get("gold_method", "none"),
                              "pred_method": comp.get("pred_method", "none"),
                              "treatment": comp.get("treatment", 0.0), "outcome": comp.get("outcome", 0.0),
                              "pred_treatment": comp.get("pred_treatment", ""), "gold_treatment": comp.get("gold_treatment", ""),
                              "pred_outcome": comp.get("pred_outcome", ""), "gold_outcome": comp.get("gold_outcome", ""),
                              "gold_controls": comp.get("gold_controls", [])})
                if len(samples) < 30:
                    samples.append({"split": r["split"], "comp": comp, "completion": sol[:2500]})
    model.train()
    if samples_out and samples:
        with open(samples_out, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
    if preds_out and preds:
        with open(preds_out, "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
    return compute_causci_metrics(items)


class EvalCallback(TrainerCallback):
    def __init__(self, model, tok, eval_rows, csv_path):
        self.model, self.tok, self.rows, self.csv = model, tok, eval_rows, csv_path

    def _run(self, step):
        m = eval_transfer(self.model, self.tok, self.rows)
        row = {"step": step, **{k: round(m.get(k, 0.0), 4) for k in LOG_KEYS}}
        print(f"[sft_eval] step {step} | " + " ".join(f"{k}={m.get(k,0):.3f}" for k in LOG_KEYS), flush=True)
        new = not self.csv.exists()
        with open(self.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def on_step_end(self, args, state, control, **kw):
        if state.is_world_process_zero and state.global_step % EVAL_EVERY == 0:
            self._run(state.global_step)


# ── standalone checkpoint grade (base/final snapshots for the transfer plot) ──

def grade_checkpoint(model_path, phase=None):
    outdir = OUT / "sft"
    outdir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True)
    rows = [json.loads(l) for l in open(DATA / "eval.jsonl")]
    m = eval_transfer(model, tok, rows, samples_out=outdir / f"samples_{phase or 'x'}.jsonl",
                      preds_out=outdir / f"preds_{phase or 'x'}.jsonl")
    print(f"\n=== CauSci synth→rest transfer{f' [{phase}]' if phase else ''} ===")
    print("  overall: " + " ".join(f"{k}={m.get(k,0):.3f}" for k in
          ("method_correctness", "treatment_acc", "outcome_acc", "control_recall", "valid_rate")))
    for sp in ("real", "qr"):
        print(f"  {sp}: method={m.get(f'{sp}/method',0):.3f} treat={m.get(f'{sp}/treatment',0):.3f} "
              f"outcome={m.get(f'{sp}/outcome',0):.3f} control={m.get(f'{sp}/control_recall',0):.3f}")
    json.dump(m, open(outdir / "grade.json", "w"))
    if phase:
        with open(outdir / "phase_metrics.jsonl", "a") as f:
            f.write(json.dumps({"phase": phase, **{k: v for k, v in m.items()
                                                   if isinstance(v, (int, float))}}) + "\n")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=None, help="synth-split SFT jsonl")
    ap.add_argument("--grade", default=None, help="grade a merged ckpt on real+qr eval, no training")
    ap.add_argument("--phase", default=None, help="label (base/final) → append to phase_metrics.jsonl")
    ap.add_argument("--init", default=None, help="LoRA-merged ckpt to warm-start from")
    ap.add_argument("--out", default=str(OUT / "sft" / "final"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    if args.grade:
        grade_checkpoint(args.grade, args.phase)
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
    model.enable_input_require_grads()    # required for gradient checkpointing + LoRA (frozen base)

    train_ds = JsonlDS(args.train, tok)
    eval_rows = [json.loads(l) for l in open(DATA / "eval.jsonl")]
    if EVAL_N:
        eval_rows = eval_rows[:EVAL_N]
    (OUT / "sft").mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=args.out + "_hf", per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum, num_train_epochs=args.epochs, learning_rate=args.lr,
        bf16=True, logging_steps=20, save_strategy="no", report_to=[], warmup_ratio=0.03,
        lr_scheduler_type="cosine", gradient_checkpointing=True, ddp_find_unused_parameters=False)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                      callbacks=[EvalCallback(model, tok, eval_rows, OUT / "sft" / "metrics.csv")])
    trainer.train()

    if trainer.is_world_process_zero():
        merged = model.merge_and_unload()
        merged.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        print(f"[sft] merged model → {args.out}", flush=True)


if __name__ == "__main__":
    main()
