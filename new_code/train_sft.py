"""SFT trainer for CauSciBench (phase 1). Base = POLICY_MODEL, LoRA, DDP via torchrun.

Teaches prompt -> {step1, step2} JSON on TRAIN_SFT_JSONL (no-think targets, since we
have no reference reasoning traces). Every SFT_EVAL_EVERY steps it greedily generates on
TEST_SFT_JSONL, parses + scores with compute_eval_metrics, and appends a row to
SFT_METRICS_CSV — the SFT segment of the SFT->RL timeline. Saves the LoRA-merged model to
SFT_CKPT, which RL loads as its base (config.RL_BASE_MODEL).

Run:  torchrun --nproc_per_node=<gpus> train_sft.py
"""

import csv
import json
import re

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainerCallback, TrainingArguments)
from peft import LoraConfig, get_peft_model

import config
from prompts import CAUSCI_SYSTEM
from eval_metrics import compute_eval_metrics

METRIC_KEYS = ["causci/method_acc", "causci/treatment_acc", "causci/outcome_acc",
               "causci/control_acc", "causci/effect_acc", "causci/mre"]


# ── Targets & tokenization ──────────────────────────────────────────────

def _target(gt):
    """The JSON the model should emit, built from ground-truth (fields we don't have → null)."""
    s1 = gt["step1"]
    obj = {"step1": {"treatment": s1.get("treatment"), "outcome": s1.get("outcome"),
                     "controls": s1.get("controls") or [], "instrument": s1.get("instrument"),
                     "running_variable": s1.get("running_variable"), "cutoff": None,
                     "time_variable": s1.get("time_variable"), "group_variable": s1.get("group_variable"),
                     "mediator": None, "estimand": None},
           "step2": gt["step2"]}
    return json.dumps(obj, ensure_ascii=False)


def _prompt_ids(tok, prompt):
    msgs = [{"role": "system", "content": CAUSCI_SYSTEM}, {"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
    return ids["input_ids"] if not isinstance(ids, list) else ids


def _tokenize(row, tok):
    prompt_ids = _prompt_ids(tok, row["prompt"])
    target_ids = tok(_target(row["groundtruth"]) + tok.eos_token, add_special_tokens=False)["input_ids"]
    max_prompt = config.SFT_MAX_SEQ_LEN - len(target_ids)
    if len(prompt_ids) > max_prompt:                  # left-truncate: keep task + JSON schema at the end
        prompt_ids = prompt_ids[-max_prompt:]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class JsonlDS(Dataset):
    def __init__(self, path, tok):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return _tokenize(self.rows[i], self.tok)


# ── Eval-time parsing (mirrors reward.extract_causci: column-match predicted names) ──

def _sanit(s):
    return re.sub(r"[.\s\-]", "_", str(s).strip().lower())


def _extract_json(s):
    if "</think>" in s:
        s = s.split("</think>")[-1]
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1:
        return None
    js = s[a:b + 1]
    for fix in (lambda x: x, lambda x: re.sub(r",\s*([}\]])", r"\1", x)):
        try:
            return json.loads(fix(js))
        except Exception:
            pass
    return None


def _columns(csv_path):
    try:
        return pd.read_csv(csv_path, nrows=0).columns.tolist()
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, nrows=0, encoding="latin-1").columns.tolist()


def _parse(completion, csv_path):
    obj = _extract_json(completion)
    if not isinstance(obj, dict) or "step1" not in obj or "step2" not in obj:
        return None
    step2 = obj.get("step2")
    if not isinstance(step2, str):                  # model may nest step2 as a dict/list
        return None
    method = step2.strip().lower()
    s1 = obj.get("step1")
    if not isinstance(s1, dict):
        return None
    cmap = {_sanit(c): c for c in _columns(csv_path)}

    def m(v):
        if isinstance(v, list):
            v = v[0] if v else ""
        return cmap.get(_sanit(v)) if isinstance(v, str) else None

    treatment, outcome = m(s1.get("treatment")), m(s1.get("outcome"))
    if treatment is None or outcome is None:
        return None
    instrument       = m(s1.get("instrument"))
    running_variable = m(s1.get("running_variable"))
    time_variable    = m(s1.get("time_variable"))
    group_variable   = m(s1.get("group_variable"))
    mediator         = m(s1.get("mediator"))
    # reject incomplete method-specific specs (mirrors reward.extract_causci) — keeps eval/RL
    # consistent and prevents library_fn from running with missing vars
    if method == "iv"        and instrument       is None:                                   return None
    if method == "rdd"       and (running_variable is None or s1.get("cutoff") is None):      return None
    if method == "did"       and (time_variable    is None or group_variable is None):        return None
    if method == "frontdoor" and mediator         is None:                                   return None
    return {"step1": {"treatment": treatment, "outcome": outcome,
                      "controls": [c for x in (s1.get("controls") or []) if (c := m(x))],
                      "instrument": instrument or "", "running_variable": running_variable or "",
                      "cutoff": s1.get("cutoff"), "time_variable": time_variable or "",
                      "group_variable": group_variable or "", "mediator": mediator or "",
                      "estimand": (s1.get("estimand") or "")},
            "step2": method}


@torch.no_grad()
def _gen_items(model, tok, rows, disable=False):
    """Generate on `rows` and return [(parsed, gt, csv_path), …] (no scoring — done after gather)."""
    was_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True   # grad-checkpointing turns this off in training → re-enable for fast KV-cached gen
    model.eval()
    items = []
    for r in tqdm(rows, desc="sft_eval gen", disable=disable, leave=False):
        enc = tok.apply_chat_template(
            [{"role": "system", "content": CAUSCI_SYSTEM}, {"role": "user", "content": r["prompt"]}],
            add_generation_prompt=True, enable_thinking=False, tokenize=True, return_tensors="pt")
        ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"])
        ids = ids[:, -config.SFT_MAX_SEQ_LEN:].to(model.device)
        out = model.generate(ids, max_new_tokens=config.SFT_EVAL_MAX_NEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        completion = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        items.append((_parse(completion, r["csv_path"]), r["groundtruth"], r["csv_path"]))
    model.train()
    model.config.use_cache = was_cache
    return items


def _append_csv(pass_idx, step, metrics):
    new = not config.SFT_METRICS_CSV.exists()
    with open(config.SFT_METRICS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["eval_pass", "step"] + METRIC_KEYS)
        w.writerow([pass_idx, step] + [f"{metrics.get(k, 0.0):.6f}" for k in METRIC_KEYS])


class CausalEval(TrainerCallback):
    """Greedy causal eval on the test set: at train start (baseline) and every SFT_EVAL_EVERY steps."""

    def __init__(self, trainer, tok, rows):
        self.trainer, self.tok, self.rows, self.passes = trainer, tok, rows, 0

    def _run(self, state):
        acc = self.trainer.accelerator
        world, rank = acc.num_processes, acc.process_index
        shard = self.rows[rank::world]                       # each rank evaluates a disjoint slice
        if rank == 0:
            print(f"[sft_eval] generating on {len(self.rows)} test rows across {world} rank(s)…", flush=True)

        model = acc.unwrap_model(self.trainer.model)
        local = _gen_items(model, self.tok, shard, disable=(rank != 0))   # progress bar on rank 0 only

        if dist.is_available() and dist.is_initialized() and world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, local)          # collect each rank's items
            items = [it for sub in gathered for it in sub]
        else:
            items = local

        if rank == 0:                                        # score the full set once, on rank 0
            metrics = compute_eval_metrics(items)
            self.passes += 1
            _append_csv(self.passes, state.global_step, metrics)
            print(f"[sft_eval] pass:{self.passes} step:{state.global_step} "
                  + " ".join(f"{k}:{v:.4f}" for k, v in sorted(metrics.items())), flush=True)
        acc.wait_for_everyone()   # all ranks resync after rank-0 generates

    def on_train_begin(self, args, state, control, **kw):
        self._run(state)

    def on_step_end(self, args, state, control, **kw):
        if state.global_step % config.SFT_EVAL_EVERY == 0:
            self._run(state)

    def on_train_end(self, args, state, control, **kw):
        self._run(state)   # always capture a final point, even if the cadence missed


def main():
    tok = AutoTokenizer.from_pretrained(config.POLICY_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.POLICY_MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=config.SFT_LORA_R, lora_alpha=2 * config.SFT_LORA_R, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.enable_input_require_grads()   # needed for gradient checkpointing + LoRA

    train_ds = JsonlDS(config.TRAIN_SFT_JSONL, tok)
    test_rows = [json.loads(l) for l in open(config.TEST_SFT_JSONL)]
    if config.SFT_EVAL_N:
        test_rows = test_rows[:config.SFT_EVAL_N]

    args = TrainingArguments(
        output_dir=str(config.SFT_CKPT.parent / "ckpts"),
        per_device_train_batch_size=config.SFT_BATCH_SIZE, gradient_accumulation_steps=config.SFT_GRAD_ACCUM,
        learning_rate=config.SFT_LR, num_train_epochs=config.SFT_EPOCHS, lr_scheduler_type="cosine",
        warmup_ratio=0.03, bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, save_strategy="steps", save_steps=config.SFT_EVAL_EVERY, save_total_limit=2,
        ddp_find_unused_parameters=False, remove_unused_columns=False, report_to="none")

    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.add_callback(CausalEval(trainer, tok, test_rows))
    trainer.train()

    if trainer.is_world_process_zero():
        merged = trainer.accelerator.unwrap_model(model).merge_and_unload()
        config.SFT_CKPT.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(config.SFT_CKPT)
        tok.save_pretrained(config.SFT_CKPT)
        print(f"Saved merged SFT model → {config.SFT_CKPT}")


if __name__ == "__main__":
    main()
