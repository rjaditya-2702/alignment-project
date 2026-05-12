import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from src.config import SFT_OUTPUT_DIR, TEST_DATA, EVAL_MAX_TOKENS
from src.eval.parser import parse_completion


def _extract_answer(completion: str, source: str):
    """Return step5 answer from a parsed completion."""
    parsed = parse_completion(completion, source)
    v = parsed.get("step5")
    return str(v) if v is not None else ""


def evaluate():
    """Run greedy inference on TEST_DATA and report per-source accuracy."""
    with open(TEST_DATA) as f:
        rows = [json.loads(l) for l in f]

    model_path = SFT_OUTPUT_DIR / "final"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()

    results = []
    for row in tqdm(rows):
        inputs = tokenizer(
            row["prompt"], return_tensors="pt", truncation=True, max_length=3072,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=EVAL_MAX_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        pred = _extract_answer(completion, row["source"])
        gt   = str(row["groundtruth"].get("step5", ""))
        results.append({
            "id":      row.get("id", ""),
            "source":  row["source"],
            "pred":    pred,
            "gt":      gt,
            "correct": pred.strip().lower() == gt.strip().lower(),
        })

    for src in ("cladder", "causcibench"):
        subset = [r for r in results if r["source"] == src]
        if subset:
            acc = sum(r["correct"] for r in subset) / len(subset) * 100
            print(f"{src}: {acc:.1f}%  ({len(subset)} samples)")

    out_path = SFT_OUTPUT_DIR / "eval_results.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    evaluate()
