"""
Evaluation pipeline for causal alignment.

Loads output/test.jsonl, generates completions from the policy model (greedy, temp=0),
parses per-step outputs, scores each row using heuristics + DeepSeek-Math judge,
and writes results + aggregate metrics.

Usage:
    python src/eval/eval.py [--limit N] [--model MODEL] [--output-dir OUTPUT_DIR]
    python src/eval/eval.py --model output/checkpoints/final --output-dir output/eval_post_grpo
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.metrics import aggregate_metrics, score_causcibench, score_cladder
from src.eval.parser import parse_completion

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Config ─────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen3-14B"
JUDGE_MODEL   = "deepseek-ai/deepseek-math-7b-instruct"
TEST_DATA     = ROOT / "output" / "test.jsonl"
OUTPUT_DIR    = ROOT / "output" / "eval"

GENERATION_KWARGS = dict(
    max_new_tokens=4096,
    do_sample=False,
    temperature=1.0,
    repetition_penalty=1.1,
)

BATCH_SIZE = 4

JUDGE_QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_name: str):
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_judge(model_name: str):
    print(f"Loading judge tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading judge model: {model_name} (4-bit)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=JUDGE_QUANT_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# ── Generation ─────────────────────────────────────────────────────────────────

def generate_completions(prompts: list[str], model, tokenizer, batch_size: int = BATCH_SIZE) -> list[str]:
    completions = []
    device = next(model.parameters()).device

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        if (i // batch_size) % 10 == 0:
            print(f"  Generating batch {i // batch_size + 1} / {math.ceil(len(prompts) / batch_size)}", flush=True)

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                **GENERATION_KWARGS,
            )

        for j, out in enumerate(output_ids):
            prompt_len = inputs["input_ids"].shape[1]
            completions.append(tokenizer.decode(out[prompt_len:], skip_special_tokens=True))

    return completions


# ── Eval loop ──────────────────────────────────────────────────────────────────

def run_eval(rows: list[dict], model, tokenizer, judge_model, judge_tokenizer) -> list[dict]:
    print(f"\nGenerating completions for {len(rows)} rows...")
    prompts = [r["prompt"] for r in rows]
    completions = generate_completions(prompts, model, tokenizer)

    print("\nParsing and scoring...")
    results = []
    for i, (row, completion) in enumerate(zip(rows, completions)):
        if i % 500 == 0:
            print(f"  Scoring row {i} / {len(rows)}", flush=True)

        parsed = parse_completion(completion, row["source"])
        gt = row["groundtruth"]

        if row["source"] == "cladder":
            scores = score_cladder(parsed, gt, judge_model, judge_tokenizer)
        else:
            scores = score_causcibench(parsed, gt, judge_model, judge_tokenizer)

        results.append({
            "id":          row["id"],
            "source":      row["source"],
            "label":       row["label"],
            "completion":  completion,
            "parsed":      parsed,
            "groundtruth": gt,
            "scores":      scores,
        })

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",      type=int, default=None, help="Eval only first N rows")
    parser.add_argument("--model",      type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading test data from {TEST_DATA}")
    with open(TEST_DATA) as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[:args.limit]
    print(f"  {len(rows)} rows loaded")

    model, tokenizer          = load_model(args.model)
    judge_model, judge_tokenizer = load_judge(JUDGE_MODEL)

    results = run_eval(rows, model, tokenizer, judge_model, judge_tokenizer)

    results_path = out_dir / "results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} results → {results_path}")

    metrics = aggregate_metrics(results)
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics → {metrics_path}")

    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    if "cladder" in metrics:
        cl = metrics["cladder"]
        print(f"\nCLadder (n={cl['n']})")
        print(f"  Accuracy:    {cl['accuracy']:.1f}%")
        print(f"  Avg score:   {cl['avg_score']:.1f}/70")
        print(f"  Step scores: s1={cl['step1_avg']:.1f}  s2={cl['step2_avg']:.1f}  "
              f"s3={cl['step3_avg']:.1f}  s5={cl['step5_avg']:.1f}")
        print(f"  By query type:")
        for qt, v in cl["by_query_type"].items():
            print(f"    {qt:<22} n={v['n']:<5} acc={v['accuracy']:.1f}%")

    if "causcibench" in metrics:
        cs = metrics["causcibench"]
        print(f"\nCauSciBench (n={cs['n']})")
        print(f"  Avg score:         {cs['avg_score']:.1f}/60")
        print(f"  Method accuracy:   {cs['method_accuracy']:.1f}%")
        print(f"  Median rel error:  {cs['median_rel_error']}")
        print(f"  Step scores: s1={cs['step1_avg']:.1f}  s2={cs['step2_avg']:.1f}  "
              f"s3={cs['step3_avg']:.1f}  s5={cs['step5_avg']:.1f}")
        print(f"  By method:")
        for m, v in cs["by_method"].items():
            print(f"    {m:<20} n={v['n']:<4} acc={v['accuracy']:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
