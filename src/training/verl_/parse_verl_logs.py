"""
parse_verl_logs.py — parse verl_training.log into a CSV for local plotting.

Run on HPC after (or during) training:
    python3 src/training/verl_/parse_verl_logs.py

Then copy verl_metrics.csv to your laptop and run plot_verl.py.

Two line types are parsed:

1. veRL native step lines  (trainer.logger='["console"]' output):
   Pattern: contains 'step:' followed by a digit.
   Extracts: step and any key:value or key: value pairs on the line.
   Example keys emitted by veRL: actor/loss, critic/loss, kl, reward/mean, ...

2. Our eval lines:
   Pattern: starts with '[verl_eval]'
   Extracts: eval_pass and all metric key:value pairs.
   Example: [verl_eval] eval_pass:1 cladder/overall_acc:0.5100 ...

The two sets of rows are written to separate sections of the CSV:
  - train_step rows have a 'step' column
  - eval rows have an 'eval_pass' column
Both are written to the same CSV; missing columns are NaN.
"""

import re
import sys
import pandas as pd
from pathlib import Path


# Matches a numeric value: int, float, negative, scientific notation
_NUM = r'[\-]?\d+(?:\.\d+)?(?:[eE][\-+]?\d+)?'

# Matches key:value or key: value (space after colon allowed)
_KV  = re.compile(rf'([\w/]+):\s*({_NUM})')

# veRL native step line detector
_STEP = re.compile(r'\bstep[:\s]+(\d+)', re.IGNORECASE)

# Our eval line detector
_EVAL = re.compile(r'^\[verl_eval\]\s+eval_pass:(\d+)\s+(.*)')


def _extract_kv(text: str) -> dict:
    return {k: float(v) for k, v in _KV.findall(text)}


def parse_log(log_file: str = "verl_training.log", output: str = "verl_metrics.csv") -> None:
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"ERROR: {log_file} not found.", file=sys.stderr)
        sys.exit(1)

    train_rows = {}   # step -> dict of metrics
    eval_rows  = {}   # eval_pass -> dict of metrics

    with open(log_path, errors="ignore") as f:
        for line in f:
            line = line.rstrip()

            # ── eval line ──────────────────────────────────────────────
            m = _EVAL.match(line)
            if m:
                eval_pass = int(m.group(1))
                metrics   = _extract_kv(m.group(2))
                eval_rows.setdefault(eval_pass, {}).update(metrics)
                continue

            # ── veRL native step line ───────────────────────────────────
            m = _STEP.search(line)
            if m:
                step    = int(m.group(1))
                metrics = _extract_kv(line)
                metrics.pop("step", None)   # don't double-store as a metric
                if metrics:
                    train_rows.setdefault(step, {}).update(metrics)

    train_df = pd.DataFrame(
        [{"step": s, **v} for s, v in sorted(train_rows.items())]
    ) if train_rows else pd.DataFrame()

    eval_df = pd.DataFrame(
        [{"eval_pass": p, **v} for p, v in sorted(eval_rows.items())]
    ) if eval_rows else pd.DataFrame()

    combined = pd.concat([train_df, eval_df], ignore_index=True, sort=False)
    combined.to_csv(output, index=False)
    print(f"Saved {len(train_df)} train rows, {len(eval_df)} eval rows → {output}")


if __name__ == "__main__":
    log  = sys.argv[1] if len(sys.argv) > 1 else "verl_training.log"
    out  = sys.argv[2] if len(sys.argv) > 2 else "verl_metrics.csv"
    parse_log(log, out)
