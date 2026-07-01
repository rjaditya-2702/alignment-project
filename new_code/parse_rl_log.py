"""Parse [verl_eval] lines from the veRL training log into rl_metrics.csv —
one row per eval pass (eval_pass + causci/* columns), the format plot.py expects
(same shape train_sft.py writes for sft_metrics.csv).

Run:  python parse_rl_log.py <log> <out.csv>
"""

import csv
import re
import sys

KEYS = ["causci/method_acc", "causci/treatment_acc", "causci/outcome_acc",
        "causci/control_acc", "causci/effect_acc", "causci/mre"]

_KV   = re.compile(r"([\w/]+):\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_EVAL = re.compile(r"\[verl_eval\]\s+eval_pass:(\d+)\s+(.*)")


def main(log, out):
    rows = {}
    for line in open(log, errors="ignore"):
        m = _EVAL.search(line)
        if not m:
            continue
        kv = {k: float(v) for k, v in _KV.findall(m.group(2)) if k in KEYS}
        if not kv:                       # skip the per-call lines (reward=…, no causci metrics)
            continue
        rows.setdefault(int(m.group(1)), {}).update(kv)

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eval_pass"] + KEYS)
        for ep in sorted(rows):
            w.writerow([ep] + [rows[ep].get(k, "") for k in KEYS])
    print(f"wrote {len(rows)} eval rows → {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "verl_training.log",
         sys.argv[2] if len(sys.argv) > 2 else "output/rl_metrics.csv")
