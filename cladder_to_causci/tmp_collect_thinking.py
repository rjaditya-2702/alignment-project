"""TEMPORARY — reconstruct causci_thinking.jsonl + plot from the per-sample dumps of a run that
died before the final write. Aggregates the `comp` dicts in output/sft/causci_samples_thinking_*.jsonl
(partial: only whatever passes finished, only the 30 saved samples each). Discard after use."""
import json
from pathlib import Path
import causci_eval as ce
import plot

SFT = Path("output/sft")
rows = []
for f in sorted(SFT.glob("causci_samples_thinking_*.jsonl")):
    stem = f.stem[len("causci_samples_thinking_"):]      # <label>_<off|on>
    label, mode = stem.rsplit("_", 1)
    items = [(None, r["comp"]) for r in (json.loads(l) for l in open(f)) if r.get("comp")]
    m = ce.compute_causci_metrics(items)
    m.update({"label": label, "think": mode == "on"})
    rows.append({k: v for k, v in m.items() if isinstance(v, (int, float, str, bool))})
    print(f"{label:<8} think={mode:<3} n={m['n']:>3} " +
          " ".join(f"{k}={m.get(k,0):.3f}" for k in
                   ("method_correctness", "treatment_acc", "outcome_acc", "confounder_f1")))

with open(SFT / "causci_thinking.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
plot.plot_causci_thinking()
print("NOTE: partial — base only, n=30/pass (sampled).")
