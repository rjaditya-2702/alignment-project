"""demo.py — generate reasoning for the first N synth records and print for review. Writes nothing.

Real API call (Sonnet 4.6 → Opus 4.8 fallback). Run where there's internet + OPENROUTER_API_KEY:
  OPENROUTER_API_KEY=... python3 demo.py            # 5 samples
  OPENROUTER_API_KEY=... python3 demo.py 3
"""

import sys

import build_data as b

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
for i, r in enumerate(b.load_synth()[:n], 1):
    cols = b.csv_columns(r["csv_path"])
    gold = b.gold_output({"step1": r["step1"], "step2": r["method"]})
    reasoning = b.generate_reasoning(r["description"], cols, r["query"], gold, r["step1"])
    print("=" * 90)
    print(f"[{i}] {r['id']}  (gold method={r['method']})")
    print("QUESTION:", r["query"])
    print("COLUMNS :", ", ".join(cols))
    print("\nGOLD ANSWER:\n" + gold)
    print("\nTEACHER REASONING:\n" + reasoning)
    print()
