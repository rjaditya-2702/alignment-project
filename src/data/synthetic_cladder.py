"""
Generate CLadder synthetic data using causalbenchmark generator.
Uses RandomBuilder to sample random SCM parameters per story.
"""

from pathlib import Path

from original_data.Cladder.cladder.causalbenchmark.generator import generate_questions
from original_data.Cladder.cladder.causalbenchmark.graphs.builders import RandomBuilder
from original_data.Cladder.cladder.causalbenchmark.queries import create_query
from original_data.Cladder.cladder.causalbenchmark import util
from data import CLADDER_PROMPT

STORY_ROOT = Path(util.story_root())

# Registered query names from config/queries/reg.yml
QUERY_NAMES = ['ate', 'ett', 'nie', 'nde', 'marginal', 'correlation',
               'backadj', 'collider_bias', 'exp_away']


def _entry_to_unified(entry: dict, idx: int) -> dict:
    r    = entry.get("reasoning") or {}
    meta = entry.get("meta", {})

    problem = f"{entry.get('given_info', '')}\n{entry.get('question', '')}".strip()
    step1   = (r.get("step0", "") + "\n" + r.get("step1", "")).strip()
    step4   = "\n".join(
        p for p in [r.get("step2"), r.get("step3"), r.get("step4"), r.get("step5"), r.get("end")]
        if p
    )

    story_id = meta.get("story_id", "")
    graph_id = meta.get("graph_id", "")

    return {
        "id":         f"cladder_synth_{story_id}_{graph_id}_{idx}",
        "source":     "cladder_synthetic",
        "prompt":     CLADDER_PROMPT.format(problem=problem),
        "label":      entry["answer"],
        "label_type": "binary",
        "groundtruth": {
            "step1": step1,
            "step2": meta.get("query_type"),
            "step3": meta.get("formal_form"),
            "step4": step4,
            "step5": entry["answer"],
        },
    }


def load_cladder_synthetic(spec_limit: int = 50, seed: int = 42) -> list[dict]:
    story_ids = [p.stem for p in sorted(STORY_ROOT.glob("*.yml"))]
    builder   = RandomBuilder(seed=seed)
    queries   = [create_query(name) for name in QUERY_NAMES]
    rows      = []

    for story_id in story_ids:
        count = 0
        for entry in generate_questions(
            story_id, builder,
            transformation=None,
            queries=queries,
            spec_limit=spec_limit,
            include_reasoning=True,
            pbar=False,
        ):
            rows.append(_entry_to_unified(entry, len(rows)))
            count += 1
        print(f"  {story_id}: {count}")

    return rows


if __name__ == "__main__":
    rows = load_cladder_synthetic(spec_limit=2, seed=42)
    print(f"\nTotal CLadder synthetic: {len(rows)}")
    ex = rows[0]
    print(f"id:     {ex['id']}")
    print(f"label:  {ex['label']}")
    print(f"gt step2: {ex['groundtruth']['step2']}")
    print(f"gt step3: {ex['groundtruth']['step3']}")
    print(f"prompt[:300]:\n{ex['prompt'][:300]}")
