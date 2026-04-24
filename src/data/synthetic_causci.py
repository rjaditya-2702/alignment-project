"""
Generate CauSciBench synthetic training instances.
Uses causci_bench generators for data, OpenAI GPT for realistic context.
Methods: diff_in_means, ols, ipw, matching, did, rdd, iv, frontdoor, glm
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from openai import OpenAI
from causci_bench.synthetic.generation.generator import (
    RCTGenerator, PSMGenerator, IVGenerator,
    DiDGenerator, RDDGenerator, FrontDoorGenerator,
)
from causci_bench.synthetic.context.prompts import generate_data_summary
from data import CAUSCIBENCH_PROMPT

OUTPUT_DIR = Path("dataset/synthetic_causci")

DOMAINS = ["education", "healthcare", "labor", "policy"]

METHOD_TO_CAUSCI_TYPE = {
    "diff_in_means": "rct",
    "ols":           "observational",
    "ipw":           "observational",
    "matching":      "observational",
    "iv":            "IV",
    "did":           "did_canonical",
    "rdd":           "rdd",
    "frontdoor":     "frontdoor",
    "glm":           "observational",
}

# Covariate mean/variance pools from generate_synthetic.py
OBS_MEAN = np.array([[28,22,8,15,3],[30,20,10,14,4],[25,24,6,16,2],[32,18,12,13,5]])
OBS_COV  = np.array([[81,25,7,16,2],[64,36,9,25,3],[100,16,5,20,1],[49,30,8,18,4]])
IV_MEAN  = np.array([[13,28,10,12],[15,25,12,10],[10,30,8,14],[12,26,9,11]])
IV_COV   = np.array([[7,9,8,8],[5,7,6,9],[8,5,10,7],[6,12,7,5]])
RDD_MEAN = np.array([[10,5],[12,4],[8,6],[14,3]])
RDD_COV  = np.array([[8,4],[5,6],[10,2],[6,8]])


def _sample_params(seed, method):
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, 4))
    tau = float(rng.uniform(1.0, 10.0))

    if method == "iv":
        mean_pool, cov_pool = IV_MEAN, IV_COV
        max_cont = 4
    elif method == "rdd":
        mean_pool, cov_pool = RDD_MEAN, RDD_COV
        max_cont = 2
    else:
        mean_pool, cov_pool = OBS_MEAN, OBS_COV
        max_cont = 4

    n_cont = int(rng.integers(2, max_cont + 1))
    n_bin  = int(rng.integers(2, 5))
    n_obs  = int(rng.integers(300, 801))
    cutoff = int(rng.integers(5, 26))

    mean_vec = mean_pool[idx % len(mean_pool)][:n_cont]
    cov_mat  = np.diag(cov_pool[idx % len(cov_pool)][:n_cont])

    return dict(tau=tau, n_cont=n_cont, n_bin=n_bin, n_obs=n_obs,
                cutoff=cutoff, mean=mean_vec, cov=cov_mat)


def _make_generator(method, params, seed):
    kw = dict(
        n_observations=params['n_obs'],
        n_continuous_covars=params['n_cont'],
        n_binary_covars=params['n_bin'],
        mean=params['mean'],
        covar=params['cov'],
        true_effect=params['tau'],
        seed=seed,
    )
    if method == "diff_in_means":
        return RCTGenerator(**kw)
    elif method in ("ols", "ipw", "matching", "glm"):
        return PSMGenerator(**kw)
    elif method == "iv":
        return IVGenerator(**kw)
    elif method == "did":
        return DiDGenerator(**kw, n_periods=2)
    elif method == "rdd":
        return RDDGenerator(**kw, cutoff=params['cutoff'], plot=False)
    elif method == "frontdoor":
        return FrontDoorGenerator(**kw)


def _get_data(gen):
    gen.generate_data()
    return gen.data


def _binarize_y(df):
    df = df.copy()
    df["Y"] = (df["Y"] > df["Y"].median()).astype(int)
    return df


def _call_oai(df, n_cont, n_bin, causci_type, domain, history, cutoff=None):
    summary = generate_data_summary(df, n_cont, n_bin, causci_type, cutoff=cutoff)

    method_display = {
        "rct":          "Randomized Control Trial",
        "observational":"Observational Study",
        "IV":           "Instrumental Variables",
        "did_canonical":"Difference-in-Differences",
        "rdd":          "Regression Discontinuity Design",
        "frontdoor":    "Front-Door Causal Inference",
    }
    domain_guides = {
        "education": "Student performance, school interventions, socioeconomic factors.",
        "healthcare": "Treatments, diagnoses, recovery outcomes, patient demographics.",
        "labor":     "Wages, training programs, employment history, job markets.",
        "policy":    "Program rollouts, regional effects, housing, crime, public health.",
    }
    special = {
        "IV":           "Z is the instrument variable (affects outcome only through treatment).",
        "did_canonical":"post=1 means post-treatment period. D=1 means treated group. Keep 'post' and 'unit_id' as-is.",
        "rdd":          "running_X is the running variable; units above the cutoff receive treatment.",
        "frontdoor":    "M is the mediator on the causal path from D to Y.",
    }

    prompt = f"""You are generating a realistic dataset context for a {method_display.get(causci_type, causci_type)} study in the {domain} domain.

Dataset summary (generic variable names):
{summary}
{special.get(causci_type, '')}

Previously used contexts (avoid duplication):
{history[-800:]}

Domain: {domain_guides.get(domain, '')}

Tasks:
1. Rename generic variables (X1, X2, ..., D, Y, and Z/M/running_X if present) to realistic snake_case names for the {domain} domain. For did_canonical, keep 'post' and 'unit_id' unchanged.
2. Write a 2-3 sentence background description of this dataset.
3. Write a natural-language causal question. No statistical jargon. No variable names. Written as if for a policy report.

Return ONLY a valid JSON object (no markdown):
{{
  "variable_labels": {{"X1": "realistic_name", "D": "treatment_name", "Y": "outcome_name"}},
  "description": "...",
  "query": "..."
}}"""

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        max_completion_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)


def _build_prompt(df, description, query, csv_path):
    columns_and_types = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
    missing = df.isnull().sum()
    missing_str = "\n".join(f"  {c}: {v}" for c, v in missing.items() if v > 0) or "  none"
    return CAUSCIBENCH_PROMPT.format(
        dataset_description=description,
        file_path=str(csv_path.resolve()),
        columns_and_types=columns_and_types,
        df_head=df.head().to_string(index=False),
        df_describe=df.describe().to_string(),
        missing_values=missing_str,
        query=query,
    )


def _make_row(idx, method, df, tau, csv_path, description, query, variable_labels):
    treatment   = variable_labels.get("D", "D")
    outcome     = variable_labels.get("Y", "Y")
    instrument  = variable_labels.get("Z")  if method == "iv"        else None
    running_var = variable_labels.get("running_X") if method == "rdd" else None
    mediator    = variable_labels.get("M")  if method == "frontdoor"  else None

    skip = {treatment, outcome}
    if instrument:  skip.add(instrument)
    if running_var: skip.add(running_var)
    if mediator:    skip.add(mediator)
    if method == "did":
        skip.update({"post", "unit_id"})

    controls = [c for c in df.columns if c not in skip]

    step1 = {
        "treatment":        treatment,
        "outcome":          outcome,
        "controls":         controls,
        "instrument":       instrument,
        "running_variable": running_var,
        "time_variable":    "post"      if method == "did" else None,
        "group_variable":   treatment   if method == "did" else None,
        "mediator":         mediator,
    }

    return {
        "id":         f"causci_new_synth_{method}_{idx}",
        "source":     "causcibench_synthetic",
        "prompt":     _build_prompt(df, description, query, csv_path),
        "label":      tau,
        "label_type": "continuous",
        "groundtruth": {
            "step1": step1,
            "step2": method,
            "step3": None,
            "step4": None,
            "step5": tau,
        },
    }


METHODS = [
    "diff_in_means", "ols", "ipw", "matching",
    "iv", "did", "rdd", "frontdoor", "glm",
]


def generate_causci_synthetic(n_per_method: int = 200, seed: int = 42) -> list[dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    history = ""

    for method in METHODS:
        causci_type = METHOD_TO_CAUSCI_TYPE[method]
        rows = []

        for i in range(n_per_method):
            s = (seed * 1000 + abs(hash(method)) % 10000 + i) % (2 ** 31)
            domain = DOMAINS[i % len(DOMAINS)]

            params = _sample_params(s, method)
            gen    = _make_generator(method, params, s)
            df     = _get_data(gen)

            if method == "glm":
                df = _binarize_y(df)

            cutoff = params['cutoff'] if method == "rdd" else None
            ctx = _call_oai(df, params['n_cont'], params['n_bin'],
                               causci_type, domain, history, cutoff=cutoff)

            variable_labels = ctx["variable_labels"]
            description     = ctx["description"]
            query           = ctx["query"]

            df_renamed = df.rename(columns={k: v for k, v in variable_labels.items() if k in df.columns})

            csv_path = OUTPUT_DIR / f"{method}_{i}.csv"
            df_renamed.to_csv(csv_path, index=False)

            history += description[:100] + "\n"

            rows.append(_make_row(i, method, df_renamed, params['tau'],
                                  csv_path, description, query, variable_labels))

        print(f"  {method}: {len(rows)}")
        all_rows.extend(rows)

    return all_rows


if __name__ == "__main__":
    rows = generate_causci_synthetic(n_per_method=2, seed=42)
    print(f"\nTotal: {len(rows)}")
    ex = rows[0]
    print(f"id:     {ex['id']}")
    print(f"label:  {ex['label']:.4f}")
    print(f"step1:  {ex['groundtruth']['step1']}")
    print(f"query:  {ex['groundtruth']['step1']}")
    print(f"prompt[:300]:\n{ex['prompt'][:300]}")
