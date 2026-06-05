sample_1 = """
## Query Types

| Type | Formula | Use when |
|------|---------|----------|
| marginal | P(Y=y) | Baseline probability of an outcome, no conditions or interventions |
| correlation | P(Y=y\\|X=x) | Observing X changes probability of Y, no intervention |
| ate | E[Y\\|do(X=1)] - E[Y\\|do(X=0)] | Forcing X to a value — what is the causal effect on Y |
| backadj | Does set S block all backdoor paths X→Y? | Question asks whether adjusting for a variable set is valid |
| det-counterfactual | P(Y_x=y \\| evidence) | What would Y have been if X were different, given observed facts |
| ett | E[Y₁-Y₀ \\| X=1] | Among those who received treatment, what if they hadn't |
| nde | E[Y_{1,M₀} - Y_{0,M₀}] | Direct effect of X on Y, holding mediator at its natural value |
| nie | E[Y_{0,M₁} - Y_{0,M₀}] | Indirect effect of X on Y, only through the mediator |
| collider_bias | Does do(X) affect Y when Z is a collider? | X and Y share only a common effect, no common cause |
| exp_away | Does P(Y\\|X) change when conditioning on collider Z? | Conditioning on a common effect creates spurious association |

## Estimation Rules

- **ate — backdoor (confounders exist)**: Σ_z P(Z=z) [P(Y=1|X=1,Z=z) - P(Y=1|X=0,Z=z)]
- **ate — frontdoor (mediator, confounded treatment)**: Σ_m P(M=m|X=1) Σ_x P(X=x) P(Y=1|M=m,X=x) — same with X=0, subtract
- **ate — instrumental variable (instrument V2 exists)**: [P(Y=1|V2=1) - P(Y=1|V2=0)] / [P(X=1|V2=1) - P(X=1|V2=0)]
- **ett**: Σ_z P(Z=z|X=1) [P(Y=1|X=1,Z=z) - P(Y=1|X=0,Z=z)]
- **det-counterfactual**: (1) Abduction — infer U from evidence, (2) Action — set X=x, (3) Prediction — compute P(Y)
- **nde**: Σ_m P(M=m|X=0) [P(Y=1|X=1,M=m) - P(Y=1|X=0,M=m)]
- **nie**: Σ_m [P(M=m|X=1) - P(M=m|X=0)] P(Y=1|X=0,M=m)
- **backadj / collider_bias / exp_away**: graph analysis only — trace paths, check d-separation, no arithmetic

## Answer Interpretation

- **ate / ett / nde / nie**: compute the value. Positive → treatment increases outcome. Negative → decreases. Match to what the question asks.
- **marginal**: compare P(Y=1) to threshold or what the question asks.
- **correlation**: compare P(Y=1|X=1) vs P(Y=1|X=0).
- **det-counterfactual**: compare computed probability to prior or threshold.
- **backadj / collider_bias / exp_away**: yes or no from graph structure alone.

## Scenario

For those who are not xevo, the probability of tijv is 49%. For those who are xevo, the probability of tijv is 37%. For those who are not xevo and are not tijv, the probability of gyzp is 71%. For those who are not xevo and are tijv, the probability of gyzp is 31%. For those who are xevo and are not tijv, the probability of gyzp is 70%. For those who are xevo and are tijv, the probability of gyzp is 38%. The overall probability of xevo is 73%.
Does xevo positively affect gyzp through tijv?

## Task

Step 1 — Causal Structure: Assign short variable names (X, Y, Z, M, V1, V2, ...) to each entity in the scenario. List every directed edge as A -> B.

Step 2 — Query Type: Classify as exactly one type from the table above. One word only.

Step 3 — Estimand: Write the mathematical expression for the query. Apply backdoor / frontdoor / IV / abduction-action-prediction as needed. No numbers yet.

Step 4 — Compute: Substitute every numeric value from the scenario into the estimand. Show each arithmetic step explicitly. End with the final number. For backadj / collider_bias / exp_away, trace the graph paths and state your conclusion.

Step 5 - Answer: Based on the above inference performed and the question asked in the scenario, answer yes or no.

Then output this JSON and nothing else:

{{
  "step1": "<variable assignments and all directed edges>",
  "step2": "<query type>",
  "step3": "<estimand expression>",
  "step4": "<full arithmetic or graph reasoning, final value at the end>",
  "step5": "<yes or no>"
}}
"""

sample_2 = """## Study Description
This dataset comes from an observational study of adult patients treated in a healthcare system for a common medical condition. It records patient demographics, clinical status at the start of care, and whether they received a care coordinator to help manage their treatment. The outcome indicates whether patients recovered within 30 days.

## Dataset
Path: dataset/synthetic_causci/glm_49.csv
Shape: 575 rows, 8 columns

Columns and types:
  patient_age: int64
  baseline_blood_pressure: int64
  symptom_severity_score: int64
  body_mass_index: int64
  has_chronic_condition: int64
  is_smoker: int64
  received_care_coordinator: int64
  recovered_within_30_days: int64

First 5 rows:
   patient_age  baseline_blood_pressure  symptom_severity_score  body_mass_index  has_chronic_condition  is_smoker  received_care_coordinator  recovered_within_30_days
0           23                       24                       7               10                      1          0                          0                         0
1           40                       17                      15                8                      0          1                          0                         1
2           38                       27                       7                7                      1          1                          1                         1
3           29                        8                      12               10                      0          1                          1                         1
4           41                       20                       6               11                      1          1                          0                         1

Summary statistics:
       patient_age  baseline_blood_pressure  symptom_severity_score  body_mass_index  has_chronic_condition   is_smoker  received_care_coordinator  recovered_within_30_days
count   575.000000               575.000000              575.000000       575.000000             575.000000  575.000000                 575.000000                575.000000
mean     29.187826                20.041739                9.608696        13.714783               0.641739    0.638261                   0.492174                  0.499130
std       8.217398                 6.019567                3.096293         4.669798               0.479907    0.480922                   0.500374                  0.500435
min       6.000000                 4.000000                0.000000        -2.000000               0.000000    0.000000                   0.000000                  0.000000
25%      23.000000                16.000000                7.000000        11.000000               0.000000    0.000000                   0.000000                  0.000000
50%      29.000000                20.000000               10.000000        14.000000               1.000000    1.000000                   0.000000                  0.000000
75%      34.000000                24.000000               12.000000        17.000000               1.000000    1.000000                   1.000000                  1.000000
max      53.000000                40.000000               19.000000        29.000000               1.000000    1.000000                   1.000000                  1.000000

Missing values per column:
  None

Low-cardinality columns (≤10 unique values):
  has_chronic_condition: [0, 1]
  is_smoker: [0, 1]
  received_care_coordinator: [0, 1]
  recovered_within_30_days: [0, 1]

## Question
Did assigning a care coordinator to adult patients help more of them recover within 30 days?

---

## Method Reference

| Method | Use when |
|--------|----------|
| diff_in_means | RCT with enforced compliance. Groups comparable by design. No confounding. |
| ols | Observational. All confounders observed and included. No unobserved confounding. |
| ipw | Observational. Confounders observed. Reweight by propensity score. Needs overlap: 0 < e(X) < 1. |
| matching | Observational. Confounders observed. Use when propensity score overlap is poor. |
| did | Panel data. Treatment introduced at one point in time to one group. Time variable must be treatment timing, not a covariate. Parallel trends must hold. |
| rdd | Treatment assigned by a running variable crossing a known cutoff. Units just above and below cutoff are comparable. |
| iv | Unobserved confounders exist. Valid instrument available — correlated with treatment, affects outcome only through treatment. |
| frontdoor | Unobserved confounders exist. Full mediator pathway T→M→Y with no unobserved T→M or M→Y confounding. |
| glm | Binary outcome (logistic) or count outcome (Poisson). Confounders observed. |

## Estimand Reference

| Method | Estimand |
|--------|----------|
| diff_in_means | ATE |
| ols | ATE |
| ipw | ATE, ATT, or ATC — based on whether question asks about population, treated group, or control group |
| matching | ATE or ATT |
| did | ATT |
| rdd | Local ATE at the cutoff |
| iv | LATE |
| frontdoor | ATE |
| glm | Conditional effect (log-odds for binary, incidence rate ratio for counts) |

---

Think through the following before answering:
- Was treatment randomly assigned or self-selected?
- Are confounders observed or unobserved?
- Is there a time variable marking treatment timing (not just a covariate)?
- Is there a continuous running variable with a cutoff?
- Is there a variable that affects treatment but not outcome directly?
- Is the outcome binary, count, or continuous?
- Does the question ask about the full population (ATE), treated units (ATT), or local effect (LATE)?

Then output this JSON and nothing else after your thinking:

{{
  "step1": {{
    "treatment": "<exact column name>",
    "outcome": "<exact column name>",
    "controls": ["<col1>", "<col2>"],
    "instrument": null,
    "running_variable": null,
    "cutoff": null,
    "time_variable": null,
    "group_variable": null,
    "mediator": null,
    "estimand": "<ATE, ATT, ATC, LATE, or conditional>"
  }},
  "step2": "<method name>"
}}
"""

import torch
import shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── config ────────────────────────────────────────────────────────────────────
STEP            = 500
BASE_MODEL_NAME = "Qwen/Qwen3-8B"
CKPT_ROOT       = Path("src/output_RL/verl_checkpoints")
EXPORT_DIR      = Path("exported_models")
WEIGHT_EXTS     = {".bin", ".safetensors", ".h5", ".msgpack"}
# ─────────────────────────────────────────────────────────────────────────────


def has_weights(directory: Path) -> bool:
    return directory.exists() and any(
        f.suffix in WEIGHT_EXTS for f in directory.rglob("*") if f.is_file()
    )


def merge_lora_from_hub(base_model_name: str, actor_dir: Path, output_path: Path) -> None:
    hf_dir   = actor_dir / "huggingface"   # tokenizer + config
    lora_dir = actor_dir / "lora_adapter"  # adapter_config.json + adapter_model.safetensors

    print(f"  Loading base model '{base_model_name}' from Hub...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    # use checkpoint tokenizer (has any special tokens added during training)
    print(f"  Loading tokenizer from checkpoint {hf_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(hf_dir))

    print(f"  Applying LoRA adapter from {lora_dir}...")
    model = PeftModel.from_pretrained(model, str(lora_dir))

    print("  Merging and unloading LoRA weights...")
    model = model.merge_and_unload()

    print(f"  Saving merged model to {output_path}...")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))


def merge_fsdp_shards(actor_dir: Path, output_path: Path, base_model_name: str) -> None:
    shard_files = sorted(actor_dir.glob("model_world_size_*_rank_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {actor_dir}")
    print(f"  Found {len(shard_files)} shards: {[f.name for f in shard_files]}")

    merged: dict = {}
    for shard_file in shard_files:
        print(f"  Loading {shard_file.name}...")
        shard = torch.load(shard_file, map_location="cpu", weights_only=True)
        if isinstance(shard, dict) and "model" in shard:
            shard = shard["model"]
        for k, v in shard.items():
            if k not in merged:
                merged[k] = v
    print(f"  Merged {len(merged)} parameter tensors.")

    # strip common FSDP prefixes
    prefixes = ["_fsdp_wrapped_module.", "module.", "_orig_mod."]
    cleaned: dict = {}
    for k, v in merged.items():
        for pfx in prefixes:
            if k.startswith(pfx):
                k = k[len(pfx):]
        cleaned[k] = v

    print(f"  Loading base architecture from '{base_model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  [WARN] {len(missing)} missing keys  — first 5: {missing[:5]}")
    if unexpected:
        print(f"  [WARN] {len(unexpected)} unexpected keys — first 5: {unexpected[:5]}")

    print(f"  Saving to {output_path}...")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))


def export_checkpoint(step: int) -> Path:
    actor_dir  = CKPT_ROOT / f"global_step_{step}" / "actor"
    hf_dir     = actor_dir / "huggingface"
    lora_dir   = actor_dir / "lora_adapter"

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    if has_weights(hf_dir):
        print(f"[INFO] Complete HF model found at {hf_dir}")
        if lora_dir.exists() and any(lora_dir.iterdir()):
            print("[INFO] LoRA adapter also present — merging into HF weights...")
            model = AutoModelForCausalLM.from_pretrained(str(hf_dir), dtype=torch.bfloat16, device_map="cuda")
            tokenizer = AutoTokenizer.from_pretrained(str(hf_dir))
            model = PeftModel.from_pretrained(model, str(lora_dir))
            model = model.merge_and_unload()
            model.save_pretrained(str(EXPORT_DIR))
            tokenizer.save_pretrained(str(EXPORT_DIR))
        else:
            print(f"[INFO] Copying HF model to {EXPORT_DIR}...")
            shutil.copytree(hf_dir, EXPORT_DIR, dirs_exist_ok=True)

    elif lora_dir.exists() and any(lora_dir.iterdir()):
        print(f"[INFO] No HF weights found — loading base from Hub and merging LoRA...")
        merge_lora_from_hub(BASE_MODEL_NAME, actor_dir, EXPORT_DIR)

    else:
        print("[INFO] No HF weights or LoRA found — merging FSDP shards...")
        merge_fsdp_shards(actor_dir, EXPORT_DIR, BASE_MODEL_NAME)

    print(f"\n[DONE] Model exported to: {EXPORT_DIR}")
    return EXPORT_DIR

system = """You are a causal inference expert. Analyze the study design carefully before selecting variables and methods. Think through your reasoning, then output only the JSON."""
def run_inference(model, tokenizer, prompt: str, enable_thinking: bool = True) -> str:
    messages = [
        {"role": "system",    "content": system},
        {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,
            temperature=0.7,
            top_p=0.95,
            top_k=10,
            do_sample=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model_path = export_checkpoint(STEP)

    print("\n[INFO] Loading exported model for inference...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()

    # ── sample queries ────────────────────────────────────────────────────────

    for i, prompt in enumerate([sample_1, sample_2], 1):
        print(f"\n{'='*80}\nSample {i}\n{'='*80}")
        print(run_inference(model, tokenizer, prompt))