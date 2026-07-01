# Central config — paths, models, knobs. Everything reads from here.

from pathlib import Path

NEW_CODE = Path(__file__).resolve().parent
PROJECT  = NEW_CODE.parent

# ── Inputs (read-only) ──────────────────────────────────────────────────
DATA_DIR = PROJECT / "data"
CSV_DIR  = DATA_DIR / "csv_files"
META_DIR = DATA_DIR / "metadata_json"

# split key -> (metadata json, csv subdir).  train = qr + synthetic, test = real.
SPLITS = {
    "qr":        ("qr_input.json",        "qrdata"),
    "synthetic": ("synthetic_input.json", "synthetic_data"),
    "real":      ("real_input.json",      "realdata"),
}
TRAIN_SPLITS = ("qr", "synthetic")
TEST_SPLIT   = "real"

# ── Outputs ─────────────────────────────────────────────────────────────
OUT = NEW_CODE / "output"

# SFT and RL use different causci prompts, so each phase bakes its own train + test
# (test = the same realdata rows, just rendered with that phase's prompt).
TRAIN_SFT_JSONL = OUT / "train_sft.jsonl"   # SFT_FRACTION of qr+synth, SFT prompt
TEST_SFT_JSONL  = OUT / "test_sft.jsonl"    # full realdata, SFT prompt
TRAIN_RL_JSONL  = OUT / "train_rl.jsonl"    # the rest of qr+synth, RL prompt
TEST_RL_JSONL   = OUT / "test_rl.jsonl"     # full realdata, RL prompt

TRAIN_RL_PARQUET = OUT / "train_rl.parquet"
TEST_PARQUET     = OUT / "test.parquet"

SFT_CKPT = OUT / "sft" / "final"   # SFT writes here; RL loads it as its base model
RL_CKPT  = OUT / "rl"  / "final"

# eval-metric timelines (eval_pass + causci/* columns), one per phase
SFT_METRICS_CSV = OUT / "sft_metrics.csv"
RL_METRICS_CSV  = OUT / "rl_metrics.csv"
PLOT_DIR        = OUT / "plots"

# ── Models ──────────────────────────────────────────────────────────────
POLICY_MODEL  = "Qwen/Qwen3-8B"   # SFT base
RL_BASE_MODEL = SFT_CKPT          # RL continues from the SFT checkpoint
JUDGE_MODEL   = "Qwen/Qwen3-8B"

# ── Split ───────────────────────────────────────────────────────────────
SFT_FRACTION = 0.35   # rest (0.65) → RL; disjoint, assigned by stable id hash

# ── Eval ────────────────────────────────────────────────────────────────
EFFECT_TOL      = 0.05   # |pred - ref| / |ref| <= tol counts as a correct effect
EVAL_MAX_TOKENS = 4096

# ── SFT training (phase 1) ──────────────────────────────────────────────
SFT_MAX_SEQ_LEN  = 4096
SFT_LR           = 2e-5
SFT_EPOCHS       = 3
SFT_BATCH_SIZE   = 1     # per-device
SFT_GRAD_ACCUM   = 2     # small train set (~61 rows) → keep grad_accum low so there are enough updates
SFT_LORA_R       = 16
SFT_EVAL_EVERY   = 1     # eval on the test set every training step (dense SFT→RL timeline)
SFT_EVAL_N       = 64    # cap test rows per eval pass for speed (None = full 175-row test set)
SFT_EVAL_MAX_NEW = 512   # max new tokens at eval (no-think JSON output is short)
