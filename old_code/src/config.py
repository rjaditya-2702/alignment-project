# Central config — all models, hyperparameters, and paths live here.
# Change anything here and every script picks it up automatically.

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── Models ────────────────────────────────────────────────────────────────────

POLICY_MODEL = "Qwen/Qwen3-8B"  # base model for training
JUDGE_MODEL  = "Qwen/Qwen3-8B"

# ── Paths ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR_RL   = ROOT / "src" / "output_RL"
CHECKPOINT_DIR = ROOT / "src" / "output_RL" / "checkpoints"
FINAL_MODEL     = ROOT / "src" / "output_RL" / "checkpoints" / "final"  # eval loads from here
PLOT_DIR       = ROOT / "src" / "output_RL" / "plots"

TRAIN_DATA_SFT_LORA     = ROOT / "src" / "output_fine_tune_lora" / "train.jsonl"
TEST_DATA_SFT_LORA      = ROOT / "src" / "output_fine_tune_lora" / "test.jsonl"
OUTPUT_DIR_SFT_LORA = ROOT / "src" / "output_fine_tune_lora"
SFT_LORA_OUTPUT_DIR = ROOT / "src" / "output_fine_tune_lora"
SFT_LORA_PLOT_DIR   = ROOT / "src" / "output_fine_tune_lora" / "plots"
SFT_LORA_CHECKPOINT_DIR = ROOT / "src" / "output_fine_tune_lora" / "checkpoints"

TRAIN_DATA_SFT     = ROOT / "src" / "output_fine_tune" / "train.jsonl"
TEST_DATA_SFT      = ROOT / "src" / "output_fine_tune" / "test.jsonl"
OUTPUT_DIR_SFT   = ROOT / "src" / "output_fine_tune"
SFT_OUTPUT_DIR = ROOT / "src" / "output_fine_tune"
SFT_PLOT_DIR   = ROOT / "src" / "output_fine_tune" / "plots"
SFT_CHECKPOINT_DIR = ROOT / "src" / "output_fine_tune" / "checkpoints"

# ── SFT / RL data files (per source) ───────────────────────────────────────────
# TRAIN is split disjointly: preprocess(which="sft") writes 30% of each source to
# the train_sft_* files; preprocess(which="rl") writes the remaining 70% to the
# train_rl_* files. TEST is NOT split — it is the full original benchmark, written
# identically (per method's prompt) as test_cladder/test_causci in each dir.
# Load one source or both (merge) per preference.

TRAIN_SFT_CLADDER = SFT_LORA_OUTPUT_DIR / "train_sft_cladder.jsonl"
TRAIN_SFT_CAUSCI  = SFT_LORA_OUTPUT_DIR / "train_sft_causci.jsonl"
TEST_SFT_CLADDER  = SFT_LORA_OUTPUT_DIR / "test_cladder.jsonl"
TEST_SFT_CAUSCI   = SFT_LORA_OUTPUT_DIR / "test_causci.jsonl"

TRAIN_RL_CLADDER  = OUTPUT_DIR_RL / "train_rl_cladder.jsonl"
TRAIN_RL_CAUSCI   = OUTPUT_DIR_RL / "train_rl_causci.jsonl"
TEST_RL_CLADDER   = OUTPUT_DIR_RL / "test_cladder.jsonl"
TEST_RL_CAUSCI    = OUTPUT_DIR_RL / "test_causci.jsonl"



# ── Training hyperparameters ──────────────────────────────────────────────────

TRAIN_BATCH_SIZE = 1     # prompts per training step
N_ROLLOUTS       = 3       # completions per prompt
MAX_PROMPT_LEN   = 4000    # truncate prompt to this many tokens
TRAIN_MAX_TOKENS = 1024    # max completion length during training
TEMPERATURE      = 0.8
TOP_P            = 0.9

BETA             = 0.01    # KL coefficient
LR               = 2e-5
WEIGHT_DECAY     = 0.01
GRAD_ACCUM       = 1       # optimizer step every N steps
MAX_GRAD_NORM    = 1.0

MAX_EPOCHS       = 3
SAVE_EVERY       = 500     # global steps between checkpoints
LOG_EVERY        = 10      # global steps between log lines

LORA_R           = 16
IS_CLIP_RANGE    = (0.5, 2.0)   # importance sampling ratio clamp (min, max)

# ── Eval hyperparameters ──────────────────────────────────────────────────────

EVAL_BATCH_SIZE  = 4
EVAL_MAX_TOKENS  = 4096
