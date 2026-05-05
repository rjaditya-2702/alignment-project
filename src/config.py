# Central config — change POLICY_MODEL here and train.py + eval.py both pick it up.
# All other hyperparameters live here too.

# ── Models ────────────────────────────────────────────────────────────────────

POLICY_MODEL = "Qwen/Qwen3-14B"
JUDGE_MODEL  = "deepseek-ai/deepseek-math-7b-instruct"

# ── Training hyperparameters ──────────────────────────────────────────────────

TRAIN_BATCH_SIZE = 1       # prompts per training step
N_ROLLOUTS       = 8       # completions per prompt
MAX_PROMPT_LEN   = 3072    # truncate prompt to this many tokens
TRAIN_MAX_TOKENS = 2048    # max completion length during training
TEMPERATURE      = 0.8
TOP_P            = 0.9

BETA             = 0.01    # KL coefficient
LR               = 2e-5
WEIGHT_DECAY     = 0.01
GRAD_ACCUM       = 8       # optimizer step every N steps
MAX_GRAD_NORM    = 1.0

MAX_EPOCHS       = 3
SAVE_EVERY       = 500     # global steps between checkpoints
LOG_EVERY        = 10      # global steps between log lines

LORA_R           = 32

# ── Eval hyperparameters ──────────────────────────────────────────────────────

EVAL_BATCH_SIZE  = 4
EVAL_MAX_TOKENS  = 4096
