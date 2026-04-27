"""
GRPO post-training for causal alignment.

Policy:    Qwen3-14B + LoRA (r=16) — trainable
Reference: same base weights with adapters disabled — frozen

Algorithm:
  For each prompt, generate N rollouts from the policy.
  Score each rollout with the reward functions (subprocess sandbox).
  Normalize rewards within the group: â = (r - mean) / std.
  GRPO loss = -mean(â * per-token-logprob) + β * KL(policy || ref)
  where KL is computed per token from logprobs (no ratio clipping —
  single-step online update so ratio = 1 exactly at update time).

Usage:
    python src/training/train.py
    python src/training/train.py --model Qwen/Qwen3-14B --n-rollouts 8 --epochs 2
    python src/training/train.py --resume output/checkpoints/step_500
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from src.training.reward import compute_rewards

# ── Hyperparameters ───────────────────────────────────────────────────────────

DEFAULT_MODEL   = "Qwen/Qwen3-14B"
TRAIN_DATA      = ROOT / "output" / "train.jsonl"
OUTPUT_DIR      = ROOT / "output" / "checkpoints"

N_ROLLOUTS      = 8       # completions per prompt
MAX_PROMPT_LEN  = 3072    # truncate prompt to this many tokens
MAX_NEW_TOKENS  = 2048    # max completion length
TEMPERATURE     = 0.8
TOP_P           = 0.9

BETA            = 0.01    # KL coefficient
LR              = 2e-5
WEIGHT_DECAY    = 0.01
GRAD_ACCUM      = 8       # optimizer step every N prompts
MAX_GRAD_NORM   = 1.0

MAX_EPOCHS      = 3
SAVE_EVERY      = 500     # global steps between checkpoints
LOG_EVERY       = 10      # global steps between log lines
SANDBOX_WORKERS = 8

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=1024,
    lora_alpha=2048,
    lora_dropout=0.02,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.bfloat16,
    bnb_8bit_use_double_quant=True,
)

# QUANT_CONFIG = None


# ── Model loading ─────────────────────────────────────────────────────────────

def load_policy(model_name: str):
    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if "<|endoftext|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|endoftext|>"
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading model from {model_name} → {device}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=torch.bfloat16,
        quantization_config=None, # i will replace it with QUANT_CONFIG after testing
        trust_remote_code=True,
    ).to(device)

    # resize embeddings if we added a new token
    if len(tokenizer) > base.config.vocab_size:
        base.resize_token_embeddings(len(tokenizer))

    model = get_peft_model(base, LORA_CONFIG)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    return model, tokenizer


# ── Chat formatting ───────────────────────────────────────────────────────────

def format_prompt(tokenizer, prompt: str) -> str:
    """Wrap raw prompt in Qwen3 chat template with thinking disabled."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ── Generation ────────────────────────────────────────────────────────────────

def generate_rollouts(model, tokenizer, prompt: str, n: int, device: str) -> list[str]:
    """Generate n completions for one prompt. No gradient."""
    formatted = format_prompt(tokenizer, prompt)
    enc = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_LEN,
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_return_sequences=n,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = enc["input_ids"].shape[1]
    return [
        tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        for out in out_ids
    ]


# ── Logprob computation ───────────────────────────────────────────────────────

def sequence_logprob(model, prompt_ids: torch.Tensor, comp_ids: torch.Tensor) -> torch.Tensor:
    """
    Mean per-token log prob of comp_ids given prompt_ids.
    prompt_ids: [P]   (1D, on device) — must be tokenized from format_prompt() output
    comp_ids:   [C]   (1D, on device)
    Returns scalar tensor.
    """
    full_ids = torch.cat([prompt_ids, comp_ids]).unsqueeze(0)
    attn_mask = torch.ones_like(full_ids)

    outputs = model(input_ids=full_ids, attention_mask=attn_mask)
    logits = outputs.logits[0]

    log_probs = F.log_softmax(logits[:-1], dim=-1)
    labels    = full_ids[0, 1:]
    token_lp  = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)

    P, C = prompt_ids.shape[0], comp_ids.shape[0]
    comp_lp = token_lp[P - 1 : P - 1 + C]

    return comp_lp.mean()


# ── GRPO loss ─────────────────────────────────────────────────────────────────

def grpo_loss(
    policy_lps: torch.Tensor,   # [N]
    ref_lps: torch.Tensor,      # [N], detached
    rewards: torch.Tensor,      # [N]
    beta: float = BETA,
) -> torch.Tensor:
    # Group-normalize rewards → advantages
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)   # [N]

    # KL(policy || ref) per completion, per-token mean
    kl = policy_lps - ref_lps                                    # [N]

    # Policy gradient + KL penalty
    return -(adv * policy_lps).mean() + beta * kl.mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(TRAIN_DATA) as f:
        all_rows = [json.loads(l) for l in f]
    print(f"Loaded {len(all_rows)} training rows")

    model, tokenizer = load_policy(args.resume or args.model)
    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    global_step = 0

    for epoch in range(args.epochs):
        random.shuffle(all_rows)
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_reward = 0.0
        n_accum = 0

        for row in all_rows:
            # ── 1. Generate rollouts ──────────────────────────────────
            model.eval()
            completions = generate_rollouts(model, tokenizer, row["prompt"], args.n_rollouts, str(device))
            model.train()

            # ── 2. Compute rewards (parallel sandbox) ─────────────────
            rewards_list = compute_rewards(
                completions, [row] * args.n_rollouts, max_workers=args.sandbox_workers
            )
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)

            # Skip if all rewards identical (zero variance → no signal)
            if rewards.std() < 1e-6:
                continue

            # ── 3. Tokenize prompt ────────────────────────────────────
            prompt_ids = tokenizer(
                row["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_PROMPT_LEN,
                add_special_tokens=True,
            ).input_ids[0].to(device)

            policy_lps = []
            ref_lps    = []

            for completion in completions:
                comp_ids = tokenizer(
                    completion,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=MAX_NEW_TOKENS,
                ).input_ids[0].to(device)

                if comp_ids.shape[0] == 0:
                    zero = torch.zeros(1, device=device)
                    policy_lps.append(zero.squeeze().requires_grad_(True))
                    ref_lps.append(zero.squeeze())
                    continue

                # Policy logprob — with gradient
                lp = sequence_logprob(model, prompt_ids, comp_ids)
                policy_lps.append(lp)

                # Reference logprob — disable LoRA, no gradient
                model.disable_adapter_layers()
                with torch.no_grad():
                    ref_lp = sequence_logprob(model, prompt_ids, comp_ids)
                model.enable_adapter_layers()
                ref_lps.append(ref_lp)

            policy_lps_t = torch.stack(policy_lps)           # [N]
            ref_lps_t    = torch.stack(ref_lps).detach()     # [N]

            # ── 4. GRPO loss ──────────────────────────────────────────
            loss = grpo_loss(policy_lps_t, ref_lps_t, rewards, beta=args.beta)
            (loss / args.grad_accum).backward()

            accum_loss   += loss.item()
            accum_reward += rewards.mean().item()
            n_accum      += 1
            global_step  += 1

            # ── 5. Optimizer step every grad_accum prompts ────────────
            if global_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()

            # ── Logging ───────────────────────────────────────────────
            if global_step % args.log_every == 0 and n_accum > 0:
                print(
                    f"epoch={epoch+1}  step={global_step:>6}  "
                    f"loss={accum_loss/n_accum:.4f}  "
                    f"reward={accum_reward/n_accum:.3f}",
                    flush=True,
                )
                accum_loss = accum_reward = 0.0
                n_accum = 0

            # ── Checkpoint ────────────────────────────────────────────
            if global_step % args.save_every == 0:
                ckpt = out_dir / f"step_{global_step}"
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"Saved → {ckpt}")

        # End of epoch
        ckpt = out_dir / f"epoch_{epoch+1}"
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"Epoch {epoch+1} complete → {ckpt}")

    final = out_dir / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"Training complete → {final}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           default=DEFAULT_MODEL)
    parser.add_argument("--resume",          default=None,    help="Resume from checkpoint dir")
    parser.add_argument("--output-dir",      default=str(OUTPUT_DIR))
    parser.add_argument("--epochs",          type=int,   default=MAX_EPOCHS)
    parser.add_argument("--n-rollouts",      type=int,   default=N_ROLLOUTS)
    parser.add_argument("--beta",            type=float, default=BETA)
    parser.add_argument("--lr",              type=float, default=LR)
    parser.add_argument("--grad-accum",      type=int,   default=GRAD_ACCUM)
    parser.add_argument("--save-every",      type=int,   default=SAVE_EVERY)
    parser.add_argument("--log-every",       type=int,   default=LOG_EVERY)
    parser.add_argument("--sandbox-workers", type=int,   default=SANDBOX_WORKERS)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
