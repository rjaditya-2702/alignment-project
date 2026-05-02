"""
GRPO post-training for causal alignment.

Policy:    Qwen3-14B + LoRA — trainable
Reference: same base weights with adapters disabled — frozen

Algorithm:
  For each batch of B prompts, generate N rollouts each (B*N total).
  Score all rollouts with reward functions (heuristics + DeepSeek-Math judge).
  Normalize rewards within each prompt's group of N: â = (r - mean) / std.
  GRPO loss = mean over valid groups of [-mean(â * logprob) + β * KL(policy || ref)]
  Logprobs computed in one batched forward pass per model (policy and ref separately).

Usage:
    python src/training/train.py
    python src/training/train.py --model Qwen/Qwen3-14B --n-rollouts 8 --batch-size 4
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
JUDGE_MODEL     = "deepseek-ai/deepseek-math-7b-instruct"
TRAIN_DATA      = ROOT / "output" / "train.jsonl"
OUTPUT_DIR      = ROOT / "output" / "checkpoints"

BATCH_SIZE      = 4       # prompts per training step
N_ROLLOUTS      = 8       # completions per prompt
MAX_PROMPT_LEN  = 3072    # truncate prompt to this many tokens
MAX_NEW_TOKENS  = 2048    # max completion length
TEMPERATURE     = 0.8
TOP_P           = 0.9

BETA            = 0.01    # KL coefficient
LR              = 2e-5
WEIGHT_DECAY    = 0.01
GRAD_ACCUM      = 8       # optimizer step every N steps
MAX_GRAD_NORM   = 1.0

MAX_EPOCHS      = 3
SAVE_EVERY      = 500     # global steps between checkpoints
LOG_EVERY       = 10      # global steps between log lines

R = 32

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=R,
    lora_alpha=64,
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

JUDGE_QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


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
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    if len(tokenizer) > base.config.vocab_size:
        base.resize_token_embeddings(len(tokenizer))

    model = get_peft_model(base, LORA_CONFIG)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    return model, tokenizer


def load_judge(model_name: str):
    print(f"Loading judge tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading judge model from {model_name} (4-bit)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=JUDGE_QUANT_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
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

def generate_batch_rollouts(
    model, tokenizer, prompts: list[str], n: int, device: str
) -> list[str]:
    """
    Generate n rollouts for each of B prompts.
    Returns flat list of B*N completions ordered [p0r0..p0rN-1, p1r0..p1rN-1, ...].
    No gradient.
    """
    formatted = [format_prompt(tokenizer, p) for p in prompts]
    enc = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
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
    # out_ids: [B*N, seq_len] — all N rollouts for prompt 0 first, then prompt 1, etc.
    prompt_len = enc["input_ids"].shape[1]
    return [
        tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        for out in out_ids
    ]


# ── Batched logprob computation ───────────────────────────────────────────────

def batched_sequence_logprobs(
    model,
    prompt_ids_list: list[torch.Tensor],
    comp_ids_list: list[torch.Tensor],
    pad_id: int,
) -> list[torch.Tensor]:
    """
    Compute mean per-token completion logprob for each (prompt, completion) pair
    in a single batched forward pass (right-padded).

    prompt_ids_list: list of BN 1D tensors
    comp_ids_list:   list of BN 1D tensors
    Returns list of BN scalar tensors (grad flows if model is in train mode).
    """
    BN = len(prompt_ids_list)
    dev = prompt_ids_list[0].device

    full_ids = [torch.cat([p, c]) for p, c in zip(prompt_ids_list, comp_ids_list)]
    max_len = max(f.shape[0] for f in full_ids)

    padded    = torch.full((BN, max_len), pad_id, dtype=torch.long, device=dev)
    attn_mask = torch.zeros(BN, max_len, dtype=torch.long, device=dev)
    for i, f in enumerate(full_ids):
        padded[i, :f.shape[0]] = f
        attn_mask[i, :f.shape[0]] = 1

    outputs  = model(input_ids=padded, attention_mask=attn_mask)
    logits   = outputs.logits                                          # [BN, max_len, V]
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)                 # [BN, max_len-1, V]
    labels    = padded[:, 1:]                                          # [BN, max_len-1]
    token_lp  = log_probs.gather(2, labels.unsqueeze(2)).squeeze(2)   # [BN, max_len-1]

    result = []
    for i, (p, c) in enumerate(zip(prompt_ids_list, comp_ids_list)):
        P, C = p.shape[0], c.shape[0]
        if C == 0:
            result.append(torch.zeros(1, device=dev).squeeze())
            continue
        comp_lp = token_lp[i, P - 1 : P - 1 + C]
        result.append(comp_lp.mean())

    return result


# ── GRPO loss ─────────────────────────────────────────────────────────────────

def grpo_loss(
    policy_lps: torch.Tensor,   # [N]
    ref_lps: torch.Tensor,      # [N], detached
    rewards: torch.Tensor,      # [N]
    beta: float = BETA,
) -> torch.Tensor:
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    kl  = policy_lps - ref_lps
    return -(adv * policy_lps).mean() + beta * kl.mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(TRAIN_DATA) as f:
        all_rows = [json.loads(l) for l in f]
    print(f"Loaded {len(all_rows)} training rows")

    model, tokenizer             = load_policy(args.resume or args.model)
    judge_model, judge_tokenizer = load_judge(JUDGE_MODEL)
    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    global_step  = 0

    for epoch in range(args.epochs):
        random.shuffle(all_rows)
        optimizer.zero_grad()
        accum_loss = accum_reward = 0.0
        n_accum = 0

        for batch_start in range(0, len(all_rows), args.batch_size):
            batch = all_rows[batch_start : batch_start + args.batch_size]
            B     = len(batch)
            N     = args.n_rollouts

            # ── 1. Generate B*N rollouts ──────────────────────────────
            model.eval()
            all_completions = generate_batch_rollouts(
                model, tokenizer, [r["prompt"] for r in batch], N, str(device)
            )
            model.train()

            # ── 2. Compute rewards ────────────────────────────────────
            flat_rows    = [r for r in batch for _ in range(N)]
            rewards_list = compute_rewards(all_completions, flat_rows, judge_model, judge_tokenizer)
            rewards      = torch.tensor(rewards_list, dtype=torch.float32, device=device).view(B, N)

            # Skip batch if every group has zero variance
            if all(rewards[b].std() < 1e-6 for b in range(B)):
                continue

            # ── 3. Tokenize all prompts and completions ───────────────
            # Each prompt repeated N times to pair with its rollouts
            prompt_ids_list = []
            for row in batch:
                ids = tokenizer(
                    row["prompt"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_PROMPT_LEN,
                    add_special_tokens=True,
                ).input_ids[0].to(device)
                for _ in range(N):
                    prompt_ids_list.append(ids)

            comp_ids_list = []
            for comp in all_completions:
                ids = tokenizer(
                    comp,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=MAX_NEW_TOKENS,
                ).input_ids[0].to(device)
                comp_ids_list.append(ids)

            # ── 4. Policy logprobs — one batched forward pass ─────────
            policy_lps_flat = batched_sequence_logprobs(
                model, prompt_ids_list, comp_ids_list, tokenizer.pad_token_id
            )

            # ── 5. Reference logprobs — adapters off, no grad ─────────
            model.disable_adapter_layers()
            with torch.no_grad():
                ref_lps_flat = batched_sequence_logprobs(
                    model, prompt_ids_list, comp_ids_list, tokenizer.pad_token_id
                )
            model.enable_adapter_layers()

            # ── 6. Reshape [B, N] and compute GRPO loss ───────────────
            policy_lps_t = torch.stack(policy_lps_flat).view(B, N)
            ref_lps_t    = torch.stack(ref_lps_flat).view(B, N).detach()

            losses = []
            for b in range(B):
                if rewards[b].std() < 1e-6:
                    continue
                losses.append(
                    grpo_loss(policy_lps_t[b], ref_lps_t[b], rewards[b], beta=args.beta)
                )

            if not losses:
                continue

            loss = torch.stack(losses).mean()
            (loss / args.grad_accum).backward()

            accum_loss   += loss.item()
            accum_reward += rewards.mean().item()
            n_accum      += 1
            global_step  += 1

            # ── 7. Optimizer step ─────────────────────────────────────
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
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--resume",      default=None,         help="Resume from checkpoint dir")
    parser.add_argument("--output-dir",  default=str(OUTPUT_DIR))
    parser.add_argument("--epochs",      type=int,   default=MAX_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=BATCH_SIZE,  help="Prompts per training step")
    parser.add_argument("--n-rollouts",  type=int,   default=N_ROLLOUTS)
    parser.add_argument("--beta",        type=float, default=BETA)
    parser.add_argument("--lr",          type=float, default=LR)
    parser.add_argument("--grad-accum",  type=int,   default=GRAD_ACCUM)
    parser.add_argument("--save-every",  type=int,   default=SAVE_EVERY)
    parser.add_argument("--log-every",   type=int,   default=LOG_EVERY)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
