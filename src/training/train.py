"""
GRPO post-training for causal alignment.

Policy:    Qwen3-14B + LoRA — trainable
Reference: same base weights with adapters disabled — frozen (single GPU)
           OR separate 8-bit base model on GPU 1 (multi-GPU)

Algorithm:
  For each batch of B prompts, generate N rollouts each (B*N total).
  Score all rollouts with reward functions (heuristics + DeepSeek-Math judge).
  Normalize rewards within each prompt's group of N: â = (r - mean) / std.
  GRPO loss = mean over valid groups of [-mean(â * logprob) + β * KL(policy || ref)]
  Logprobs computed in one batched forward pass per model (policy and ref separately).

Multi-GPU pipeline (2+ GPUs):
  GPU 0: policy (generation + policy logprobs + backward)
  GPU 1: reference model (8-bit) + judge (reference logprobs + rewards)
  Pipeline: GPU 0 generates batch_i while GPU 1 computes ref+judge for batch_{i-1}.
  Within each iteration: GPU 0 policy logprobs and GPU 1 ref+judge also overlap.

Usage:
    python src/training/train.py
    python src/training/train.py --model Qwen/Qwen3-14B --n-rollouts 8 --batch-size 4
    python src/training/train.py --resume output/checkpoints/step_500
"""

import argparse
import json
import queue as queue_module
import random
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from src.training.reward import compute_rewards
from src.config import (
    POLICY_MODEL as DEFAULT_MODEL,
    JUDGE_MODEL,
    TRAIN_BATCH_SIZE as BATCH_SIZE,
    N_ROLLOUTS,
    MAX_PROMPT_LEN,
    TRAIN_MAX_TOKENS as MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    BETA,
    LR,
    WEIGHT_DECAY,
    GRAD_ACCUM,
    MAX_GRAD_NORM,
    MAX_EPOCHS,
    SAVE_EVERY,
    LOG_EVERY,
    LORA_R as R,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

TRAIN_DATA = ROOT / "output" / "train.jsonl"
OUTPUT_DIR = ROOT / "output" / "checkpoints"

# ── LoRA / quant configs ──────────────────────────────────────────────────────

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=R,
    lora_alpha=64,
    lora_dropout=0.02,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

JUDGE_QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

REF_QUANT_CONFIG = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_policy(model_name: str, device: str = None):
    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if "<|endoftext|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|endoftext|>"
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "left"

    if device is None:
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


def load_ref_model(model_name: str, device: str):
    """Load base weights only (no LoRA), 8-bit quantized, on the given device."""
    print(f"Loading reference model from {model_name} on {device} (8-bit)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=None, # for now. replace with REF_QUANT_CONFIG later.
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_judge(model_name: str, device: str = None):
    print(f"Loading judge tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device_map = {"": device} if device is not None else "auto"
    print(f"Loading judge model from {model_name} (4-bit, device={device_map})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=JUDGE_QUANT_CONFIG,
        device_map=device_map,
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


# ── Tokenization helpers ──────────────────────────────────────────────────────

def _tokenize_prompts_cpu(tokenizer, batch, N):
    """Tokenize each prompt N times; return list of CPU tensors (length B*N)."""
    result = []
    for row in batch:
        ids = tokenizer(
            row["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_PROMPT_LEN,
            add_special_tokens=True,
        ).input_ids[0]
        for _ in range(N):
            result.append(ids)
    return result


def _tokenize_completions_cpu(tokenizer, completions):
    """Tokenize completions; return list of CPU tensors."""
    return [
        tokenizer(
            c,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_NEW_TOKENS,
        ).input_ids[0]
        for c in completions
    ]


# ── Multi-GPU pipeline worker ─────────────────────────────────────────────────

def _ref_judge_worker(
    ref_model,
    judge_model,
    judge_tokenizer,
    pad_id: int,
    ref_device: str,
    gen_q: queue_module.Queue,
    result_q: queue_module.Queue,
):
    """
    GPU 1 worker thread.
    Receives (batch, completions, prompt_ids_cpu, comp_ids_cpu) from gen_q.
    Computes reference logprobs and rewards, puts (batch, ref_lps_cpu, rewards_list)
    in result_q. Exits on None sentinel.
    """
    while True:
        item = gen_q.get()
        if item is None:
            return

        batch, completions, prompt_ids_cpu, comp_ids_cpu = item
        N = len(completions) // len(batch)
        flat_rows = [r for r in batch for _ in range(N)]

        prompt_ids = [t.to(ref_device) for t in prompt_ids_cpu]
        comp_ids   = [t.to(ref_device) for t in comp_ids_cpu]

        with torch.no_grad():
            ref_lps_flat = batched_sequence_logprobs(
                ref_model, prompt_ids, comp_ids, pad_id
            )

        rewards_list = compute_rewards(
            completions, flat_rows, judge_model, judge_tokenizer
        )

        ref_lps_cpu = [lp.cpu() for lp in ref_lps_flat]
        result_q.put((batch, ref_lps_cpu, rewards_list))


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    n_gpu = torch.cuda.device_count()
    print(f"Detected {n_gpu} CUDA GPU(s)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(TRAIN_DATA) as f:
        all_rows = [json.loads(l) for l in f]
    print(f"Loaded {len(all_rows)} training rows")

    if n_gpu >= 2:
        _train_multi_gpu(args, all_rows, out_dir)
    else:
        _train_single_gpu(args, all_rows, out_dir)


def _train_single_gpu(args, all_rows, out_dir):
    model, tokenizer             = load_policy(args.resume or args.model)
    judge_model, judge_tokenizer = load_judge(JUDGE_MODEL)
    device = str(next(model.parameters()).device)
    pad_id = tokenizer.pad_token_id

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    global_step = 0

    for epoch in range(args.epochs):
        random.shuffle(all_rows)
        optimizer.zero_grad()
        accum_loss = accum_reward = 0.0
        n_accum = 0

        for batch_start in range(0, len(all_rows), args.batch_size):
            batch = all_rows[batch_start : batch_start + args.batch_size]
            B     = len(batch)
            N     = args.n_rollouts

            # 1. Generate rollouts
            model.eval()
            completions = generate_batch_rollouts(
                model, tokenizer, [r["prompt"] for r in batch], N, device
            )
            model.train()

            # 2. Rewards
            flat_rows    = [r for r in batch for _ in range(N)]
            rewards_list = compute_rewards(completions, flat_rows, judge_model, judge_tokenizer)
            rewards      = torch.tensor(rewards_list, dtype=torch.float32, device=device).view(B, N)
            if all(rewards[b].std() < 1e-6 for b in range(B)):
                continue

            # 3. Tokenize
            prompt_ids_list = [t.to(device) for t in _tokenize_prompts_cpu(tokenizer, batch, N)]
            comp_ids_list   = [t.to(device) for t in _tokenize_completions_cpu(tokenizer, completions)]

            # 4. Policy logprobs
            policy_lps_flat = batched_sequence_logprobs(model, prompt_ids_list, comp_ids_list, pad_id)

            # 5. Reference logprobs — adapters off, no grad
            model.disable_adapter_layers()
            with torch.no_grad():
                ref_lps_flat = batched_sequence_logprobs(model, prompt_ids_list, comp_ids_list, pad_id)
            model.enable_adapter_layers()

            # 6. GRPO loss
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

            if global_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()

            if global_step % args.log_every == 0 and n_accum > 0:
                print(
                    f"epoch={epoch+1}  step={global_step:>6}  "
                    f"loss={accum_loss/n_accum:.4f}  "
                    f"reward={accum_reward/n_accum:.3f}",
                    flush=True,
                )
                accum_loss = accum_reward = 0.0
                n_accum = 0

            if global_step % args.save_every == 0:
                ckpt = out_dir / f"step_{global_step}"
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"Saved → {ckpt}")

        ckpt = out_dir / f"epoch_{epoch+1}"
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"Epoch {epoch+1} complete → {ckpt}")

    final = out_dir / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"Training complete → {final}")


def _train_multi_gpu(args, all_rows, out_dir):
    """
    Pipelined 2-GPU training.

    GPU 0: policy (Qwen3-14B + LoRA) — generation, policy logprobs, backward
    GPU 1: reference (Qwen3-14B 8-bit, no LoRA) + judge — ref logprobs, rewards

    Pipeline per iteration i:
      [GPU 0] generate batch_i   ←→  [GPU 1] ref+judge for batch_{i-1}  (overlap A)
      [GPU 0] policy_lp batch_{i-1}  ←→  [GPU 1] ref+judge batch_{i-1}  (overlap B)
      [GPU 0] backward batch_{i-1}   (waits for both GPU 0 and GPU 1 to finish)
    """
    policy_device = "cuda:0"
    ref_device    = "cuda:1"

    # Policy on GPU 0; reference model always loaded from base (not checkpoint)
    # so it represents the pre-training distribution regardless of resume state.
    model, tokenizer = load_policy(args.resume or args.model, policy_device)
    ref_model        = load_ref_model(args.model, ref_device)
    judge_model, judge_tokenizer = load_judge(JUDGE_MODEL, ref_device)

    pad_id = tokenizer.pad_token_id

    gen_q    = queue_module.Queue(maxsize=2)
    result_q = queue_module.Queue(maxsize=2)

    worker = threading.Thread(
        target=_ref_judge_worker,
        args=(ref_model, judge_model, judge_tokenizer,
              pad_id, ref_device, gen_q, result_q),
        daemon=True,
    )
    worker.start()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    global_step = 0

    for epoch in range(args.epochs):
        random.shuffle(all_rows)
        optimizer.zero_grad()
        accum_loss = accum_reward = 0.0
        n_accum = 0

        batches = [
            all_rows[s : s + args.batch_size]
            for s in range(0, len(all_rows), args.batch_size)
        ]

        # Rolling state for the previous batch (processed one step behind)
        prev_batch          = None
        prev_prompt_ids_cpu = None
        prev_comp_ids_cpu   = None
        prev_N              = None

        for i, batch in enumerate(batches):
            N = args.n_rollouts

            # ── Overlap A: generate batch_i on GPU 0 ─────────────────────
            # GPU 1 worker is already processing prev batch from last iter's enqueue.
            model.eval()
            completions_i = generate_batch_rollouts(
                model, tokenizer, [r["prompt"] for r in batch], N, policy_device
            )
            model.train()

            # Tokenize batch_i on CPU (fast; doesn't block either GPU)
            prompt_ids_i_cpu = _tokenize_prompts_cpu(tokenizer, batch, N)
            comp_ids_i_cpu   = _tokenize_completions_cpu(tokenizer, completions_i)

            # Enqueue batch_i — GPU 1 will process it after finishing prev batch
            gen_q.put((batch, completions_i, prompt_ids_i_cpu, comp_ids_i_cpu))

            # ── Overlap B + backward for prev batch ───────────────────────
            if prev_batch is not None:
                B_prev = len(prev_batch)

                # GPU 0: policy logprobs for prev batch
                # GPU 1: simultaneously running ref+judge for prev batch (overlap B)
                prev_prompt_ids = [t.to(policy_device) for t in prev_prompt_ids_cpu]
                prev_comp_ids   = [t.to(policy_device) for t in prev_comp_ids_cpu]
                policy_lps_flat = batched_sequence_logprobs(
                    model, prev_prompt_ids, prev_comp_ids, pad_id
                )

                # Wait for GPU 1 result for prev batch (may already be in queue)
                _, ref_lps_cpu, rewards_list = result_q.get()

                rewards = torch.tensor(
                    rewards_list, dtype=torch.float32, device=policy_device
                ).view(B_prev, prev_N)

                if not all(rewards[b].std() < 1e-6 for b in range(B_prev)):
                    policy_lps_t = torch.stack(policy_lps_flat).view(B_prev, prev_N)
                    ref_lps_t = torch.stack(
                        [lp.to(policy_device) for lp in ref_lps_cpu]
                    ).view(B_prev, prev_N).detach()

                    losses = [
                        grpo_loss(policy_lps_t[b], ref_lps_t[b], rewards[b], beta=args.beta)
                        for b in range(B_prev) if rewards[b].std() >= 1e-6
                    ]

                    if losses:
                        loss = torch.stack(losses).mean()
                        (loss / args.grad_accum).backward()

                        accum_loss   += loss.item()
                        accum_reward += rewards.mean().item()
                        n_accum      += 1
                        global_step  += 1

                        if global_step % args.grad_accum == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                            optimizer.step()
                            optimizer.zero_grad()

                        if global_step % args.log_every == 0 and n_accum > 0:
                            print(
                                f"epoch={epoch+1}  step={global_step:>6}  "
                                f"loss={accum_loss/n_accum:.4f}  "
                                f"reward={accum_reward/n_accum:.3f}",
                                flush=True,
                            )
                            accum_loss = accum_reward = 0.0
                            n_accum = 0

                        if global_step % args.save_every == 0:
                            ckpt = out_dir / f"step_{global_step}"
                            model.save_pretrained(ckpt)
                            tokenizer.save_pretrained(ckpt)
                            print(f"Saved → {ckpt}")

            # Roll over: current batch becomes previous for next iteration
            prev_batch          = batch
            prev_prompt_ids_cpu = prompt_ids_i_cpu
            prev_comp_ids_cpu   = comp_ids_i_cpu
            prev_N              = N

        # ── Drain the final batch (enqueued in last loop iteration) ───────
        if prev_batch is not None:
            B_prev = len(prev_batch)

            prev_prompt_ids = [t.to(policy_device) for t in prev_prompt_ids_cpu]
            prev_comp_ids   = [t.to(policy_device) for t in prev_comp_ids_cpu]
            policy_lps_flat = batched_sequence_logprobs(
                model, prev_prompt_ids, prev_comp_ids, pad_id
            )

            _, ref_lps_cpu, rewards_list = result_q.get()

            rewards = torch.tensor(
                rewards_list, dtype=torch.float32, device=policy_device
            ).view(B_prev, prev_N)

            if not all(rewards[b].std() < 1e-6 for b in range(B_prev)):
                policy_lps_t = torch.stack(policy_lps_flat).view(B_prev, prev_N)
                ref_lps_t = torch.stack(
                    [lp.to(policy_device) for lp in ref_lps_cpu]
                ).view(B_prev, prev_N).detach()

                losses = [
                    grpo_loss(policy_lps_t[b], ref_lps_t[b], rewards[b], beta=args.beta)
                    for b in range(B_prev) if rewards[b].std() >= 1e-6
                ]

                if losses:
                    loss = torch.stack(losses).mean()
                    (loss / args.grad_accum).backward()
                    global_step += 1

                    if global_step % args.grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                        optimizer.step()
                        optimizer.zero_grad()

        ckpt = out_dir / f"epoch_{epoch+1}"
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"Epoch {epoch+1} complete → {ckpt}")

    # Shut down worker thread
    gen_q.put(None)
    worker.join()

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
