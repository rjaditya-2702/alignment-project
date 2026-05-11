"""
SFT QLoRA Training — Qwen3-8B on CLaDDer
Loss: CE over thinking tokens + λ * CE over answer token
"""

# def train():
    # Fine tune a Lora. The model will have two components - Thinking and response
    # Thinking will have to do the reasoning, the answer will be 'yes' or 'no'
    # Loss = Next token prediction loss on thinking and Binary cross entropy on answser

    # load base model on 8bit quantization and freeze it.
    # YOUR CODE HERE:

    # load lora config and model
    # YOUR CODE HERE:

    # make sure lora parameters are the only trainable parameters
    # YOUR CODE HERE:

    # make sure lora and base model are on the same device
    # YOUR CODE HERE:

    # load training and testing data and tokenize # for this setup, we will consider only one source - cladder and ignore causci.
    # cache this to avoid tokenizing on every run.
    # YOUR CODE HERE:

    # use tqdm to track progress.
    # for each epoch:
    # YOUR CODE HERE:
        # for each batch:
        # YOUR CODE HERE:
            # take the prompt and generate answer
            # YOUR CODE HERE:
            
            # extract thinking part
            # YOUR CODE HERE:

            # extract answer part
            # YOUR CODE HERE:

            # Loss term 1: next token prediction loss on thinking part
            # YOUR CODE HERE:

            # Loss term 2: binary cross entropy on answer part (yes -> 1, no -> 0)
            # YOUR CODE HERE:

            # Backprop and optimize
            # YOUR CODE HERE:

        # Test model on test set and save metrics for plotting
        # YOUR CODE HERE:

        # Loss and accuracy logging
        # Save checkpoint every SAVE_EVERY steps
        # collect metrics for plotting
        # YOUR CODE HERE:
    
    # Save plots
    # YOUR CODE HERE:
    # pass

import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm


# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT                    = Path(__file__).resolve().parents[1]
TRAIN_DATA     = ROOT / "src" / "output" / "train.jsonl"
TEST_DATA      = ROOT / "src" / "output" / "test.jsonl"
SFT_LORA_OUTPUT_DIR     = ROOT / "src" / "output_fine_tune_lora"
SFT_LORA_PLOT_DIR       = SFT_LORA_OUTPUT_DIR / "plots"
SFT_LORA_CHECKPOINT_DIR = SFT_LORA_OUTPUT_DIR / "checkpoints"

for d in [SFT_LORA_OUTPUT_DIR, SFT_LORA_PLOT_DIR, SFT_LORA_CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
POLICY_MODEL       = "Qwen/Qwen3-8B"
TRAIN_BATCH_SIZE   = 4
MAX_PROMPT_LEN     = 4096
TRAIN_MAX_TOKENS   = 1200
LR                 = 2e-5
WEIGHT_DECAY       = 0.01
GRAD_ACCUM         = 1
MAX_GRAD_NORM      = 1.0
MAX_EPOCHS         = 3
SAVE_EVERY         = 500
LOG_EVERY          = 10
LORA_R             = 32
LORA_ALPHA         = 64
LORA_DROPOUT       = 0.05
ANSWER_LAMBDA      = 5.0
DTYPE              = torch.bfloat16
DEVICE             = "cuda"

# ── Tokenizer ──────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL)
tokenizer.padding_side = "right"

# Special token IDs we need for loss masking
THINK_CLOSE_STR = "<|im_end|>"   # end of assistant turn
IM_END_ID       = tokenizer.convert_tokens_to_ids("<|im_end|>")
# </think> token — Qwen3 uses token id 151668
THINK_CLOSE_ID  = tokenizer.convert_tokens_to_ids("</think>")

# ── QLoRA BnB Config ───────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=DTYPE,
    bnb_4bit_use_double_quant=True,
)

# ── Load & Freeze Base Model ───────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    POLICY_MODEL,
    quantization_config=bnb_config,
    device_map={"": DEVICE},
    torch_dtype=DTYPE,
    attn_implementation="flash_attention_2",
)
model = prepare_model_for_kbit_training(model)

# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# ── LoRA Config ────────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
model = get_peft_model(model, lora_config)
model = torch.compile(model)
# model.gradient_checkpointing_enable()

# Confirm only LoRA params are trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# ── Build Full Sequence ────────────────────────────────────────────────────────
def build_sequence(prompt: str, thinking: str, answer: str) -> dict:
    """
    Constructs the full token sequence for one training sample.

    Layout:
        [prompt tokens] [<think>\n thinking \n</think>\n] [answer] [<|im_end|>\n]

    Returns:
        input_ids  : (L,)   full token sequence
        loss_mask  : (L,)   0=ignore, 1=thinking CE, LAMBDA=answer CE
    """
    messages = [{"role": "user", "content": prompt}]

    # apply_chat_template with enable_thinking=True ends with:
    # <|im_start|>assistant\n<think>\n
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    # Full target: thinking content + closing tags + answer + end token
    # The prompt_text already opens <think>, so we continue from there
    response_text = thinking + "\n</think>" + answer + "<|im_end|>\n"
    full_text     = prompt_text + response_text

    # Tokenize full sequence (no truncation yet — we handle it below)
    full_ids = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    ).input_ids[0]

    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    ).input_ids[0]

    prompt_len = len(prompt_ids)

    # Truncate if over budget
    max_len = MAX_PROMPT_LEN + TRAIN_MAX_TOKENS
    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]

    seq_len   = len(full_ids)
    loss_mask = torch.zeros(seq_len, dtype=DTYPE)

    # Find </think> position in the full sequence
    think_close_pos = None
    for i in range(prompt_len, seq_len):
        if full_ids[i].item() == THINK_CLOSE_ID:
            think_close_pos = i
            break

    if think_close_pos is not None:
        # Thinking tokens: prompt_len → think_close_pos (inclusive of </think>)
        loss_mask[prompt_len : think_close_pos + 1] = 1.0
        # Answer token: immediately after </think>
        answer_pos = think_close_pos + 1
        if answer_pos < seq_len:
            loss_mask[answer_pos] = ANSWER_LAMBDA
    else:
        # Fallback: CE on everything after prompt
        loss_mask[prompt_len:] = 1.0

    return {"input_ids": full_ids, "loss_mask": loss_mask}


# ── Dataset ────────────────────────────────────────────────────────────────────
class CladderDataset(Dataset):
    def __init__(self, samples: list[dict]):
        """
        Each sample dict must have:
            "input_ids"  : torch.LongTensor  (L,)
            "loss_mask"  : torch.FloatTensor (L,)
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    """Left-pad to the longest sequence in the batch."""
    max_len    = max(s["input_ids"].shape[0] for s in batch)
    input_ids  = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    loss_masks = torch.zeros(len(batch), max_len, dtype=DTYPE)

    for i, s in enumerate(batch):
        L = s["input_ids"].shape[0]
        input_ids[i, -L:]  = s["input_ids"]
        loss_masks[i, -L:] = s["loss_mask"]

    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "loss_mask":      loss_masks,
    }


# ── Custom Loss ────────────────────────────────────────────────────────────────
def compute_loss(logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """
    logits    : (B, L, V)
    input_ids : (B, L)
    loss_mask : (B, L)  — 0=ignore, 1=thinking, LAMBDA=answer

    Shift by 1: logits[t] predicts input_ids[t+1]
    """
    shift_logits = logits[:, :-1, :].contiguous()           # (B, L-1, V)
    shift_labels = input_ids[:, 1:].contiguous()             # (B, L-1)
    shift_mask   = loss_mask[:, 1:].contiguous()             # (B, L-1)

    # Per-token CE, no reduction
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.shape)                               # (B, L-1)

    # Apply mask (0 = no gradient, 1 = full CE, LAMBDA = upweighted)
    weighted_loss = (per_token_loss * shift_mask).sum()
    denom         = (shift_mask > 0).float().sum().clamp(min=1)
    return weighted_loss / denom


# ── Data Loading ────────────────────────────────────────────────────────────────
def format_groundtruth(gt: dict) -> str:
    return "\n\n".join(
        f"## Step {i}: {gt[f'step{i}']}"
        for i in range(1, 5)
        if f"step{i}" in gt
    )

def load_and_tokenize_cladder() -> tuple[CladderDataset, CladderDataset]:
    train_samples, test_samples = [], []

    for path, bucket in tqdm([(TRAIN_DATA, train_samples), (TEST_DATA, test_samples)]):
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                if item["source"] != "cladder":
                    continue
                seq = build_sequence(
                    prompt   = item["prompt"],
                    thinking = format_groundtruth(item["groundtruth"]),
                    answer   = item["label"],
                )
                bucket.append(seq)

    print(f"Train: {len(train_samples)} | Test: {len(test_samples)} CLaDDer samples.")
    return CladderDataset(train_samples), CladderDataset(test_samples)

# ── Optimizer ──────────────────────────────────────────────────────────────────
def build_optimizer(model) -> AdamW:
    return AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

# -- Evalluate ------------------------------------------------------------------
def evaluate(dataset: CladderDataset) -> float:
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    correct, total = 0, 0
    yes_id = tokenizer.convert_tokens_to_ids("Yes")
    no_id  = tokenizer.convert_tokens_to_ids("No")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            loss_mask      = batch["loss_mask"][0]

            # Find answer token position (where loss_mask == ANSWER_LAMBDA)
            answer_positions = (loss_mask == ANSWER_LAMBDA).nonzero(as_tuple=True)[0]
            if len(answer_positions) == 0:
                continue
            answer_pos = answer_positions[0].item()

            with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # logits at answer_pos - 1 predicts the token at answer_pos
            answer_logits = outputs.logits[0, answer_pos - 1, :]
            pred_id       = answer_logits[[yes_id, no_id]].argmax().item()
            pred_label    = "Yes" if pred_id == 0 else "No"
            true_label    = tokenizer.decode(input_ids[0, answer_pos]).strip()

            if pred_label.lower() == true_label.lower():
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {correct}/{total} = {accuracy:.4f}")
    model.train()
    return accuracy

# ── Training Loop ──────────────────────────────────────────────────────────────
def train():
    train_dataset, test_dataset = load_and_tokenize_cladder()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    optimizer  = build_optimizer(model)
    model.train()

    global_step = 0
    for epoch in range(MAX_EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            loss_mask      = batch["loss_mask"].to(DEVICE)

            with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = compute_loss(outputs.logits, input_ids, loss_mask)
                loss    = loss / GRAD_ACCUM

            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    MAX_GRAD_NORM,
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % LOG_EVERY == 0:
                    avg = epoch_loss / (step + 1)
                    print(f"[epoch {epoch+1} | step {global_step}] loss: {avg:.4f}")

                if global_step % SAVE_EVERY == 0:
                    ckpt_path = SFT_LORA_CHECKPOINT_DIR / f"step_{global_step}"
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    print(f"Checkpoint saved → {ckpt_path}")

        print(f"Epoch {epoch+1} complete. Avg loss: {epoch_loss / len(train_dataloader):.4f}")
        # Evaluate at the end of each epoch
        evaluate(test_dataset)

    # Final save
    model.save_pretrained(SFT_LORA_OUTPUT_DIR / "final")
    tokenizer.save_pretrained(SFT_LORA_OUTPUT_DIR / "final")
    print("Training complete.")


if __name__ == "__main__":
    train()