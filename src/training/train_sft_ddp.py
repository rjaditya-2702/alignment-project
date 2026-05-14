"""
SFT QLoRA Training — Qwen3-8B on CLaDDer
Loss: CE over thinking tokens + λ * CE over answer token
"""
import sys
import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────

from src.config import (TRAIN_DATA_SFT_LORA as TRAIN_DATA,
                        TEST_DATA_SFT_LORA  as TEST_DATA,
                        SFT_LORA_OUTPUT_DIR,
                        SFT_LORA_PLOT_DIR, SFT_LORA_CHECKPOINT_DIR,
                        TRAIN_BATCH_SIZE)
from src.data.preprocess import preprocess

CLADDER_PROMPT = """You are given a scenario describing relationships between variables, along with numerical data and a question. Your task is to determine the answer by following these steps precisely.
---

Strict rules (follow these exactly):
- Nothing before "## Step 1" and nothing after the single word in Step 5.
- Write each step exactly once.
- Each step must be short and direct. No long paragraphs or verbosity.
- Do not repeat content from previous steps.
- Output Steps 1–4 inside the thinking block only.
- After Step 4, close the thinking block.
- After </think>, output exactly one word: "Yes" or "No". No quotes, no punctuation, no extra text.
- Stop immediately after that word.
- Do not repeat any step, any code block, or the word "Yes".

### Query Type Definitions

1. **marginal** — What is the overall probability of a variable?
   Formula: P(Y = y)
   Use when: The question asks about the baseline likelihood of an outcome across the whole population, with no conditions or interventions.

2. **correlation** — Does observing one variable change the probability of another?
   Formula: P(Y = y | X = x)
   Use when: The question asks whether knowing or observing one variable's value changes the likelihood of another. No intervention — just observation.

3. **ate** — What is the effect of actively changing (intervening on) a variable?
   Formula: E[Y | do(X=1)] - E[Y | do(X=0)]
   Use when: The question asks whether forcing or setting a variable to a value increases or decreases an outcome. The key word is "intervention" or "effect of doing X."
   Key technique: Use backdoor adjustment if confounders exist: Σ_z P(Z=z)[P(Y=1|X=1,Z=z) - P(Y=1|X=0,Z=z)]. Use frontdoor adjustment if treatment is confounded but a mediator satisfies the frontdoor criterion.

4. **backadj** — Should we adjust for a set of variables when estimating an effect?
   Formula: Check if the set S blocks all backdoor paths between treatment X and outcome Y in the graph.
   Use when: The question asks whether to look at the overall correlation between X and Y, or to look at it stratified by (adjusted for) other variables.
   Answer is yes if S is a valid adjustment set (blocks all non-causal paths), no otherwise.

5. **det-counterfactual** — What would have happened under a different condition?
   Formula: P(Y_x = y | evidence)
   Use when: The question asks what the outcome would have been if the treatment had been different, given specific observed facts. Uses the three-step procedure: (1) Abduction — update P(U) given evidence, (2) Action — set X = x in the structural equations, (3) Prediction — compute P(Y = y) in the modified model.

6. **ett** — For those who received treatment, what would have happened without it?
   Formula: E[Y₁ - Y₀ | X = 1]
   Use when: The question focuses specifically on the treated subgroup and asks how their outcome would change in the absence of treatment. Also called Average Treatment Effect on the Treated (ATT).

7. **nde** — What is the direct effect, not through any mediator?
   Formula: E[Y_{1,M₀} - Y_{0,M₀}]
   Use when: The question asks about the effect of X on Y while holding the mediator at its natural value under no treatment. Also called Natural Direct Effect.

8. **nie** — What is the indirect effect, only through the mediator?
   Formula: E[Y_{0,M₁} - Y_{0,M₀}]
   Use when: The question asks about the effect of X on Y that operates only through an intermediate variable (mediator), not directly. Also called Natural Indirect Effect.

9. **collider_bias** — Does intervening on one cause of a common effect create a spurious association with another cause?
   Formula: Check whether do(X) changes Y when X and Y share only a common effect (collider), not a common cause.
   Use when: The question involves a variable that is caused by both X and Y (a collider), and asks whether intervening on X affects Y. The answer is always no if X and Y have no common causes — the apparent association through the collider is spurious.

10. **exp_away** — Does conditioning on a common effect change the association between its causes?
    Formula: Compare P(Y | X) versus P(Y | X, Z) where Z is a collider.
    Use when: The question asks whether holding fixed (conditioning on) a common effect of X and Y changes how X and Y are associated. This is the "explaining away" phenomenon — conditioning on a collider can create a spurious association between its parents.

---

Now solve the problem in the following way:

```
<think>
## Step 1: Causal Structure
Assign algebraic variables (e.g., X, Y, Z) to each entity mentioned in the scenario. Identify all directed causal edges.
For example: V1 -> V2, V2 -> V3

## Step 2: Query Classification
Based on the question and the definitions above, classify this query. Return exactly one of:
marginal, correlation, ate, backadj, det-counterfactual, ett, nde, nie, collider_bias, exp_away

## Step 3: Derive Estimand
Using the causal graph from Step 1 and the query type from Step 2, write the mathematical expression that answers the question.
- If the query involves do(), apply do-calculus rules (backdoor adjustment, frontdoor adjustment) to eliminate do() terms and express everything in terms of observable probabilities.
- If the query is counterfactual, apply the three-step abduction-action-prediction procedure.
- If the query is about adjustment sets or collider bias, reason about the graph structure (paths, d-separation).

Show your derivation.

## Step 4: Compute
Using the estimand from Step 3 and the numerical values given in the Data section, compute the result step by step. Show the arithmetic explicitly — substitute each probability value and simplify to a final number.
</think>
```

Based on the computed result and what the question is asking, answer Yes or No. One word only.
- For ate/ett/nde/nie: positive result → Yes if question asks "does X increase Y", No if "decrease". Flip if question asks the opposite.
- For marginal: P(Y) > 0.5 and question asks "is Y more likely than not" → Yes.
- For correlation: P(Y|X=1) > P(Y|X=0) and question asks "does observing X increase Y" → Yes.
- For backadj/collider_bias/exp_away: Yes or No based on graph analysis.
- For det-counterfactual: Yes or No based on computed probability.

IMPORTANT: After writing answer with a single word, STOP. No more text is allowed.

## Scenario
{verbalized_story}

Respond now. Begin directly with <think>
"""

for d in [SFT_LORA_OUTPUT_DIR, SFT_LORA_PLOT_DIR, SFT_LORA_CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
POLICY_MODEL       = "Qwen/Qwen3-8B"
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

# ── Tokenizer ──────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL)
tokenizer.padding_side = "right"

# Special token IDs we need for loss masking
THINK_CLOSE_STR = "<|im_end|>"   # end of assistant turn
IM_END_ID       = tokenizer.convert_tokens_to_ids("<|im_end|>")
# </think> token — Qwen3 uses token id 151668
THINK_CLOSE_ID  = tokenizer.convert_tokens_to_ids("</think>")


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
    messages = [
        {"role": "system","content": "You are a causal reasoning expert and a helpful assistant. Don't explain. just do the task"},
        {"role": "user", "content": prompt}]

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

TOKENIZED_CACHE = SFT_LORA_OUTPUT_DIR / "tokenized_data.pt"

def load_and_tokenize_cladder() -> tuple[CladderDataset, CladderDataset]:
    if not os.path.exists(TOKENIZED_CACHE):
        train_samples, test_samples = [], []
        for path, bucket in tqdm([(TRAIN_DATA, train_samples), (TEST_DATA, test_samples)]):
            with open(path, "r") as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    # REMOVE LATER #################
                    if item["source"] != "cladder":#
                        continue                   #
                    ################################
                    seq = build_sequence(
                        prompt   = item["prompt"],
                        thinking = format_groundtruth(item["groundtruth"]),
                        answer   = item["label"],
                    )
                    train_samples.append(seq) if path == TRAIN_DATA else test_samples.append(seq)
        print(f"Train: {len(train_samples)} | Test: {len(test_samples)} CLaDDer samples.")
        tmp = str(TOKENIZED_CACHE) + f".{os.getpid()}.tmp"
        torch.save({"train": train_samples, "test": test_samples}, tmp)
        os.replace(tmp, TOKENIZED_CACHE)  # atomic — identical content across ranks, last writer wins

    saved = torch.load(TOKENIZED_CACHE, weights_only=False, map_location="cpu")
    return CladderDataset(saved["train"]), CladderDataset(saved["test"])

# ── Optimizer ──────────────────────────────────────────────────────────────────
def build_optimizer(model) -> AdamW:
    return AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

# -- Evalluate ------------------------------------------------------------------
def evaluate(ddp_model: torch.nn.Module, dataloader: DataLoader, device: str) -> float:
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)
    ddp_model.eval()
    correct, total = 0, 0
    yes_id = tokenizer.convert_tokens_to_ids("Yes")
    no_id  = tokenizer.convert_tokens_to_ids("No")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            loss_mask      = batch["loss_mask"][0]

            # Find answer token position (where loss_mask == ANSWER_LAMBDA)
            answer_positions = (loss_mask == ANSWER_LAMBDA).nonzero(as_tuple=True)[0]
            if len(answer_positions) == 0:
                continue
            answer_pos = answer_positions[0].item()

            with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                outputs = ddp_model.module(input_ids=input_ids, attention_mask=attention_mask)

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
    ddp_model.train()
    return accuracy

import traceback 

# ── Training Loop ──────────────────────────────────────────────────────────────
def train(train_dataset, test_dataset):
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)

    try:
        # ── QLoRA BnB Config ───────────────────────────────────────────────────────
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_use_double_quant=True,
        )

        # ── Load & Freeze Base Model ───────────────────────────────────────────────
        model = AutoModelForCausalLM.from_pretrained(
            POLICY_MODEL,
            quantization_config=None,
            device_map={"": local_rank},
            torch_dtype=DTYPE,
            attn_implementation="flash_attention_2",
        )
        # model = prepare_model_for_kbit_training(model)

        for param in model.parameters():
            param.requires_grad = False

        # ── LoRA Config ────────────────────────────────────────────────────────────
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

        if local_rank == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in model.parameters())
            print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        train_sampler    = DistributedSampler(train_dataset, shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn, pin_memory=False)
        test_dataloader  = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, pin_memory=False)

        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

        optimizer = build_optimizer(ddp_model)
        ddp_model.train()

        yes_id = tokenizer.convert_tokens_to_ids("Yes")
        no_id  = tokenizer.convert_tokens_to_ids("No")

        metric_steps      = []
        metric_train_loss = []
        metric_train_acc  = []

        global_step = 0
        window_loss, window_correct, window_total = 0.0, 0, 0

        for epoch in range(MAX_EPOCHS):
            train_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                loss_mask      = batch["loss_mask"].to(device)

                with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                    outputs = ddp_model(input_ids=input_ids, attention_mask=attention_mask)
                    loss    = compute_loss(outputs.logits, input_ids, loss_mask)
                    loss    = loss / GRAD_ACCUM

                # Compute train accuracy before backward (logits freed after backward)
                with torch.no_grad():
                    for i in range(input_ids.shape[0]):
                        ans_pos_list = (loss_mask[i] == ANSWER_LAMBDA).nonzero(as_tuple=True)[0]
                        if len(ans_pos_list) == 0:
                            continue
                        ans_pos  = ans_pos_list[0].item()
                        pred_idx = outputs.logits[i, ans_pos - 1, [yes_id, no_id]].argmax().item()
                        pred_tok = yes_id if pred_idx == 0 else no_id
                        if pred_tok == input_ids[i, ans_pos].item():
                            window_correct += 1
                        window_total += 1

                loss.backward()
                window_loss  += loss.item() * GRAD_ACCUM
                epoch_loss   += loss.item() * GRAD_ACCUM

                if (step + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in ddp_model.parameters() if p.requires_grad],
                        MAX_GRAD_NORM,
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % LOG_EVERY == 0 and local_rank == 0:
                        avg = epoch_loss / (step + 1)
                        print(f"[epoch {epoch+1} | step {global_step}] loss: {avg:.4f}")

                    if global_step % SAVE_EVERY == 0 and local_rank == 0:
                        train_acc      = window_correct / window_total if window_total > 0 else 0.0
                        train_loss_avg = window_loss / SAVE_EVERY
                        metric_steps.append(global_step)
                        metric_train_loss.append(train_loss_avg)
                        metric_train_acc.append(train_acc)
                        window_loss, window_correct, window_total = 0.0, 0, 0

                        ckpt_path = SFT_LORA_CHECKPOINT_DIR / f"step_{global_step}"
                        ddp_model.module.save_pretrained(ckpt_path)
                        tokenizer.save_pretrained(ckpt_path)
                        print(f"Checkpoint saved → {ckpt_path} | train_loss={train_loss_avg:.4f} train_acc={train_acc:.4f}")

            if local_rank == 0:
                print(f"Epoch {epoch+1} complete. Avg loss: {epoch_loss / len(train_dataloader):.4f}")

        dist.barrier()
        if local_rank == 0:
            ddp_model.module.save_pretrained(SFT_LORA_OUTPUT_DIR / "final")
            tokenizer.save_pretrained(SFT_LORA_OUTPUT_DIR / "final")
            print("Training complete.")
        dist.barrier()
        dist.destroy_process_group()

        if local_rank == 0:
            test_acc = evaluate(ddp_model, test_dataloader, device)
            print(f"Final test accuracy: {test_acc:.4f}")

            if metric_steps:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].plot(metric_steps, metric_train_loss); axes[0].set_title("Train Loss");     axes[0].set_xlabel("Step")
                axes[1].plot(metric_steps, metric_train_acc);  axes[1].set_title("Train Accuracy"); axes[1].set_xlabel("Step")
                fig.suptitle(f"Final Test Accuracy: {test_acc:.4f}")
                fig.tight_layout()
                plot_path = SFT_LORA_PLOT_DIR / "training_curves.png"
                fig.savefig(plot_path, dpi=100)
                plt.close(fig)
                print(f"Plots saved → {plot_path}")
    except Exception as e:
        print(f"rank {local_rank} CRACHED: {e}", flush = True)
        traceback.print_exc()
        dist.destroy_process_group()
        raise
if __name__ == "__main__":
    # preprocess(cladder_prompt=CLADDER_PROMPT, causci_prompt="CAUSCI_PROMPT {dataset_description} {file_path} {shape} {columns_and_types} {df_head} {df_describe} {missin_values} {low_cardinality} {query}", output_dir=SFT_LORA_OUTPUT_DIR)
    train_dataset, test_dataset = load_and_tokenize_cladder()
    
    train(train_dataset, test_dataset)