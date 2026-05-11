"""
Launcher for src/training/train.py.

CUDA_VISIBLE_DEVICES must be set before any CUDA initialization — not inside
the training script after torch/vllm are already imported. This launcher sets
it here, before any other import, so vLLM worker subprocesses (spawned later
with multiprocessing) inherit the correct GPU visibility.

GPU layout:
  GPU 0-1  →  vLLM gen engine  (tensor_parallel_size=2; workers claim by rank)
  GPU 2    →  vLLM judge       (pinned via device="cuda:2" in load_judge_vllm)
  GPU 3    →  HF LoRA policy   (pinned via device_map={"": "cuda:3"})
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.training.train import train_init as train

if __name__ == "__main__":   # ← add this guard
    train()