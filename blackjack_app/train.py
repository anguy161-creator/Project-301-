# train.py
# Training script for the nanoGPT-style blackjack explanation model
#
# Expected data format:
# - Place a plain text file at: data/train.txt
# - Example lines:
#   Player: 8,8 | Dealer: 6 | Action: Split | Reason: Splitting 8s avoids a weak hard 16.
#   Player: 10,6 | Dealer: 7 | Action: Hit | Reason: Hard 16 against a dealer 7 is usually too weak to stand on.
#
# This script:
# - builds a character-level tokenizer
# - splits data into train/validation sets
# - trains the GPT model
# - logs train/val loss
# - saves checkpoints
# - generates sample text during training
# - Project 301 

import os
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import GPT, GPTConfig


# -----------------------------
# Configuration
# -----------------------------
data_path = "data/train.txt"
out_dir = "out"
os.makedirs("data", exist_ok=True)

# model hyperparameters
block_size = 128
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1
bias = True

# training hyperparameters
batch_size = 32
max_iters = 3000
eval_interval = 200
eval_iters = 100
learning_rate = 3e-4
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# system
device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if device.startswith("cuda") else "cpu"
torch.manual_seed(1337)

# data split
train_ratio = 0.9

# sample generation
sample_prompt = "Player: 8,8 | Dealer: 6 |"
sample_every_eval = True
sample_max_new_tokens = 120
sample_temperature = 0.8
sample_top_k = 30


# -----------------------------
# Load text data
# -----------------------------
data_file = Path(data_path)
if not data_file.exists():
    raise FileNotFoundError(
        f"Could not find {data_path}\n"
 "Create it first by running: python build_blackjack_dataset.py"

text = data_file.read_text(encoding="utf-8")
if len(text) < block_size + 2:
    raise ValueError("Dataset is too small. Add more training text.")
)

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

# save tokenizer metadata
meta = {
    "vocab_size": vocab_size,
    "stoi": stoi,
    "itos": itos,
}
with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

data = torch.tensor(encode(text), dtype=torch.long)

n = int(train_ratio * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Dataset length: {len(data)} characters")
print(f"Train length:   {len(train_data)}")
print(f"Val length:     {len(val_data)}")
print(f"Vocab size:     {vocab_size}")
print(f"Device:         {device}")


# -----------------------------
# Batching
# -----------------------------
def get_batch(split: str):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# -----------------------------
# Model
# -----------------------------
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias,
)

model = GPT(config).to(device)
optimizer = model.configure_optimizers(
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas=(beta1, beta2),
    device_type=device_type,
)

print(f"Model parameters: {model.get_num_params():,}")


# -----------------------------
# Loss estimation
# -----------------------------
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# -----------------------------
# Text generation
# -----------------------------
@torch.no_grad()
def generate_sample(prompt: str):
    model.eval()
    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(
        prompt_ids,
        max_new_tokens=sample_max_new_tokens,
        temperature=sample_temperature,
        top_k=sample_top_k,
    )
    text_out = decode(generated[0].tolist())
    model.train()
    return text_out


# -----------------------------
# Training loop
# -----------------------------
train_losses = []
val_losses = []
eval_steps = []

best_val_loss = float("inf")
start_time = time.time()

for iter_num in range(max_iters + 1):
    # evaluate periodically
    if iter_num % eval_interval == 0 or iter_num == max_iters:
        losses = estimate_loss()
        train_losses.append(losses["train"])
        val_losses.append(losses["val"])
        eval_steps.append(iter_num)

        elapsed = time.time() - start_time
        print(
            f"step {iter_num:5d} | "
            f"train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f} | "
            f"time {elapsed:.1f}s"
        )

        # save best checkpoint
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_args": {
                    "vocab_size": vocab_size,
                    "block_size": block_size,
                    "n_layer": n_layer,
                    "n_head": n_head,
                    "n_embd": n_embd,
                    "dropout": dropout,
                    "bias": bias,
                },
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, os.path.join(out_dir, "best_model.pt"))
            print(f"Saved new best checkpoint at step {iter_num}")

        # save latest checkpoint
        latest_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_args": {
                "vocab_size": vocab_size,
                "block_size": block_size,
                "n_layer": n_layer,
                "n_head": n_head,
                "n_embd": n_embd,
                "dropout": dropout,
                "bias": bias,
            },
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
        }
        torch.save(latest_checkpoint, os.path.join(out_dir, "latest_model.pt"))

        if sample_every_eval:
            print("\n--- Sample Generation ---")
            try:
                print(generate_sample(sample_prompt))
            except KeyError:
                print("Sample prompt contains characters not in tokenizer.")
            print("--- End Sample ---\n")

    if iter_num == max_iters:
        break

    # training step
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    if grad_clip != 0.0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()


# -----------------------------
# Save final model
# -----------------------------
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_args": {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "dropout": dropout,
            "bias": bias,
        },
        "iter_num": max_iters,
        "best_val_loss": best_val_loss,
    },
    os.path.join(out_dir, "final_model.pt"),
)

print("Training complete.")
print(f"Best validation loss: {best_val_loss:.4f}")


# -----------------------------
# Plot loss curves
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(eval_steps, train_losses, label="Train Loss")
plt.plot(eval_steps, val_losses, label="Validation Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200)
plt.show()
