# sample.py
import json
import torch
from model import GPT, GPTConfig

checkpoint_path = "out/best_model.pt"
meta_path = "out/meta.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

stoi = meta["stoi"]
itos = {int(k): v for k, v in meta["itos"].items()}

def encode(s: str):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

checkpoint = torch.load(checkpoint_path, map_location=device)
config = GPTConfig(**checkpoint["model_args"])
model = GPT(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prompt = "Player: 10,6 | Dealer: 7 |"
x = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

with torch.no_grad():
    y = model.generate(x, max_new_tokens=120, temperature=0.8, top_k=30)

print(decode(y[0].tolist()))
