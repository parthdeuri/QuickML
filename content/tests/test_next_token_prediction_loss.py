import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "next-token-prediction-loss.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
V, T = 11, 9
tokens = rng.integers(0, V, size=T)
inp, tgt = ns["shift"](tokens)
assert len(inp) == len(tgt) == T - 1 and (inp[1:] == tgt[:-1]).all()

logits = rng.normal(size=(T - 1, V)) * 2
tgt = tgt.copy()
tgt[[0, 3, 4]] = -100  # masked prompt / pad positions
loss = ns["token_ce"](logits, tgt)
ref = F.cross_entropy(torch.tensor(logits), torch.tensor(tgt), ignore_index=-100).item()
assert abs(loss - ref) < 1e-12, (loss, ref)

# hand check: single position, two tokens
l1 = ns["token_ce"](np.array([[0.0, np.log(3.0)]]), np.array([1]))
assert abs(l1 - (-np.log(0.75))) < 1e-12

# perplexity of uniform logits is V; bits per token = loss / ln 2
u = ns["token_ce"](np.zeros((5, V)), rng.integers(0, V, size=5))
assert abs(ns["perplexity"](u) - V) < 1e-9
assert abs(u / np.log(2) - np.log2(V)) < 1e-12
print("next-token-prediction-loss OK")
