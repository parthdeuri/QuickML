import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "perplexity.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
N, V = 50, 17
logits = rng.normal(size=(N, V)) * 2
y = rng.integers(0, V, size=N)
ppl = ns["perplexity"](logits, y)
ce = F.cross_entropy(torch.tensor(logits), torch.tensor(y))
assert np.isclose(ppl, torch.exp(ce).item())
# uniform model: perplexity equals vocabulary size
assert np.isclose(ns["perplexity"](np.zeros((N, V)), y), V)
# near-perfect model approaches 1
sharp = np.full((N, V), -50.0)
sharp[np.arange(N), y] = 50
assert np.isclose(ns["perplexity"](sharp, y), 1.0)
# a large constant shift of the logits does not overflow or change the value
assert np.isclose(ns["perplexity"](logits + 1000, y), ppl)
# token-weighted: pooling two sequences uses the pooled mean cross-entropy
a, b = slice(0, 20), slice(20, N)
pa, pb = ns["perplexity"](logits[a], y[a]), ns["perplexity"](logits[b], y[b])
pooled = np.exp((20 * np.log(pa) + 30 * np.log(pb)) / N)
assert np.isclose(pooled, ppl)
# base-2 view: 2 ** (bits per token) == exp(nats per token)
assert np.isclose(2 ** (np.log(ppl) / np.log(2)), ppl)
print("perplexity OK")
