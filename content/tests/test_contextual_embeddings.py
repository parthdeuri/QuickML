import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "contextual-embeddings.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

vocab = {w: i for i, w in enumerate(["the", "river", "money", "bank", "deep", "rich"])}
rng = np.random.default_rng(0)
d = 8
E = rng.normal(size=(len(vocab), d))
W = {n: rng.normal(size=(d, d)) / np.sqrt(d) for n in "qkv"}
ids = lambda s: np.array([vocab[w] for w in s.split()])
s1, s2 = ids("the river bank"), ids("the money bank")

# static: same vector for "bank" in both sentences
X1, X2 = ns["static_embed"](s1, E), ns["static_embed"](s2, E)
assert np.array_equal(X1[2], X2[2])

# contextual: the "bank" output differs between sentences
H1, A1 = ns["self_attention"](X1, W)
H2, A2 = ns["self_attention"](X2, W)
assert not np.allclose(H1[2], H2[2])
assert np.linalg.norm(H1[2] - H2[2]) > 1e-3
assert np.allclose(A1.sum(1), 1) and np.allclose(A2.sum(1), 1)
# same sentence twice gives the same output (deterministic)
assert np.array_equal(ns["self_attention"](X1, W)[0], H1)

# permutation equivariance (no positional information)
X = ns["static_embed"](ids("the deep river bank rich money"), E)
H, _ = ns["self_attention"](X, W)
perm = rng.permutation(len(X))
Hp, _ = ns["self_attention"](X[perm], W)
assert np.allclose(Hp, H[perm], atol=1e-12)

# equals torch scaled_dot_product_attention
Q, K, V = X @ W["q"], X @ W["k"], X @ W["v"]
ref = F.scaled_dot_product_attention(*(torch.tensor(m)[None] for m in (Q, K, V)))[0].numpy()
assert np.allclose(H, ref, atol=1e-12), np.abs(H - ref).max()
print("contextual-embeddings OK")
