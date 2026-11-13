import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "flash-attention-online-softmax.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
fa = ns["online_softmax_attention"]

rng = np.random.default_rng(0)
T, d, dv = 23, 8, 5
Q, K, V = rng.normal(size=(T, d)), rng.normal(size=(T, d)), rng.normal(size=(T, dv))


def naive(Q, K, V, causal):
    s = Q @ K.T / np.sqrt(Q.shape[1])
    if causal:
        s = np.where(np.triu(np.ones(s.shape, bool), 1), -np.inf, s)
    e = np.exp(s - s.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True)) @ V


for causal in (False, True):
    ref = naive(Q, K, V, causal)
    tref = F.scaled_dot_product_attention(
        *(torch.tensor(m)[None] for m in (Q, K, V)), is_causal=causal)[0].numpy()
    assert np.allclose(ref, tref, atol=1e-12)
    outs = []
    for block in (1, 4, 7, 8, 23, 50):  # 4, 7, 8 do not divide 23
        o = fa(Q, K, V, block, causal=causal)
        assert o.shape == (T, dv) and np.isfinite(o).all()
        assert np.abs(o - ref).max() < 1e-10, (block, causal)
        assert np.abs(o - tref).max() < 1e-10, (block, causal)
        outs.append(o)
    for o in outs[1:]:  # block size does not change the answer
        assert np.abs(o - outs[0]).max() < 1e-10

# largest score tile is T x block, never T x T
for block in (4, 7):
    shapes = []
    fa(Q, K, V, block, stats=shapes)
    assert max(c for _, c in shapes) <= block and all(r == T for r, _ in shapes)
    assert len(shapes) == -(-T // block)
    assert block < T and max(r * c for r, c in shapes) < T * T

# huge logits: naive exp overflows, the online version stays finite and exact
Qb, Kb = Q * 400.0, K * 400.0
with np.errstate(over="ignore", invalid="ignore"):
    assert not np.isfinite(np.exp(Qb @ Kb.T / np.sqrt(d))).all()
o = fa(Qb, Kb, V, 5)
assert np.isfinite(o).all()
assert np.abs(o - naive(Qb, Kb, V, False)).max() < 1e-8
# rectangular: more keys than queries, one query row
o1 = fa(Q[:1], K, V, 6)
assert np.abs(o1 - naive(Q[:1], K, V, False)).max() < 1e-10
print("flash-attention-online-softmax OK")
