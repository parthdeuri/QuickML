import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "mixture-of-experts-routing.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
T, d, h, E = 7, 5, 8, 4
X = rng.normal(size=(T, d))
Wr = rng.normal(size=(d, E))
experts = [(rng.normal(size=(d, h)) / np.sqrt(d), rng.normal(size=(h, d)) / np.sqrt(h)) for _ in range(E)]

# gates of the chosen experts sum to 1 per token, indices distinct
for k in (1, 2, 3, 4):
    idx, g = ns["top_k_gate"](X @ Wr, k)
    assert np.allclose(g.sum(-1), 1) and (g >= 0).all()
    assert all(len(set(r)) == k for r in idx)

# k = E: dense softmax-weighted sum of all experts
dense = np.zeros_like(X)
p = ns["softmax"](X @ Wr)
for t in range(T):
    for i in range(E):
        dense[t] += p[t, i] * ns["expert"](X[t], *experts[i])
assert np.allclose(ns["moe_layer"](X, Wr, experts, E), dense)

# torch reference with the same weights
for k in (1, 2):
    tx, twr = torch.tensor(X), torch.tensor(Wr)
    tex = [(torch.tensor(a), torch.tensor(b)) for a, b in experts]
    val, ind = torch.topk(tx @ twr, k, dim=-1)
    gate = torch.softmax(val, -1)
    ref = torch.zeros_like(tx)
    for t in range(T):
        for j in range(k):
            a, b = tex[ind[t, j]]
            ref[t] += gate[t, j] * (torch.relu(tx[t] @ a) @ b)
    assert np.allclose(ns["moe_layer"](X, Wr, experts, k), ref.numpy())

# only k experts run: zeroing the unchosen experts changes nothing
idx, _ = ns["top_k_gate"](X @ Wr, 1)
used = set(idx.ravel().tolist())
pruned = [ex if i in used else (ex[0] * 0, ex[1] * 0) for i, ex in enumerate(experts)]
assert np.allclose(ns["moe_layer"](X, Wr, pruned, 1), ns["moe_layer"](X, Wr, experts, 1))

# balancing loss: 1 for perfectly uniform routing, larger when collapsed
base = np.array([2.0, 1.0, 0.0, -1.0])
uniform = np.stack([np.roll(base, t) for t in range(4)])
collapsed = np.tile(base, (4, 1))
lu, lc = ns["balance_loss"](uniform, 1), ns["balance_loss"](collapsed, 1)
assert abs(lu - 1) < 1e-12, lu
assert lc > lu + 1, lc
print("mixture-of-experts-routing OK")
