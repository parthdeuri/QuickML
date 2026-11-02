import numpy as np, yaml, torch
import torch.nn as nn
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "bahdanau-additive-attention.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
torch.manual_seed(0)
rng = np.random.default_rng(0)
T, ds, dh, da = 6, 5, 7, 4
s = rng.normal(size=ds)
H = rng.normal(size=(T, dh))
Ws = rng.normal(size=(ds, da))
Wh = rng.normal(size=(dh, da))
v = rng.normal(size=da)

ctx, a = ns["additive_attention"](s, H, Ws, Wh, v)

# same weights in torch.nn.Linear layers, composed from torch ops
lin_s = nn.Linear(ds, da, bias=False).double()
lin_h = nn.Linear(dh, da, bias=False).double()
lin_v = nn.Linear(da, 1, bias=False).double()
with torch.no_grad():
    lin_s.weight.copy_(torch.tensor(Ws.T))
    lin_h.weight.copy_(torch.tensor(Wh.T))
    lin_v.weight.copy_(torch.tensor(v)[None])
st, Ht = torch.tensor(s), torch.tensor(H)
scores = lin_v(torch.tanh(lin_s(st)[None, :] + lin_h(Ht))).squeeze(-1)    # (T,)
w = torch.softmax(scores, dim=0)
ref_ctx = (w[:, None] * Ht).sum(0)
assert a.shape == (T,) and ctx.shape == (dh,)
assert np.allclose(a, w.detach().numpy())
assert np.allclose(ctx, ref_ctx.detach().numpy())
assert np.isclose(a.sum(), 1) and (a > 0).all()

# explicit per-position loop form of the score
e = np.array([v @ np.tanh(s @ Ws + H[t] @ Wh) for t in range(T)])
assert np.allclose(a, np.exp(e) / np.exp(e).sum())
print("bahdanau-additive-attention OK")
