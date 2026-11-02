import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "transformer-block.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
T, d, h, f = 6, 8, 2, 20
shapes = dict(Wq=(d, d), Wk=(d, d), Wv=(d, d), Wo=(d, d), W1=(d, f), W2=(f, d))
P = {n: rng.normal(size=s) / np.sqrt(d) for n, s in shapes.items()}
P.update(g1=1 + 0.1 * rng.normal(size=d), b1=0.1 * rng.normal(size=d),
         g2=1 + 0.1 * rng.normal(size=d), b2=0.1 * rng.normal(size=d),
         c1=0.1 * rng.normal(size=f), c2=0.1 * rng.normal(size=d))
x = rng.normal(size=(T, d))
out = ns["transformer_block"](x, P, h)

t = {k: torch.tensor(v) for k, v in P.items()}
ln1 = torch.nn.LayerNorm(d).double()
ln2 = torch.nn.LayerNorm(d).double()
with torch.no_grad():
    ln1.weight.copy_(t["g1"]); ln1.bias.copy_(t["b1"])
    ln2.weight.copy_(t["g2"]); ln2.bias.copy_(t["b2"])
xt = torch.tensor(x)
y = ln1(xt)
q, k, v = (F.linear(y, t[n].T).view(T, h, d // h).transpose(0, 1) for n in ("Wq", "Wk", "Wv"))
a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(0, 1).reshape(T, d)
xt = xt + F.linear(a, t["Wo"].T)
ref = xt + F.linear(F.gelu(F.linear(ln2(xt), t["W1"].T, t["c1"])), t["W2"].T, t["c2"])
assert out.shape == (T, d)
assert np.allclose(out, ref.detach().numpy(), atol=1e-10)
# causal: changing the last token leaves earlier rows unchanged
x2 = x.copy()
x2[-1] += 1
assert np.allclose(ns["transformer_block"](x2, P, h)[:-1], out[:-1])
print("transformer-block OK")
