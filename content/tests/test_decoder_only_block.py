import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "decoder-only-block.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
T, d, h, f = 7, 8, 2, 32
P = {n: rng.normal(size=(d, d)) / np.sqrt(d) for n in ("Wq", "Wk", "Wv", "Wo")}
P["W1"], P["W2"] = rng.normal(size=(d, f)) / np.sqrt(d), rng.normal(size=(f, d)) / np.sqrt(f)
for i in "12":
    P["g" + i], P["b" + i] = 1 + 0.1 * rng.normal(size=d), 0.1 * rng.normal(size=d)
x = rng.normal(size=(T, d))
out = ns["block"](x, P, h)
assert out.shape == x.shape


def torch_block(x):
    t = {k: torch.tensor(v) for k, v in P.items()}
    x = torch.tensor(x)
    y = F.layer_norm(x, (d,), t["g1"], t["b1"], 1e-5)
    q, k, v = (m.reshape(T, h, d // h).transpose(0, 1) for m in (y @ t["Wq"], y @ t["Wk"], y @ t["Wv"]))
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    x = x + a.transpose(0, 1).reshape(T, d) @ t["Wo"]
    y = F.layer_norm(x, (d,), t["g2"], t["b2"], 1e-5)
    return (x + F.gelu(y @ t["W1"], approximate="tanh") @ t["W2"]).numpy()


assert np.allclose(out, torch_block(x), atol=1e-10), np.abs(out - torch_block(x)).max()

# causality: changing token j must not change outputs before j
for j in range(1, T):
    x2 = x.copy()
    x2[j] += rng.normal(size=d)
    out2 = ns["block"](x2, P, h)
    assert np.allclose(out2[:j], out[:j], atol=1e-12)
    assert not np.allclose(out2[j:], out[j:])
print("decoder-only-block OK")
