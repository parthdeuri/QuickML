import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "derivatives-chain-rule-gradient-check.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

import torch

rng = np.random.default_rng(0)
W, b, x = rng.normal(size=(4, 3)), rng.normal(size=4), rng.normal(size=3)
fn = lambda z: ns["f"](z, W, b)
ana = ns["analytic_grad"](x, W, b)

xt = torch.tensor(x, requires_grad=True)
out = (torch.tanh(torch.tensor(W) @ xt + torch.tensor(b)) ** 2).sum()
out.backward()
assert np.allclose(ana, xt.grad.numpy(), rtol=0, atol=1e-10)

def err(kind, h):
    return np.linalg.norm(ns["numerical_grad"](fn, x, h, kind) - ana)

c2, c3 = err("central", 1e-2), err("central", 1e-3)
f2, f3 = err("forward", 1e-2), err("forward", 1e-3)
assert c2 < f2 and c3 < f3
assert 50 < c2 / c3 < 200, c2 / c3
assert 5 < f2 / f3 < 20, f2 / f3

good = ns["rel_error"](ana, ns["numerical_grad"](fn, x))
assert good < 1e-6, good
bad = ns["rel_error"](ana * 1.1 + 0.05, ns["numerical_grad"](fn, x))
assert bad > 1e-2, bad
print("derivatives-chain-rule-gradient-check OK")
