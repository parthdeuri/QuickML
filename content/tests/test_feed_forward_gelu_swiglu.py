import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "feed-forward-gelu-swiglu.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
x = rng.normal(size=(5, 12)) * 2
t = torch.tensor(x)
assert np.allclose(ns["gelu"](x), F.gelu(t).numpy(), atol=1e-12)
assert np.allclose(ns["gelu"](x, approx=True), F.gelu(t, approximate="tanh").numpy(), atol=1e-12)
assert np.allclose(ns["silu"](x), F.silu(t).numpy(), atol=1e-12)
assert abs(ns["gelu"](np.array(0.0))) < 1e-15

d, h4, h = 12, 48, 32
W1, W2 = rng.normal(size=(d, h4)) / 4, rng.normal(size=(h4, d)) / 4
b1, b2 = rng.normal(size=h4), rng.normal(size=d)
ref = (F.gelu(t @ torch.tensor(W1) + torch.tensor(b1)) @ torch.tensor(W2) + torch.tensor(b2)).numpy()
assert np.allclose(ns["ffn_gelu"](x, W1, b1, W2, b2), ref, atol=1e-12)

A, B, C = rng.normal(size=(d, h)) / 4, rng.normal(size=(d, h)) / 4, rng.normal(size=(h, d)) / 4
ref = ((F.silu(t @ torch.tensor(A)) * (t @ torch.tensor(B))) @ torch.tensor(C)).numpy()
out = ns["ffn_swiglu"](x, A, B, C)
assert out.shape == x.shape and np.allclose(out, ref, atol=1e-12)

# parameter counts: 2*d*4d equals 3*d*h when h = 8d/3
assert W1.size + W2.size == 2 * d * 4 * d
assert A.size + B.size + C.size == 3 * d * h == 2 * d * 4 * d
print("feed-forward-gelu-swiglu OK")
