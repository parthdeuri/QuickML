import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "loss-functions-regression-classification.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
T = lambda a, **k: torch.tensor(a, requires_grad=True, **k)

p, y = rng.normal(size=40) * 2, rng.normal(size=40)
d = 0.8
pairs = [
    ("mse", lambda t: F.mse_loss(t, torch.tensor(y)), ()),
    ("mae", lambda t: F.l1_loss(t, torch.tensor(y)), ()),
    ("huber", lambda t: F.huber_loss(t, torch.tensor(y), delta=d), (d,)),
]
for name, tf, extra in pairs:
    t = T(p)
    out = tf(t)
    out.backward()
    assert np.isclose(ns[name](p, y, *extra), out.item()), name
    assert np.allclose(ns[name + "_grad"](p, y, *extra), t.grad.numpy()), name

# bce with logits (large logits too, for stability)
z = rng.normal(size=50) * 30
yb = rng.integers(0, 2, 50).astype(float)
t = T(z)
out = F.binary_cross_entropy_with_logits(t, torch.tensor(yb))
out.backward()
assert np.isfinite(ns["bce_logits"](z, yb))
assert np.isclose(ns["bce_logits"](z, yb), out.item())
assert np.allclose(ns["bce_logits_grad"](z, yb), t.grad.numpy())

# categorical cross-entropy
Z = rng.normal(size=(30, 6)) * 3
yc = rng.integers(0, 6, 30)
t = T(Z)
out = F.cross_entropy(t, torch.tensor(yc))
out.backward()
assert np.isclose(ns["cce"](Z, yc), out.item())
assert np.allclose(ns["cce_grad"](Z, yc), t.grad.numpy())

# pinball and hinge: hand values plus finite-difference gradients
assert np.isclose(ns["pinball"](np.array([0.0]), np.array([2.0]), 0.9), 1.8)
assert np.isclose(ns["pinball"](np.array([2.0]), np.array([0.0]), 0.9), 0.2)
assert np.isclose(ns["pinball"](p, y, 0.5), 0.5 * ns["mae"](p, y))
s, ys = rng.normal(size=30), rng.choice([-1.0, 1.0], 30)
assert np.isclose(ns["hinge"](np.array([0.5, -2.0]), np.array([1.0, 1.0])), (0.5 + 3.0) / 2)

def fd(f, a, eps=1e-6):
    g = np.zeros_like(a)
    for i in range(a.size):
        b = a.copy(); b[i] += eps
        c = a.copy(); c[i] -= eps
        g[i] = (f(b) - f(c)) / (2 * eps)
    return g

assert np.allclose(ns["pinball_grad"](p, y, 0.3), fd(lambda a: ns["pinball"](a, y, 0.3), p), atol=1e-6)
assert np.allclose(ns["hinge_grad"](s, ys), fd(lambda a: ns["hinge"](a, ys), s), atol=1e-6)

# probabilistic view: Gaussian NLL = MSE up to scale/constant, Laplace NLL = MAE
sig, b = 1.7, 0.9
r = rng.normal(size=1000)
nll_g = np.mean(r**2 / (2 * sig**2) + np.log(sig) + 0.5 * np.log(2 * np.pi))
assert np.isclose(nll_g, ns["mse"](r, 0) / (2 * sig**2) + np.log(sig) + 0.5 * np.log(2 * np.pi))
nll_l = np.mean(np.abs(r) / b + np.log(2 * b))
assert np.isclose(nll_l, ns["mae"](r, 0) / b + np.log(2 * b))
# the MSE minimiser is the mean, the MAE minimiser the median
data = rng.exponential(size=301)
grid = np.linspace(0, 3, 3001)
best_mse = grid[np.argmin([ns["mse"](np.full_like(data, c), data) for c in grid])]
best_mae = grid[np.argmin([ns["mae"](np.full_like(data, c), data) for c in grid])]
assert abs(best_mse - data.mean()) < 2e-3 and abs(best_mae - np.median(data)) < 2e-3

# one big outlier: MSE moves far more than MAE or Huber
clean = rng.normal(size=100)
dirty = clean.copy(); dirty[0] = 100.0
zero = np.zeros(100)
inc = {n: ns[n](zero, dirty) - ns[n](zero, clean) for n in ("mse", "mae", "huber")}
assert inc["mse"] > 50 * inc["mae"] and inc["mse"] > 50 * inc["huber"], inc
print("loss-functions-regression-classification OK")
