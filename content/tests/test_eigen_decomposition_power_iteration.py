import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "eigen-decomposition-power-iteration.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)


rng = np.random.default_rng(0)
Qo, _ = np.linalg.qr(rng.normal(size=(5, 5)))
ev = np.array([5.0, 3.0, 1.0, 0.5, -0.2])
A = Qo @ np.diag(ev) @ Qo.T

w, V = np.linalg.eigh(A)
lam, v = ns["power_iteration"](A, 200)
assert np.isclose(lam, w.max()) and np.isclose(lam, 5.0)
assert min(np.abs(v - V[:, -1]).max(), np.abs(v + V[:, -1]).max()) < 1e-8
assert np.allclose(A @ v, lam * v, atol=1e-8)

# Rayleigh quotient lies in [lambda_min, lambda_max]
for _ in range(200):
    r = ns["rayleigh"](A, rng.normal(size=5))
    assert w.min() - 1e-12 <= r <= w.max() + 1e-12

# deflation exposes the second eigenvalue
A2 = ns["deflate"](A, lam, v)
lam2, v2 = ns["power_iteration"](A2, 300)
assert np.isclose(lam2, 3.0) and abs(v2 @ v) < 1e-6

# error ratio per iteration is close to |lambda2/lambda1| = 0.6
u = rng.normal(size=5)
u /= np.linalg.norm(u)
u1 = Qo[:, 0]
errs = []
for _ in range(15):
    u = A @ u
    u /= np.linalg.norm(u)
    errs.append(np.linalg.norm(u - (u @ u1) * u1))
ratios = np.array(errs[1:]) / np.array(errs[:-1])
assert np.allclose(ratios[-5:], 0.6, atol=0.01), ratios

# equal magnitude eigenvalues (1, -1): no fixed vector
D = np.diag([1.0, -1.0])
_, a = ns["power_iteration"](D, 100)
_, b = ns["power_iteration"](D, 101)
assert np.abs(a - b).max() > 0.1 and np.allclose(a, -b * [-1, 1])
print("eigen-decomposition-power-iteration OK")
