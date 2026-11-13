import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "convex-optimisation-lagrange.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

from scipy.optimize import minimize

Q = np.array([[3, 1, 0], [1, 2, 0.5], [0, 0.5, 1.0]])
c = np.array([1.0, -2.0, 0.5])
A = np.array([[1.0, 1.0, 1.0]])
b = np.array([1.0])
assert ns["is_convex_quadratic"](Q)
x, lam = ns["solve_eq_qp"](Q, c, A, b)

obj = lambda z: 0.5 * z @ Q @ z + c @ z
r = minimize(obj, np.zeros(3), method="SLSQP",
             constraints=[{"type": "eq", "fun": lambda z: A @ z - b}],
             options={"ftol": 1e-14})
assert np.allclose(x, r.x, atol=1e-5), (x, r.x)

stat, feas = ns["kkt_residuals"](Q, c, A, b, x, lam)
assert stat < 1e-12 and feas < 1e-12

# a non-PSD matrix is flagged
assert not ns["is_convex_quadratic"](np.array([[1.0, 0], [0, -1.0]]))

# shadow price: d(optimal value)/db = -lambda
val = lambda bb: obj(ns["solve_eq_qp"](Q, c, A, bb)[0])
eps = 1e-5
fd = (val(b + eps) - val(b - eps)) / (2 * eps)
assert np.allclose(fd, -lam, atol=1e-6), (fd, -lam)
print("convex-optimisation-lagrange OK")
