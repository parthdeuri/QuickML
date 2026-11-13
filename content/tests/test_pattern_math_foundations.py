import yaml, numpy as np
d = yaml.safe_load(open("content/patterns/math-foundations.yaml", encoding="utf8"))
ns = {}
exec(d["template"], ns)
a, b = np.array([1.0, 0, 0]), np.array([0, 2.0, 0])
assert abs(ns["cosine"](a, b)) < 1e-12
assert abs(ns["cosine"](a, 5 * a) - 1) < 1e-12
A = np.array([[2.0, 1], [1, 3]])
lam, v = ns["power_iteration"](A)
assert abs(lam - np.linalg.eigvalsh(A).max()) < 1e-8
assert np.allclose(A @ v, lam * v, atol=1e-6)
f = lambda x: np.sum(np.tanh(x) ** 2)
x = np.array([0.3, -0.7, 1.1])
an = 2 * np.tanh(x) * (1 - np.tanh(x) ** 2)
assert np.allclose(ns["num_grad"](f, x), an, atol=1e-8)
s = ns["sample_exponential"](2.0, 200000, np.random.default_rng(0))
assert abs(s.mean() - 0.5) < 0.01 and (s >= 0).all()
for k in ("name", "order", "problems", "aliases"):
    assert k in d
assert d["id"] == "math-foundations"
print("pattern math-foundations OK")
