import numpy as np, yaml
from pathlib import Path
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_ridge import KernelRidge

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "kernel-trick-rbf.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
X = rng.normal(size=(60, 3))
Xt = rng.normal(size=(25, 3))
y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.05 * rng.normal(size=60)
g, lam = 0.7, 0.1
K = ns["rbf"](X, X, g)
assert np.allclose(K, rbf_kernel(X, X, gamma=g), atol=1e-12)
assert np.allclose(ns["rbf"](Xt, X, g), rbf_kernel(Xt, X, gamma=g), atol=1e-12)
assert np.allclose(K, K.T) and np.allclose(np.diag(K), 1)
assert np.linalg.eigvalsh(K).min() > -1e-10  # PSD Gram matrix
ref = KernelRidge(alpha=lam, kernel="rbf", gamma=g).fit(X, y)
assert np.allclose(ns["kernel_ridge"](X, y, Xt, g, lam), ref.predict(Xt), atol=1e-8)


# Kernel trick: polynomial kernel (x.z)^2 equals an explicit feature-map dot product
def phi(x):
    return np.array([x[0] ** 2, x[1] ** 2, np.sqrt(2) * x[0] * x[1]])


a, b = rng.normal(size=2), rng.normal(size=2)
assert np.isclose((a @ b) ** 2, phi(a) @ phi(b))
print("kernel-trick-rbf OK")
