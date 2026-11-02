import numpy as np, yaml
from pathlib import Path
from sklearn.linear_model import ElasticNet

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "elastic-net-coordinate-descent.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(9)
n, d = 120, 15
X = rng.normal(size=(n, d))
X[:, 1] = X[:, 0] + 0.1 * rng.normal(size=n)      # correlated pair
X = (X - X.mean(0)) / X.std(0)
w_true = np.zeros(d)
w_true[[0, 1, 5, 9]] = [2.0, 1.5, -2.0, 1.0]
y = X @ w_true + rng.normal(0, 0.5, n)
y = y - y.mean()                                  # centred: no intercept needed
for alpha, l1r in [(0.1, 0.5), (0.3, 0.9), (0.05, 0.2), (0.2, 1.0)]:
    w = ns["enet"](X, y, alpha, l1r)
    ref = ElasticNet(alpha=alpha, l1_ratio=l1r, fit_intercept=False,
                     tol=1e-14, max_iter=200000).fit(X, y)
    assert np.allclose(w, ref.coef_, atol=1e-7), (alpha, l1r)
    # KKT: zero coefficients have |gradient| <= alpha * l1r
    g = X.T @ (y - X @ w) / n - alpha * (1 - l1r) * w
    assert np.all(np.abs(g[w == 0]) <= alpha * l1r + 1e-8)
assert (ns["enet"](X, y, 0.3, 0.9) == 0).sum() > 0     # produces sparsity
print("elastic-net-coordinate-descent OK")
