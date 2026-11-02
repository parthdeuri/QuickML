import numpy as np, yaml
from pathlib import Path
from sklearn.linear_model import Lasso, Ridge

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "ridge-lasso-regularization.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
X = rng.normal(size=(150, 6))
X[:, 1] = X[:, 0] + 0.1 * rng.normal(size=150)
y = X @ [2.0, 0.0, -1.5, 0.0, 0.0, 0.5] + 1.0 + 0.3 * rng.normal(size=150)
w, b = ns["ridge"](X, y, 3.0)
ref = Ridge(alpha=3.0).fit(X, y)
assert np.allclose(w, ref.coef_) and np.allclose(b, ref.intercept_)
w, b = ns["lasso"](X, y, 0.1)
ref = Lasso(alpha=0.1, tol=1e-12, max_iter=100000).fit(X, y)
assert np.allclose(w, ref.coef_, atol=1e-6) and np.allclose(b, ref.intercept_, atol=1e-6)
assert (w == 0).sum() >= 1 and (w == 0).sum() == (ref.coef_ == 0).sum()
print("ridge-lasso-regularization OK")
