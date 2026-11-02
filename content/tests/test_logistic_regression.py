import numpy as np, yaml
from pathlib import Path
from sklearn.linear_model import LogisticRegression

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "logistic-regression.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
X = rng.normal(size=(300, 3))
logit = X @ [1.5, -2.0, 0.5] + 0.3
y = (rng.random(300) < 1 / (1 + np.exp(-logit))).astype(float)
w = ns["fit"](X, y)
ref = LogisticRegression(penalty=None, tol=1e-10, max_iter=1000).fit(X, y)
assert np.allclose(w[0], ref.intercept_[0], atol=1e-4)
assert np.allclose(w[1:], ref.coef_[0], atol=1e-4)
assert np.allclose(ns["predict_proba"](w, X), ref.predict_proba(X)[:, 1], atol=1e-5)
print("logistic-regression OK")
