import numpy as np, yaml
from pathlib import Path
from sklearn.linear_model import LinearRegression

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "linear-regression.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3)); y = X @ [1.5, -2.0, 0.5] + 4 + rng.normal(0, .1, 200)
w = ns["fit"](X, y)
ref = LinearRegression().fit(X, y)
assert np.allclose(w[0], ref.intercept_) and np.allclose(w[1:], ref.coef_)
assert np.allclose(ns["predict"](w, X), ref.predict(X))
print("linear-regression OK")
