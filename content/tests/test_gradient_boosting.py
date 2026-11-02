import numpy as np, yaml
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "gradient-boosting.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 4))
y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 - X[:, 2] + 0.1 * rng.normal(size=200)
Xt = rng.normal(size=(80, 4))
model = ns["fit"](X, y, rounds=50, lr=0.1)
ref = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=1,
                                random_state=0).fit(X, y)
assert np.allclose(ns["predict"](model, X), ref.predict(X), atol=1e-6)
assert np.allclose(ns["predict"](model, Xt), ref.predict(Xt), atol=1e-6)
print("gradient-boosting OK")
