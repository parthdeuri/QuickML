import numpy as np, yaml
from pathlib import Path
from sklearn.linear_model import LinearRegression

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "gradient-descent.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
y = X @ [1.5, -2.0, 0.5] + 4 + rng.normal(0, .1, 200)
w = ns["descend"](ns["mse_grad"](X, y), np.zeros(4))
ref = LinearRegression().fit(X, y)
assert np.allclose(w[0], ref.intercept_, atol=1e-6)
assert np.allclose(w[1:], ref.coef_, atol=1e-6)
print("gradient-descent OK")
