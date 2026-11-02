import numpy as np, yaml
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "gaussian-process-regression.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(3)
X = rng.uniform(-3, 3, (25, 2))
y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0, 0.1, 25)
Xs = rng.uniform(-4, 4, (40, 2))
ell, s2, noise = 1.3, 2.0, 0.05
mean, var, lml = ns["gp"](X, y, Xs, ell, s2, noise)
ref = GaussianProcessRegressor(ConstantKernel(s2, "fixed") * RBF(ell, "fixed"),
                               alpha=noise, optimizer=None).fit(X, y)
rm, rs = ref.predict(Xs, return_std=True)
assert np.allclose(mean, rm, atol=1e-8)
assert np.allclose(np.sqrt(var), rs, atol=1e-6)
assert np.isclose(lml, ref.log_marginal_likelihood_value_, atol=1e-8)
# far from the data the variance returns to the prior variance
_, vfar, _ = ns["gp"](X, y, np.array([[100.0, 100.0]]), ell, s2, noise)
assert np.isclose(vfar[0], s2)
print("gaussian-process-regression OK")
