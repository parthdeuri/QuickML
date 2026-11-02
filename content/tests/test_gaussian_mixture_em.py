import numpy as np, yaml
from pathlib import Path
from sklearn.mixture import GaussianMixture

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "gaussian-mixture-em.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
X = np.r_[rng.normal([0, 0], [1.0, 0.5], (150, 2)),
          rng.normal([5, 5], [0.7, 1.2], (100, 2)),
          rng.normal([-4, 6], [0.6, 0.6], (80, 2))]
mu0 = X[[0, 160, 260]].copy()
w, mu, S, ll = ns["em"](X, mu0.copy(), iters=500)
d = X.shape[1]
S0 = np.cov(X.T, bias=True) + 1e-6 * np.eye(d)
ref = GaussianMixture(3, covariance_type="full", weights_init=np.full(3, 1 / 3),
                      means_init=mu0, precisions_init=np.array([np.linalg.inv(S0)] * 3),
                      tol=1e-12, max_iter=500, reg_covar=1e-6, n_init=1).fit(X)
assert ref.converged_
assert np.allclose(w, ref.weights_, atol=1e-6)
assert np.allclose(mu, ref.means_, atol=1e-5)
assert np.allclose(S, ref.covariances_, atol=1e-5)
lj = ns["log_joint"](X, w, mu, S)
m = lj.max(1, keepdims=True)
total = (m[:, 0] + np.log(np.exp(lj - m).sum(1))).sum()
assert np.isclose(total, ref.score(X) * len(X), rtol=1e-9)
# EM never decreases the log-likelihood
assert all(b >= a - 1e-9 for a, b in zip(ll, ll[1:]))
print("gaussian-mixture-em OK")
