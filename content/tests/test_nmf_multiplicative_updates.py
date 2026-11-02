import numpy as np, yaml
from pathlib import Path
from sklearn.decomposition import NMF

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "nmf-multiplicative-updates.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(7)
X = rng.random((30, 4)) @ rng.random((4, 20)) + 0.01 * rng.random((30, 20))
r = 4
W0, H0 = rng.random((30, r)) + 0.1, rng.random((r, 20)) + 0.1
W, H, loss = ns["nmf"](X, W0, H0, iters=60)
assert (W >= 0).all() and (H >= 0).all()
assert all(b <= a + 1e-12 for a, b in zip(loss, loss[1:]))
ref = NMF(r, init="custom", solver="mu", beta_loss="frobenius", max_iter=60,
          tol=0, random_state=0)
Wr = ref.fit_transform(X, W=W0.copy(), H=H0.copy())
assert np.allclose(W, Wr, rtol=1e-5, atol=1e-8)
assert np.allclose(H, ref.components_, rtol=1e-5, atol=1e-8)
assert np.isclose(np.sqrt(2 * loss[-1]), ref.reconstruction_err_, rtol=1e-8)
print("nmf-multiplicative-updates OK")
