import numpy as np, yaml
from pathlib import Path
from sklearn.decomposition import PCA

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "pca.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5)) + rng.normal(size=5)
k = 3
model = ns["fit"](X, k)
ref = PCA(n_components=k, svd_solver="full").fit(X)
assert np.allclose(model[2][:k], ref.explained_variance_)
assert np.allclose(ns["explained_ratio"](model)[:k], ref.explained_variance_ratio_)
sign = np.sign((model[1] * ref.components_).sum(1))[:, None]
assert np.allclose(model[1] * sign, ref.components_, atol=1e-8)
Z, Zr = ns["transform"](model, X), ref.transform(X)
assert np.allclose(Z * sign.T, Zr, atol=1e-8)
assert np.allclose(ns["inverse"](model, Z), ref.inverse_transform(Zr), atol=1e-8)
print("pca OK")
