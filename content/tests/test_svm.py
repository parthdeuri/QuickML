import numpy as np, yaml
from pathlib import Path
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "svm.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
X, y = make_blobs(300, centers=2, cluster_std=2.0, random_state=0)
y = 2 * y - 1
lam = 0.01
w, b = ns["fit"](X, y, lam=lam)
ref = SVC(kernel="linear", C=1 / (lam * len(y))).fit(X, y)
obj = lambda w, b: lam / 2 * w @ w + np.maximum(0, 1 - y * (X @ w + b)).mean()
assert abs(obj(w, b) - obj(ref.coef_[0], ref.intercept_[0])) < 1e-3
assert np.allclose(w, ref.coef_[0], atol=0.02)
assert (ns["predict"](w, b, X) == ref.predict(X)).mean() > 0.98
print("svm OK")
