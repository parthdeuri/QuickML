import numpy as np, yaml
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "knn.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
X = rng.normal(size=(300, 4))
y = (X[:, 0] + X[:, 1] ** 2 + 0.3 * rng.normal(size=300) > 0.5).astype(int) + (X[:, 2] > 1)
Xt = rng.normal(size=(100, 4))
for k in (1, 3, 5, 9):
    ref = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    assert (ns["predict"](X, y, Xt, k) == ref.predict(Xt)).all(), k
print("knn OK")
