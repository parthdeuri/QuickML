import numpy as np, yaml
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "k-means.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=0)
C0 = X[[0, 1, 2]].copy()
C, labels = ns["kmeans"](X, C0)
ref = KMeans(3, init=C0, n_init=1, max_iter=300, tol=0).fit(X)
assert np.allclose(C, ref.cluster_centers_, atol=1e-6)
assert (labels == ref.labels_).all()
print("k-means OK")
