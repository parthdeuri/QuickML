import numpy as np, yaml
from pathlib import Path
from scipy.cluster.hierarchy import linkage as sp_linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "hierarchical-clustering.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(4)
X = np.r_[rng.normal([0, 0], 0.8, (20, 2)), rng.normal([5, 1], 1.0, (20, 2)),
          rng.normal([2, 6], 0.6, (15, 2))]
n = len(X)
for method in ("single", "average", "ward"):
    Z = ns["linkage"](X, method)
    R = sp_linkage(X, method=method)
    assert np.allclose(Z[:, 2], R[:, 2], atol=1e-9), method
    assert np.allclose(Z[:, 3], R[:, 3]), method
    # merge pairs agree as sets of cluster ids
    assert all({int(a), int(b)} == {int(c), int(d)}
               for (a, b, _, _), (c, d, _, _) in zip(Z, R)), method
    for k in (2, 3, 5):
        lab = ns["cut"](Z, n, k)
        ref = AgglomerativeClustering(n_clusters=k, linkage=method).fit_predict(X)
        assert adjusted_rand_score(lab, ref) == 1.0, (method, k)
print("hierarchical-clustering OK")
