import numpy as np, yaml
from pathlib import Path
from sklearn.cluster import KMeans

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "ann-vector-index.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
N, d, nlist, k = 2000, 8, 16, 10
cent = rng.normal(size=(nlist, d)) * 3
X = cent[rng.integers(0, nlist, N)] + rng.normal(size=(N, d)) * 1.5
Q = cent[rng.integers(0, nlist, 40)] + rng.normal(size=(40, d)) * 1.5
idx = ns["build_ivf"](X, nlist, seed=0)

# every vector appears in exactly one cell
allids = np.concatenate(idx["cells"])
assert len(allids) == N and sorted(allids.tolist()) == list(range(N))

recalls = []
for p in range(1, nlist + 1):
    rs = []
    for q in Q:
        exact = np.argsort(((X - q) ** 2).sum(1), kind="stable")[:k]
        ids, cost = ns["search_ivf"](idx, q, k, p)
        rs.append(ns["recall_at_k"](ids, exact))
        # distance evaluations = centroid ranking + scanned cell sizes
        order = np.argsort(((idx["C"] - q) ** 2).sum(1))[:p]
        assert cost == nlist + sum(len(idx["cells"][j]) for j in order)
        if p == nlist:
            assert list(ids) == list(exact)  # exactly brute force
    recalls.append(np.mean(rs))
assert all(b >= a - 1e-9 for a, b in zip(recalls, recalls[1:])), recalls
assert recalls[-1] == 1.0
assert recalls[0] < 1.0  # approximation really loses something

# centroid quality: same order of magnitude as sklearn k-means
def inertia(C):
    return ns["dist2"](X, C).min(1).sum()
sk = KMeans(nlist, n_init=3, random_state=0).fit(X).inertia_
mine = inertia(idx["C"])
assert 0.5 * sk < mine < 3 * sk, (mine, sk)
print("ann-vector-index OK")
