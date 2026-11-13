import numpy as np, yaml
from pathlib import Path
from scipy.spatial import distance as D

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "vectors-norms-distances.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
norm, unit, cosd, mahal = ns["norm"], ns["unit"], ns["cosine_dist"], ns["mahalanobis"]

rng = np.random.default_rng(0)
x, y = rng.normal(size=6), rng.normal(size=6)

# norms equal numpy / scipy
for p in (1, 2, 3, 0.5):
    assert np.isclose(norm(x, p), (np.abs(x) ** p).sum() ** (1 / p))
for p in (1, 2, np.inf):
    assert np.isclose(norm(x, p), np.linalg.norm(x, p))
for p in (1, 2, 3):
    assert np.isclose(norm(x - y, p), D.minkowski(x, y, p))
assert np.isclose(norm(x - y, np.inf), D.chebyshev(x, y))
assert np.isclose(norm(unit(x), 2), 1.0)

# cosine distance: scipy, range, scale invariance, orthogonal / opposite
assert np.isclose(cosd(x, y), D.cosine(x, y))
assert np.isclose(cosd(x, 5 * x), 0.0) and np.isclose(cosd(x, -x), 2.0)
assert np.isclose(cosd(np.array([1.0, 0]), np.array([0, 2.0])), 1.0)
assert np.isclose(cosd(3 * x, 0.1 * y), cosd(x, y))

# Mahalanobis equals scipy, and reduces to Euclidean for identity cov
A = rng.normal(size=(6, 6))
cov = A @ A.T + 0.5 * np.eye(6)
assert np.isclose(mahal(x, y, cov), D.mahalanobis(x, y, np.linalg.inv(cov)))
assert np.isclose(mahal(x, y, np.eye(6)), np.linalg.norm(x - y))
# whitening: Euclidean distance after Sigma^{-1/2} is the same number
w, V = np.linalg.eigh(cov)
W = V @ np.diag(w ** -0.5) @ V.T
assert np.isclose(mahal(x, y, cov), np.linalg.norm(W @ (x - y)))

# unit-vector identity ||u-v||^2 = 2 - 2cos(u,v), and equal rankings
U = np.stack([unit(r) for r in rng.normal(size=(200, 8))])
q = unit(rng.normal(size=8))
sq = ((U - q) ** 2).sum(1)
cs = U @ q
assert np.allclose(sq, 2 - 2 * cs)
assert (np.argsort(sq) == np.argsort(-cs)).all()

# concentration: max/min distance ratio shrinks with dimension
def ratio(dim, n=500, seed=0):
    r = np.random.default_rng(seed)
    X, qq = r.normal(size=(n, dim)), r.normal(size=dim)
    dist = np.linalg.norm(X - qq, axis=1)
    return dist.max() / dist.min()
rs = [ratio(d) for d in (2, 10, 100, 1000)]
assert all(a > b for a, b in zip(rs, rs[1:])), rs
assert rs[0] > 20 and rs[-1] < 1.5, rs
print("ratios by dim 2,10,100,1000:", [round(v, 3) for v in rs])
print("vectors-norms-distances OK")
