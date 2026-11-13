import itertools, numpy as np, yaml
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.special import softmax

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "numpy-broadcasting-vectorisation.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# broadcast_shape equals numpy on many pairs, including errors
rng = np.random.default_rng(0)
cases = [((), ()), ((3,), (4,)), ((4, 3), (3,)), ((4, 3), (4,)), ((5, 1), (1, 6)), ((0,), (1,)), ((2, 1, 3), (4, 1))]
for _ in range(300):
    cases.append(tuple(tuple(int(v) for v in rng.choice([1, 2, 3], size=rng.integers(0, 5))) for _ in range(2)))
n_err = 0
for a, b in cases:
    try:
        want = np.broadcast_shapes(a, b)
    except ValueError:
        want = None
    try:
        got = ns["broadcast_shape"](a, b)
    except ValueError:
        got = None
    assert got == want, (a, b, got, want)
    n_err += want is None
assert 0 < n_err < len(cases)

# distances: broadcasting, identity, scipy and naive loop agree
A, B = rng.normal(size=(7, 4)), rng.normal(size=(5, 4))
ref = cdist(A, B, "sqeuclidean")
naive = np.array([[sum((A[i, k] - B[j, k]) ** 2 for k in range(4)) for j in range(5)] for i in range(7)])
assert np.allclose(naive, ref)
assert np.allclose(ns["pairwise_sq_dists"](A, B), ref)
assert np.allclose(ns["pairwise_sq_dists_dot"](A, B), ref)
assert ns["pairwise_sq_dists"](A, B).shape == (7, 5)

# softmax, including large values that would overflow without the shift
X = rng.normal(size=(6, 9)) * 5
X[0] += 1000
assert np.allclose(ns["softmax_rows"](X), softmax(X, axis=1))
assert np.allclose(ns["softmax_rows"](X).sum(axis=1), 1)
print("numpy-broadcasting-vectorisation OK")
