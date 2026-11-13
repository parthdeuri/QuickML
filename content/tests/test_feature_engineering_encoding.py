import numpy as np, yaml
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "feature-engineering-encoding.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)

# one_hot equals sklearn
vals = rng.choice(["a", "b", "c", "d"], 50)
cats = sorted(set(vals))
ref = OneHotEncoder(categories=[cats], sparse_output=False).fit_transform(vals[:, None])
assert np.array_equal(ns["one_hot"](vals, cats), ref)

# equal_width_bins equals sklearn uniform KBinsDiscretizer
x = rng.exponential(2.0, 200)
for k in (3, 5, 10):
    ref = KBinsDiscretizer(k, encode="ordinal", strategy="uniform").fit_transform(x[:, None])[:, 0]
    assert np.array_equal(ns["equal_width_bins"](x, k), ref.astype(int))

# cyclical: 23h close to 0h, far from 12h
c = ns["cyclical"](np.array([0, 23, 12]), 24)
assert np.allclose(np.linalg.norm(c, axis=1), 1)
d = lambda i, j: np.linalg.norm(c[i] - c[j])
assert d(0, 1) < 0.3 and d(0, 2) > 1.9
assert np.isclose(d(0, 1), 2 * abs(np.sin(np.pi * 1 / 24)))

# leakage demo: random labels, high-cardinality feature
n = 2000
cat = rng.integers(0, 1000, n)
y = rng.integers(0, 2, n).astype(float)

def naive(cat, y):  # encoding computed on all rows, includes own label
    return np.array([y[cat == v].mean() for v in cat])

def auc(score, y):
    r = score.argsort().argsort() + 1
    pos = y == 1
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())

a_naive = auc(naive(cat, y) + 1e-9 * rng.normal(size=n), y)
oof = ns["target_encode_oof"](cat, y, 5, 5.0, 0)
a_oof = auc(oof + 1e-9 * rng.normal(size=n), y)
assert a_naive > 0.8, a_naive
assert abs(a_oof - 0.5) < 0.06, a_oof

# smoothing: unseen category gets the prior, true signal is still found
r2 = np.random.default_rng(5)
cat2 = r2.integers(0, 5, 5000)
y2 = (r2.random(5000) < 0.1 + 0.15 * cat2).astype(float)
e2 = ns["target_encode_oof"](cat2, y2, 5, 10.0, 1)
for v in range(5):
    assert abs(e2[cat2 == v].mean() - (0.1 + 0.15 * v)) < 0.03
print("feature-engineering-encoding OK")
