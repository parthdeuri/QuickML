import math, numpy as np, yaml
from pathlib import Path
from scipy.stats import binom

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "metadata-filtering.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
pre, post, need = ns["prefilter_topk"], ns["postfilter_topk"], ns["needed_candidates"]

rng = np.random.default_rng(0)
N, d, k = 400, 8, 5
X = rng.normal(size=(N, d))
s = 0.2

# pre-filter equals brute force over masked rows
for t in range(20):
    q = rng.normal(size=d)
    mask = rng.random(N) < s
    got = pre(q, X, mask, k)
    allowed = np.flatnonzero(mask)
    ref = allowed[np.argsort(-(X[allowed] @ q), kind="stable")[:k]]
    assert list(got) == list(ref)
    assert mask[got].all()
    for ov in (1, 3, 10):
        assert mask[post(q, X, mask, k, ov)].all()

# constructed: filter keeps only the lowest-scoring rows
Xc = np.arange(20, dtype=float)[:, None]
qc = np.array([1.0])
mc = np.zeros(20, dtype=bool)
mc[:5] = True
assert len(post(qc, Xc, mc, 3, 1)) == 0            # fewer than k (zero)
assert list(pre(qc, Xc, mc, 3)) == [4, 3, 2]        # pre-filter is exact
assert list(post(qc, Xc, mc, 3, 20)) == [4, 3, 2]   # full oversample agrees

# formula arithmetic
assert need(5, 0.2) == 25 and need(10, 0.3) == 34 and need(3, 1.0) == 3

# seeded simulation: success rate vs binomial prediction
trials, hit = {1: 0, 5: 0, 10: 0}, None
T = 2000
for ov in trials:
    for t in range(T):
        q = rng.normal(size=d)
        mask = rng.random(N) < s
        trials[ov] += len(post(q, X, mask, k, ov)) == k
    rate = trials[ov] / T
    c = k * ov
    pred = 1 - binom.cdf(k - 1, c, s)
    assert abs(rate - pred) < 0.05, (ov, rate, pred)
assert trials[1] / T < 0.01
# c = k/s (ov=5) is only about 0.58; c = 2k/s (ov=10) is high
assert 0.5 < trials[5] / T < 0.67 and trials[10] / T > 0.95
print("metadata-filtering OK")
