import numpy as np, yaml
from pathlib import Path
from sklearn.metrics import precision_recall_curve

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "semantic-cache.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
SC, tune, sav = ns["SemanticCache"], ns["tune_threshold"], ns["expected_savings"]


class Clock:
    def __init__(self):
        self.t = 0.0
    def __call__(self):
        return self.t


def unit_at(cos):
    # unit vector whose cosine with e0 = [1, 0] is `cos`
    return np.array([cos, np.sqrt(1 - cos**2)])


tau = 0.9
clk = Clock()
cache = SC(tau, 100.0, clk)
e0 = np.array([1.0, 0.0])
cache.put("a", e0, "answer-0")
# exact repeat hits
assert cache.get("a", e0) == "answer-0"
# constructed paraphrase above tau hits, near-miss just below misses
assert cache.get("a", unit_at(tau + 0.01)) == "answer-0"
assert cache.get("a", unit_at(tau - 0.01)) is None
# orthogonal query misses
assert cache.get("a", np.array([0.0, 1.0])) is None
# tenant isolation: same vector, different tenant misses
assert cache.get("b", e0) is None
# TTL expiry with a fake clock
clk.t = 99.0
assert cache.get("a", e0) == "answer-0"
clk.t = 100.0
assert cache.get("a", e0) is None
# nearest entry wins
cache.put("a", unit_at(0.95), "close")
cache.put("a", e0, "exact")
assert cache.get("a", e0) == "exact"

# tune_threshold agrees with the precision_recall_curve computation
rng = np.random.default_rng(0)
n = 400
labels = rng.random(n) < 0.5
scores = np.clip(np.where(labels, rng.normal(0.85, 0.08, n),
                          rng.normal(0.6, 0.12, n)), -1, 1)
prec, _, thr = precision_recall_curve(labels, scores)
for target in (0.8, 0.9, 0.95, 0.99):
    got = tune(scores, labels, target)
    ok = thr[prec[:-1] >= target]
    ref = ok.min() if len(ok) else None
    assert (got is None and ref is None) or abs(got - ref) < 1e-12, (target, got, ref)
    if got is not None:
        assert labels[scores >= got].mean() >= target
        lower = scores[scores < got]
        for t in np.unique(lower):
            assert labels[scores >= t].mean() < target
assert tune([0.9, 0.8], [0, 0], 0.5) is None
# higher precision targets never lower the threshold
ts = [tune(scores, labels, p) for p in (0.8, 0.9, 0.95)]
assert ts == sorted(ts)

# savings arithmetic
assert abs(sav(0.3, 0.02, 0.001) - 0.005) < 1e-12
assert sav(0.04, 0.02, 0.001) < 0          # hit rate below c_lookup / c_llm = 0.05
assert abs(sav(0.05, 0.02, 0.001)) < 1e-12  # exactly break-even
print("semantic-cache OK")
