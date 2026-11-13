import numpy as np, yaml
from pathlib import Path
from sklearn.metrics import precision_recall_curve

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "content-safety-classifiers.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
pick, conf, cost, band = (ns[k] for k in ("pick_threshold", "confusion_at", "cascade_cost", "escalate_band"))

rng = np.random.default_rng(0)
n = 400
y = (rng.random(n) < 0.2).astype(int)
s = np.clip(rng.normal(0.3 + 0.4 * y, 0.18), 0, 1)
scores, labels = s.tolist(), y.tolist()
P = int(y.sum())

for target in [0.5, 0.8, 0.9, 0.95, 0.99, 1.0]:
    thr = pick(scores, labels, target)
    # brute force: highest candidate threshold whose recall >= target
    best = max(t for t in set(scores)
               if conf(scores, labels, t)[0] / P >= target)
    assert thr == best, (target, thr, best)
    # sklearn precision_recall_curve gives the same answer
    prec, rec, th = precision_recall_curve(y, s)
    ok = th[rec[:-1] >= target - 1e-12]
    assert abs(thr - ok.max()) < 1e-12, (target, thr, ok.max())
    tp, fp, fn, tn = conf(scores, labels, thr)
    assert tp / (tp + fn) >= target - 1e-12
    assert tp + fp + fn + tn == n
    # a slightly higher threshold would miss the target
    higher = min(t for t in set(scores) if t > thr) if thr < max(scores) else None
    if higher is not None:
        assert conf(scores, labels, higher)[0] / P < target

# hand case: positives 0.9 0.8 0.6 0.4, target 0.75 -> k=3 -> thr 0.6
sc = [0.9, 0.1, 0.8, 0.6, 0.3, 0.4, 0.2]
lb = [1, 0, 1, 1, 0, 1, 0]
assert pick(sc, lb, 0.75) == 0.6
assert conf(sc, lb, 0.6) == (3, 0, 1, 3)
assert pick(sc, lb, 1.0) == 0.4 and pick(sc, lb, 0.25) == 0.9

# cascade cost hand checks and exact break-even condition p < 1 - c1/c2
assert cost(1, 10, 0.25) == 3.5
assert cost(1, 10, 0.0) == 1
c1, c2 = 1.0, 10.0
pstar = 1 - c1 / c2
assert abs(pstar - 0.9) < 1e-12
assert cost(c1, c2, 0.89) < c2
assert cost(c1, c2, 0.91) > c2
assert abs(cost(c1, c2, pstar) - c2) < 1e-12
for p in [0.0, 0.3, 0.6, 0.85]:
    assert p < pstar and cost(c1, c2, p) < c2

# band selection: lo inclusive, hi exclusive
assert band([0.1, 0.4, 0.5, 0.7, 0.9], 0.4, 0.7) == [1, 2]
assert band([0.1, 0.9], 0.4, 0.7) == []
assert band([0.4, 0.7], 0.4, 0.7) == [0]
print("content-safety-classifiers OK")
