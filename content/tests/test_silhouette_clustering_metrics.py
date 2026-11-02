import numpy as np, yaml
from pathlib import Path
from sklearn import metrics

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "silhouette-clustering-metrics.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(5)
X = np.r_[rng.normal([0, 0], 1.0, (40, 2)), rng.normal([3, 3], 1.0, (50, 2)),
          rng.normal([-3, 4], 1.0, (30, 2))]
truth = np.repeat([0, 1, 2], [40, 50, 30])
pred = truth.copy()
flip = rng.choice(len(X), 25, replace=False)
pred[flip] = rng.integers(0, 3, 25)
pred = (pred + 5) * 7          # arbitrary cluster names
assert np.isclose(ns["silhouette"](X, pred), metrics.silhouette_score(X, pred))
assert np.isclose(ns["ari"](truth, pred), metrics.adjusted_rand_score(truth, pred))
assert np.isclose(ns["nmi"](truth, pred), metrics.normalized_mutual_info_score(truth, pred))
# a singleton cluster scores 0 for that point, as in sklearn
y2 = pred.copy()
y2[0] = 999
assert np.isclose(ns["silhouette"](X, y2), metrics.silhouette_score(X, y2))
# renaming clusters changes nothing; identical labelings score 1
assert np.isclose(ns["ari"](truth, truth + 10), 1)
assert np.isclose(ns["nmi"](truth, truth + 10), 1)
print("silhouette-clustering-metrics OK")
