import numpy as np, yaml
from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "local-outlier-factor.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(6)
X = np.r_[rng.normal(0, 0.3, (60, 2)), rng.normal(5, 1.5, (60, 2)),
          [[2.5, 2.5], [9.0, -2.0]]]
X = np.r_[X, X[:2]]      # exact duplicates exercise the zero-distance case
for k in (5, 10, 20):
    mine = ns["lof"](X, k)
    ref = -LocalOutlierFactor(n_neighbors=k).fit(X).negative_outlier_factor_
    assert np.allclose(mine, ref, rtol=1e-6), k
# the planted points rank among the top scores, typical inliers are near 1
s = ns["lof"](X, 10)
assert {120, 121} <= set(np.argsort(s)[-5:])
assert s[120] > 1.5 and s[121] > 1.5 and np.median(s[:120]) < 1.3
print("local-outlier-factor OK")
