import numpy as np, yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "feature-scaling.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
Xtr = rng.normal([5, -3, 100], [1, 10, 1000], (200, 3))
Xtr[:3, 0] += 50                      # outliers
Xtr = np.c_[Xtr, np.full(200, 7.0)]   # constant column
Xte = rng.normal([5, -3, 100], [1, 10, 1000], (50, 3))
Xte = np.c_[Xte, np.full(50, 7.0)]

for name, fit, Ref in [("standard", ns["fit_standard"], StandardScaler),
                       ("minmax", ns["fit_minmax"], MinMaxScaler),
                       ("robust", ns["fit_robust"], RobustScaler)]:
    stats = fit(Xtr)
    ref = Ref().fit(Xtr)
    assert np.allclose(ns["apply"](Xtr, stats), ref.transform(Xtr)), name
    assert np.allclose(ns["apply"](Xte, stats), ref.transform(Xte)), name
    assert np.all(ns["apply"](Xte, stats)[:, 3] == 0), name

# train-only statistics: standardised train has mean 0 and std 1 (non-constant cols)
Z = ns["apply"](Xtr, ns["fit_standard"](Xtr))
assert np.allclose(Z[:, :3].mean(0), 0) and np.allclose(Z[:, :3].std(0), 1)
# min-max train range is exactly [0, 1]; test may fall outside
M = ns["apply"](Xtr, ns["fit_minmax"](Xtr))
assert np.allclose(M[:, :3].min(0), 0) and np.allclose(M[:, :3].max(0), 1)
# robust scaling is much less affected by outliers than min-max
assert ns["apply"](Xtr, ns["fit_robust"](Xtr))[:, 0].std() > M[:, 0].std()
print("feature-scaling OK")
