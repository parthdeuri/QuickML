import numpy as np, yaml
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "isotonic-regression-pava.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(8)
for trial in range(50):
    n = int(rng.integers(2, 80))
    x = np.sort(rng.choice(1000, n, replace=False)).astype(float)
    y = np.cumsum(rng.normal(0.1, 1.0, n)) + rng.normal(0, 2, n)
    w = rng.uniform(0.2, 3.0, n) if trial % 2 else None
    mine = ns["pava"](y, w)
    ref = IsotonicRegression().fit_transform(x, y, sample_weight=w)
    assert np.allclose(mine, ref, atol=1e-10), trial
    assert np.all(np.diff(mine) >= -1e-12)
# an already sorted sequence is unchanged; a decreasing one pools to the mean
assert np.allclose(ns["pava"]([1, 2, 3]), [1, 2, 3])
assert np.allclose(ns["pava"]([3, 2, 1]), [2, 2, 2])
print("isotonic-regression-pava OK")
