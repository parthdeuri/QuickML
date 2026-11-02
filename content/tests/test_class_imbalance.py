import numpy as np, yaml
from pathlib import Path
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_class_weight

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "class-imbalance.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
y = (rng.random(500) < 0.08).astype(int)
p = np.clip(rng.random(500), 0.01, 0.99)

w = ns["balanced_weights"](y)
classes = np.unique(y)
ref = compute_class_weight("balanced", classes=classes, y=y)
assert np.allclose([w[c] for c in classes], ref)
# every class carries equal total weight
tot = [w[c] * (y == c).sum() for c in classes]
assert np.allclose(tot, tot[0])

sw = np.array([w[c] for c in y])
assert np.isclose(ns["weighted_logloss"](y, p, w), log_loss(y, p, sample_weight=sw))
assert not np.isclose(ns["weighted_logloss"](y, p, {0: 1, 1: 1}), ns["weighted_logloss"](y, p, w))

# 3-class case too
y3 = rng.choice(3, 300, p=[0.7, 0.2, 0.1])
w3 = ns["balanced_weights"](y3)
assert np.allclose([w3[c] for c in range(3)], compute_class_weight("balanced", classes=np.arange(3), y=y3))

# random oversampling demo
X = np.arange(len(y) * 2, dtype=float).reshape(-1, 2)
Xo, yo = ns["oversample"](X, y, np.random.default_rng(1))
counts = np.bincount(yo)
assert counts[0] == counts[1] == np.bincount(y).max()
assert set(map(tuple, Xo)) == set(map(tuple, X))        # only original rows are reused
assert all((yo[np.all(Xo == X[i], 1)] == y[i]).all() for i in range(0, 500, 50))
assert Xo.shape[0] == 2 * np.bincount(y).max()
print("class-imbalance OK")
