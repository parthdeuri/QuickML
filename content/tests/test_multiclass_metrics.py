import numpy as np, yaml
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "multiclass-metrics.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
K = 5
y = rng.choice(K, size=500, p=[0.5, 0.25, 0.15, 0.07, 0.03])
p = np.where(rng.random(500) < 0.7, y, rng.integers(K, size=500))
C = ns["confusion"](y, p, K)
assert np.array_equal(C, confusion_matrix(y, p, labels=range(K)))
f1, macro, weighted, micro = ns["f1_report"](C)
assert np.allclose(f1, f1_score(y, p, average=None, labels=range(K)))
assert np.isclose(macro, f1_score(y, p, average="macro"))
assert np.isclose(weighted, f1_score(y, p, average="weighted"))
assert np.isclose(micro, f1_score(y, p, average="micro"))
assert np.isclose(micro, (y == p).mean())  # micro F1 == accuracy

# a class that is never predicted: F1 = 0 (zero_division=0), no NaNs
p2 = np.where(p == 4, 3, p)
C2 = ns["confusion"](y, p2, K)
f2, mac2, w2, mi2 = ns["f1_report"](C2)
assert np.isfinite(f2).all() and f2[4] == 0
assert np.isclose(mac2, f1_score(y, p2, average="macro", labels=range(K), zero_division=0))
assert np.isclose(w2, f1_score(y, p2, average="weighted", labels=range(K), zero_division=0))
print("multiclass-metrics OK")
