import numpy as np, yaml
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "precision-recall-f1.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
y = (rng.random(500) < 0.1).astype(int)  # imbalanced
score = y * 0.8 + rng.normal(size=500)
for thr in (0.0, 0.5, 1.0):
    pred = (score > thr).astype(int)
    p, r, f1 = ns["prf1"](y, pred)
    assert np.isclose(p, precision_score(y, pred))
    assert np.isclose(r, recall_score(y, pred))
    assert np.isclose(f1, f1_score(y, pred))

# degenerate: no predicted positives -> convention 0.0 (sklearn zero_division=0)
zero = np.zeros_like(y)
assert ns["prf1"](y, zero) == (0.0, 0.0, 0.0)
assert f1_score(y, zero, zero_division=0) == 0.0
print("precision-recall-f1 OK")
