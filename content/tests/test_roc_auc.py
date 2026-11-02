import numpy as np, yaml
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "roc-auc.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
y = (rng.random(300) < 0.3).astype(int)
for score in (y * 0.9 + rng.normal(size=300),
              np.round(y * 0.9 + rng.normal(size=300), 1)):  # heavy ties
    fpr, tpr = ns["roc_curve"](y, score)
    rf, rt, _ = roc_curve(y, score, drop_intermediate=False)
    assert np.allclose(rf, fpr) and np.allclose(rt, tpr)
    auc = ns["auc"](fpr, tpr)
    assert np.isclose(auc, roc_auc_score(y, score))
    # rank interpretation: P(pos > neg) + 0.5 * P(tie)
    pos, neg = score[y == 1][:, None], score[y == 0][None, :]
    pairs = np.mean((pos > neg) + 0.5 * (pos == neg))
    assert np.isclose(auc, pairs)
print("roc-auc OK")
