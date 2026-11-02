import numpy as np, yaml
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "probability-calibration.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
n = 5000
p_true = rng.beta(2, 2, n)
y = (rng.random(n) < p_true).astype(int)
over = 1 / (1 + np.exp(-3 * np.log(p_true / (1 - p_true))))   # overconfident
p_edge = np.r_[p_true[:-4], [0.0, 0.1, 0.5, 1.0]]               # exact bin edges

for p in (p_true, over, p_edge):
    for nb in (5, 10, 15):
        acc, conf, cnt = ns["reliability"](y, p, nb)
        ref_acc, ref_conf = calibration_curve(y, p, n_bins=nb, strategy="uniform")
        assert np.allclose(acc, ref_acc) and np.allclose(conf, ref_conf)
        assert cnt.sum() == len(y)
    assert np.isclose(ns["brier"](y, p), brier_score_loss(y, p))

# ECE equals the hand formula and ranks calibrated < overconfident
acc, conf, cnt = ns["reliability"](y, over, 10)
assert np.isclose(ns["ece"](y, over, 10), (cnt / cnt.sum() * abs(acc - conf)).sum())
assert ns["ece"](y, p_true) < 0.05 < ns["ece"](y, over)
assert ns["brier"](y, p_true) < ns["brier"](y, over)
print("probability-calibration OK")
