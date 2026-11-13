import numpy as np, yaml
from pathlib import Path
from sklearn.metrics import confusion_matrix

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "fairness-metrics.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# hand computation. A: y=[1,1,0,0], pred=[1,0,0,0]; B: y=[1,1,0,0], pred=[1,1,1,0]
yt = np.array([1, 1, 0, 0, 1, 1, 0, 0])
yp = np.array([1, 0, 0, 0, 1, 1, 1, 0])
g = np.array(list("AAAABBBB"))
r = ns["group_rates"](yt, yp, g)
assert (r["A"]["sel"], r["A"]["tpr"], r["A"]["fpr"]) == (0.25, 0.5, 0.0)
assert (r["B"]["sel"], r["B"]["tpr"], r["B"]["fpr"]) == (0.75, 1.0, 0.5)
assert ns["demographic_parity_diff"](yt, yp, g) == 0.5
assert ns["equal_opportunity_diff"](yt, yp, g) == 0.5
assert ns["equalized_odds_diff"](yt, yp, g) == 0.5
assert abs(ns["disparate_impact"](yt, yp, g) - 1 / 3) < 1e-12

# agree with sklearn confusion matrices on random data
rng = np.random.default_rng(0)
n = 500
g = rng.choice(["A", "B"], n)
yt = (rng.random(n) < np.where(g == "A", 0.3, 0.5)).astype(int)
yp = (rng.random(n) < 0.2 + 0.5 * yt).astype(int)
r = ns["group_rates"](yt, yp, g)
for k in "AB":
    m = g == k
    tn, fp, fn, tp = confusion_matrix(yt[m], yp[m], labels=[0, 1]).ravel()
    assert np.isclose(r[k]["tpr"], tp / (tp + fn))
    assert np.isclose(r[k]["fpr"], fp / (fp + tn))
    assert np.isclose(r[k]["sel"], (tp + fp) / m.sum())
tprs = [r[k]["tpr"] for k in "AB"]; fprs = [r[k]["fpr"] for k in "AB"]
assert np.isclose(ns["equal_opportunity_diff"](yt, yp, g), abs(tprs[0] - tprs[1]))
assert np.isclose(ns["equalized_odds_diff"](yt, yp, g),
                  max(abs(tprs[0] - tprs[1]), abs(fprs[0] - fprs[1])))
sels = [r[k]["sel"] for k in "AB"]
assert np.isclose(ns["disparate_impact"](yt, yp, g), min(sels) / max(sels))

# constructed case: identical label pattern in both groups, predictor ignores the group
yt = np.tile([1, 1, 1, 0, 0, 0, 0, 0], 2)
yp = np.tile([1, 1, 0, 0, 0, 0, 1, 0], 2)  # same function of position, not of group
g = np.array(["A"] * 8 + ["B"] * 8)
assert ns["demographic_parity_diff"](yt, yp, g) == 0
assert ns["equal_opportunity_diff"](yt, yp, g) == 0
assert ns["equalized_odds_diff"](yt, yp, g) == 0
assert ns["disparate_impact"](yt, yp, g) == 1.0

# calibration per group: mean predicted - observed rate
p = np.array([0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.1, 0.1])
y = np.array([1, 0, 1, 0, 1, 1, 0, 0])
gc = ns["group_calibration"](y, p, np.array(list("AAAABBBB")))
assert abs(gc["A"]) < 1e-12 and abs(gc["B"]) < 1e-12

# counterfactual gap
tpl = "The {g} applicant asked for a loan."
assert ns["counterfactual_gap"](lambda s: 0.7, tpl, ["A", "B"]) == 0
assert ns["counterfactual_gap"](lambda s: len(s.split()), tpl, ["A", "B", "C"]) == 0
biased = lambda s: 0.9 if "A" in s.split()[1] else 0.6
assert abs(ns["counterfactual_gap"](biased, tpl, ["A", "B"]) - 0.3) < 1e-12
print("fairness-metrics OK")
