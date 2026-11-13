import math, yaml
from pathlib import Path
from scipy.stats import binomtest

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "finetune-evaluation.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# sign test equals scipy exact binomial test
for w in range(0, 21):
    for l in range(0, 21):
        if w + l == 0:
            assert ns["sign_test_p"](w, l) == 1.0
            continue
        ref = binomtest(w, w + l, 0.5).pvalue
        assert math.isclose(ns["sign_test_p"](w, l), ref, rel_tol=1e-9, abs_tol=1e-12), (w, l)
assert math.isclose(ns["sign_test_p"](9, 1), 22 / 1024)  # 2*(1+10)/2^10

# ties hand cases: 6 wins, 2 losses, 2 ties
r = ns["win_rate"]([1] * 6 + [-1] * 2 + [0] * 2)
assert (r["wins"], r["losses"], r["ties"]) == (6, 2, 2)
assert math.isclose(r["rate_ties_half"], 0.7)
assert math.isclose(r["rate_excl_ties"], 0.75)
r = ns["win_rate"]([0, 0, 0, 0])
assert r["rate_ties_half"] == 0.5 and math.isnan(r["rate_excl_ties"])

# required n: 1.96^2 * .25 / .05^2 = 384.16 -> 385
assert ns["required_n"](0.5, 0.05) == 385
assert ns["required_n"](0.5, 0.025) == 1537
assert ns["required_n"](0.5, 0.1, z=1.0) == 25
assert ns["required_n"](0.8, 0.05, z=2.0) == 256

# Wilson interval matches scipy's Wilson CI
lo, hi = ns["wilson_ci"](30, 50)
ci = binomtest(30, 50).proportion_ci(0.95, method="wilson")
assert math.isclose(lo, ci.low, abs_tol=1e-4) and math.isclose(hi, ci.high, abs_tol=1e-4)

# forgetting: negative means capability lost
assert math.isclose(ns["forgetting_delta"](0.70, 0.65), -0.05)
assert ns["forgetting_delta"](0.5, 0.5) == 0
print("finetune-evaluation OK")
