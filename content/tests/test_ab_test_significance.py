import numpy as np, yaml
from pathlib import Path
from scipy import stats

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "ab-test-significance.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# statsmodels is not required; scipy.stats.norm / chi2 are the references.
x1, n1, x2, n2 = 520, 5000, 590, 5000
z, pval, ci = ns["two_prop_z"](x1, n1, x2, n2)
p1, p2, p = x1 / n1, x2 / n2, (x1 + x2) / (n1 + n2)
z_ref = (p2 - p1) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
assert np.isclose(z, z_ref)
assert np.isclose(pval, 2 * stats.norm.sf(abs(z_ref)), rtol=1e-9)
# z^2 equals the uncorrected chi-square statistic, same p-value
chi2, p_chi, _, _ = stats.chi2_contingency([[x1, n1 - x1], [x2, n2 - x2]], correction=False)
assert np.isclose(z ** 2, chi2) and np.isclose(pval, p_chi, rtol=1e-6)
# CI vs the exact critical value (the card hard-codes a rounded 1.96)
zc = stats.norm.ppf(0.975)
sd = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
assert np.allclose(ci, ((p2 - p1) - zc * sd, (p2 - p1) + zc * sd), atol=1e-4)

# sample size vs the exact-critical-value formula (rounded constants: 2% tolerance)
p0, delta = 0.1, 0.02
n = ns["sample_size"](p0, delta)
za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
n_ref = (za + zb) ** 2 * (p0 * (1 - p0) + (p0 + delta) * (1 - p0 - delta)) / delta ** 2
assert abs(n - n_ref) / n_ref < 0.02, (n, n_ref)

# seeded simulation: size under the null ~ 0.05 (+-0.015), power ~ 0.8 (+-0.03)
rng = np.random.default_rng(0)


def reject_rate(pa, pb, reps=4000):
    a, b = rng.binomial(n, pa, reps), rng.binomial(n, pb, reps)
    pv = [ns["two_prop_z"](int(u), n, int(v), n)[1] for u, v in zip(a, b)]
    return (np.array(pv) < 0.05).mean()


assert abs(reject_rate(p0, p0) - 0.05) < 0.015
assert abs(reject_rate(p0, p0 + delta) - 0.8) < 0.03
print("ab-test-significance OK")
