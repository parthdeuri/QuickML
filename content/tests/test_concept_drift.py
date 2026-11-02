import numpy as np, yaml
from pathlib import Path
from scipy.stats import entropy, ks_2samp

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "concept-drift.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
psi, ks = ns["psi"], ns["ks_stat"]
rng = np.random.default_rng(0)
ref = rng.normal(0, 1, 5000)
same = rng.normal(0, 1, 5000)
shift = rng.normal(0.8, 1, 5000)
wide = rng.normal(0, 2, 5000)

# KS statistic equals scipy's, including ties
for a, b in [(ref, same), (ref, shift), (ref, wide), (np.round(ref, 1), np.round(shift, 1))]:
    assert np.isclose(ks(a, b), ks_2samp(a, b).statistic)

# PSI equals the symmetric KL divergence of the clipped bin shares
def shares(r, c, bins=10, eps=1e-4):
    e = np.quantile(r, np.linspace(0, 1, bins + 1)); e[0], e[-1] = -np.inf, np.inf
    a = np.clip(np.histogram(r, e)[0] / len(r), eps, None)
    b = np.clip(np.histogram(c, e)[0] / len(c), eps, None)
    return a, b

for cur in (same, shift, wide):
    a, b = shares(ref, cur)
    assert np.isclose(psi(ref, cur), entropy(a, b) + entropy(b, a))
assert np.isclose(psi(ref, ref), 0)
assert psi(ref, same) < psi(ref, shift) and psi(ref, same) < psi(ref, wide)
assert ks(ref, same) < ks(ref, shift)
# values beyond the reference range still fall in the outer bins
assert np.isfinite(psi(ref, ref + 100))

# Concept drift: P(y|x) flips while P(x) is unchanged -> input statistics are blind to it
x_old, x_new = rng.normal(size=4000), rng.normal(size=4000)
y_new = (x_new < 0).astype(int)          # rule reversed
rule = lambda x: (x > 0).astype(int)     # model learned on old data
assert (rule(x_old) == (x_old > 0)).all() and (rule(x_new) == y_new).mean() < 0.01
assert psi(x_old, x_new) < psi(ref, shift) / 5
assert ks(x_old, x_new) < ks(ref, shift) / 3
print("concept-drift OK")
