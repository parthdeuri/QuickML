import numpy as np, yaml
from pathlib import Path
from scipy import stats

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "probability-distributions-sampling.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)

# pmf / pdf equal scipy
for n, p in [(1, 0.3), (10, 0.4), (25, 0.8)]:
    for k in range(n + 1):
        assert np.isclose(ns["binomial_pmf"](k, n, p), stats.binom.pmf(k, n, p))
for lam in (0.5, 3.0, 9.0):
    for k in range(20):
        assert np.isclose(ns["poisson_pmf"](k, lam), stats.poisson.pmf(k, lam))
x = np.linspace(-4, 6, 41)
assert np.allclose(ns["gaussian_pdf"](x, 1.0, 1.7), stats.norm.pdf(x, 1.0, 1.7))
xe = np.linspace(0, 5, 26)
assert np.allclose(ns["exponential_pdf"](xe, 1.3), stats.expon.pdf(xe, scale=1 / 1.3))

# sample moments match theory
N = 200_000
z = ns["box_muller"](N, rng)
assert abs(z.mean()) < 0.01 and abs(z.var() - 1) < 0.02
lam = 2.5
e = ns["sample_exponential"](lam, N, rng)
assert abs(e.mean() - 1 / lam) < 0.005 and abs(e.var() - 1 / lam**2) < 0.005

# Kolmogorov-Smirnov tests
assert stats.kstest(z, "norm").pvalue > 0.01
assert stats.kstest(e, "expon", args=(0, 1 / lam)).pvalue > 0.01

# categorical frequencies pass a chi-square test
p = np.array([0.1, 0.2, 0.05, 0.4, 0.25])
s = ns["sample_categorical"](p, N, rng)
assert s.min() == 0 and s.max() == len(p) - 1
obs = np.bincount(s, minlength=len(p))
assert stats.chisquare(obs, p * N).pvalue > 0.01
# near-1 draw never indexes out of range even if cumsum falls short
q = np.array([0.1] * 9 + [0.0999999])
assert ns["sample_categorical"](q, 1000, rng).max() <= 9

# Binomial = sum of Bernoulli draws; Bernoulli is Binomial with n = 1
n, pb = 12, 0.35
bern = (rng.random((N, n)) < pb).sum(axis=1)
freq = np.bincount(bern, minlength=n + 1) / N
theory = np.array([ns["binomial_pmf"](k, n, pb) for k in range(n + 1)])
assert np.abs(freq - theory).max() < 0.005
assert ns["binomial_pmf"](1, 1, pb) == pb and np.isclose(ns["binomial_pmf"](0, 1, pb), 1 - pb)
print("probability-distributions-sampling OK")
