import numpy as np, yaml
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "statistics-hypothesis-testing.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)

# Welch t equals scipy (unequal sizes and variances)
for _ in range(20):
    x = rng.normal(0.0, 1.0, rng.integers(8, 40))
    y = rng.normal(0.3, 3.0, rng.integers(8, 60))
    t, df, p = ns["welch_t"](x, y)
    r = stats.ttest_ind(x, y, equal_var=False)
    assert np.isclose(t, r.statistic) and np.isclose(p, r.pvalue)
    assert np.isclose(df, r.df)

# BH and Bonferroni equal statsmodels
for _ in range(50):
    m = int(rng.integers(5, 200))
    pv = np.concatenate([rng.uniform(size=m), rng.uniform(0, 0.002, size=m // 4)])
    for alpha in (0.01, 0.05, 0.2):
        for name, fn in (("fdr_bh", "bh_fdr"), ("bonferroni", "bonferroni")):
            ref = multipletests(pv, alpha=alpha, method=name)[0]
            assert (ns[fn](pv, alpha) == ref).all(), (name, alpha)
assert not ns["bh_fdr"]([0.9, 0.8], 0.05).any()

# bootstrap CI for the mean covers the true mean about 95% of the time
hits, trials = 0, 400
for s in range(trials):
    x = np.random.default_rng(1000 + s).normal(5.0, 2.0, 50)
    lo, hi = ns["bootstrap_ci"](x, np.mean, 300, 0.05, s)
    hits += lo <= 5.0 <= hi
assert 0.91 <= hits / trials <= 0.98, hits / trials

# under a true null the p-values are about uniform, so rejection rate ~ alpha
ps = np.array([ns["welch_t"](rng.normal(size=30), rng.normal(size=25))[2] for _ in range(3000)])
assert abs((ps <= 0.05).mean() - 0.05) < 0.015
assert stats.kstest(ps, "uniform").pvalue > 0.001
print("statistics-hypothesis-testing OK")
