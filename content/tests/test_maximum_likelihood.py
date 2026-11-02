import numpy as np, yaml
from pathlib import Path
from scipy.stats import norm

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "maximum-likelihood.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
x = rng.normal(loc=2.0, scale=1.5, size=400)

mu, var = ns["gaussian_mle"](x)
assert np.isclose(mu, np.mean(x)) and np.isclose(var, np.var(x))
loc, scale = norm.fit(x)  # scipy's Gaussian fit is the MLE
assert np.isclose(mu, loc) and np.isclose(var, scale ** 2)

assert np.isclose(ns["nll"](x, mu, var), -norm.logpdf(x, mu, np.sqrt(var)).sum())
best = ns["nll"](x, mu, var)
for dm, dv in [(0.05, 0), (-0.05, 0), (0, 0.1), (0, -0.1)]:
    assert ns["nll"](x, mu + dm, var + dv) > best
print("maximum-likelihood OK")
