import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "metropolis-hastings.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

mean, sd = 2.0, 1.5


def logp(x):  # unnormalised log density of N(2, 1.5^2)
    return -0.5 * ((x - mean) / sd) ** 2


n, burn = 200000, 5000
out, rate = ns["metropolis"](logp, 10.0, n, step=2.4 * sd, seed=0)
s = out[burn:]
# stated tolerances: mean within 0.05, variance within 5%, acceptance moderate
assert abs(s.mean() - mean) < 0.05, s.mean()
assert abs(s.var() / sd ** 2 - 1) < 0.05, s.var()
assert 0.3 < rate < 0.7, rate

# reproducible for a fixed seed (prefix of the chain is identical)
out2, _ = ns["metropolis"](logp, 10.0, 1000, step=2.4 * sd, seed=0)
assert np.array_equal(out2, out[:1000])

# extremes: tiny steps accept almost always, huge steps almost never
_, hi = ns["metropolis"](logp, 2.0, 20000, step=0.01, seed=1)
_, lo = ns["metropolis"](logp, 2.0, 20000, step=200.0, seed=1)
assert hi > 0.95 and lo < 0.05


# exact detailed balance of the acceptance rule: pi(x) a(x->y) = pi(y) a(y->x)
def dens(x):
    return np.exp(logp(x)) / (sd * np.sqrt(2 * np.pi))


def acc(x, y):
    return min(1.0, np.exp(logp(y) - logp(x)))


rng = np.random.default_rng(0)
for x, y in rng.normal(mean, 3 * sd, (200, 2)):
    assert np.isclose(dens(x) * acc(x, y), dens(y) * acc(y, x), rtol=1e-12, atol=0)
print("metropolis-hastings OK")
