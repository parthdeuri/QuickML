import numpy as np, yaml
from pathlib import Path
from scipy.special import rel_entr

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "rlhf-kl-penalty.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
p = rng.dirichlet(5 * np.ones(6))
q = 0.7 * p + 0.3 * rng.dirichlet(5 * np.ones(6))
kl = ns["exact_kl"](p, q)
assert np.isclose(kl, rel_entr(p, q).sum())

x = rng.choice(6, size=200000, p=p)
k1, k2, k3 = ns["kl_estimators"](np.log(p[x]), np.log(q[x]))
se = lambda a: a.std() / np.sqrt(len(a))
assert abs(k1.mean() - kl) < 4 * se(k1), (k1.mean(), kl)
assert abs(k3.mean() - kl) < 4 * se(k3), (k3.mean(), kl)
assert k3.var() < k1.var(), (k3.var(), k1.var())
assert (k3 >= 0).all() and (k1 < 0).any()
# exact expectations
lr = np.log(q / p)
assert np.isclose((p * -lr).sum(), kl)
assert np.isclose((p * ((np.exp(lr) - 1) - lr)).sum(), kl)

# shaped rewards: per-token KL rewards sum to -beta * sequence log-ratio
T, beta = 7, 0.1
lp = rng.normal(-2, 0.5, T)
lq = rng.normal(-2, 0.5, T)
r = ns["shaped_rewards"](lp, lq, 0.0, beta)
assert np.isclose(r.sum(), -beta * (lp.sum() - lq.sum()))
r2 = ns["shaped_rewards"](lp, lq, 1.5, beta)
assert np.isclose(r2[-1] - r[-1], 1.5) and np.allclose(r2[:-1], r[:-1])
assert np.isclose(r2.sum(), -beta * (lp.sum() - lq.sum()) + 1.5)
print("rlhf-kl-penalty OK")
