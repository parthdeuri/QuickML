import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "speculative-decoding.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
step, acc, exp_tok = ns["speculative_step"], ns["accept_or_resample"], ns["expected_tokens"]

p = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
qs = {
    "close": np.array([0.35, 0.3, 0.2, 0.1, 0.05]),
    "uniform": np.full(5, 0.2),
    "bad": np.array([0.02, 0.03, 0.05, 0.1, 0.8]),
    "same": p.copy(),
}
p_fn = lambda prefix: p
CRIT = 18.47  # chi-square, df=4, 99.9th percentile

# 1) first emitted token ~ p for every draft q
N = 40000
for name, q in qs.items():
    for gamma in (1, 3):
        rng = np.random.default_rng(1)
        q_fn = lambda prefix, q=q: q
        first = np.array([step(p_fn, q_fn, [], gamma, rng)[0] for _ in range(N)])
        obs = np.bincount(first, minlength=5)
        chi2 = (((obs - N * p) ** 2) / (N * p)).sum()
        assert chi2 < CRIT, (name, gamma, chi2)
        assert np.abs(obs / N - p).max() < 0.012, (name, gamma)

# context-dependent target and draft: still exact for the first token
T = np.random.default_rng(5).dirichlet(np.ones(5), size=5)
Q = np.random.default_rng(6).dirichlet(np.ones(5) * 0.5, size=5)
pc = lambda s: T[s[-1]]
qc = lambda s: Q[s[-1]]
rng = np.random.default_rng(2)
first = np.array([step(pc, qc, [3], 4, rng)[0] for _ in range(N)])
obs = np.bincount(first, minlength=5)
assert (((obs - N * T[3]) ** 2) / (N * T[3])).sum() < CRIT

# 2) residual distribution sums to 1 and rejected tokens follow it
for q in qs.values():
    r = np.maximum(p - q, 0)
    if r.sum() > 0:
        assert np.isclose((r / r.sum()).sum(), 1.0) and (r >= 0).all()
rng = np.random.default_rng(3)
q = qs["bad"]
outs = []
for _ in range(20000):
    t, ok = acc(p, q, 4, rng)
    if not ok:
        outs.append(t)
r = np.maximum(p - q, 0)
r /= r.sum()
fr = np.bincount(outs, minlength=5) / len(outs)
assert np.abs(fr - r).max() < 0.02 and fr[4] == 0

# 3) empirical tokens/step vs formula with alpha estimated from data
for name in ("close", "uniform", "bad"):
    q = qs[name]
    q_fn = lambda prefix, q=q: q
    rng = np.random.default_rng(4)
    gamma, M = 4, 20000
    lens = np.array([len(step(p_fn, q_fn, [], gamma, rng)) for _ in range(M)])
    n_acc = lens - 1  # accepted drafts per step
    tried = np.where(n_acc < gamma, n_acc + 1, gamma)  # proposals checked
    alpha_hat = n_acc.sum() / tried.sum()
    assert abs(alpha_hat - np.minimum(p, q).sum()) < 0.01
    assert abs(lens.mean() - exp_tok(alpha_hat, gamma)) < 0.03, name
    assert abs(lens.mean() - exp_tok(np.minimum(p, q).sum(), gamma)) < 0.05
assert exp_tok(1.0, 4) == 5 and np.isclose(exp_tok(0.0, 4), 1.0)
assert np.isclose(exp_tok(0.5, 2), 1 + 0.5 + 0.25)

# 4) gamma = 0 is plain sampling from the target
rng = np.random.default_rng(7)
out = [step(p_fn, lambda s: qs["bad"], [], 0, rng) for _ in range(N)]
assert all(len(o) == 1 for o in out)
obs = np.bincount([o[0] for o in out], minlength=5)
assert (((obs - N * p) ** 2) / (N * p)).sum() < CRIT
print("speculative-decoding OK")
