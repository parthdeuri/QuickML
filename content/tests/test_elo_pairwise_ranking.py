import numpy as np, yaml
from pathlib import Path
from scipy.optimize import minimize

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "elo-pairwise-ranking.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
expected, elo_update, bt_fit, bt_to_elo = (ns[k] for k in ("expected", "elo_update", "bt_fit", "bt_to_elo"))

# expected scores
for ra, rb in [(1500, 1500), (1700, 1300), (1000, 1800)]:
    assert np.isclose(expected(ra, rb) + expected(rb, ra), 1.0)
assert expected(1234, 1234) == 0.5
assert np.isclose(expected(1900, 1500), 10 / 11)   # 400-point gap -> 10:1 odds
# zero-sum update, direction correct
rng = np.random.default_rng(0)
for _ in range(50):
    ra, rb, s = rng.uniform(800, 2200), rng.uniform(800, 2200), rng.choice([0, 0.5, 1])
    na, nb = elo_update(ra, rb, s)
    assert np.isclose(na + nb, ra + rb)
na, nb = elo_update(1500, 1500, 1)
assert na == 1516 and nb == 1484

# Bradley-Terry recovery from a seeded simulation
M, G = 6, 20000
true = np.exp(rng.normal(0, 0.7, M))
W = np.zeros((M, M))
for i in range(M):
    for j in range(i + 1, M):
        w = rng.binomial(G, true[i] / (true[i] + true[j]))
        W[i, j], W[j, i] = w, G - w
p = bt_fit(W, 500)
lt = np.log(true) - np.log(true).mean()
assert np.allclose(np.log(p), lt, atol=0.05), (np.log(p), lt)
assert np.isclose(np.exp(np.log(p).mean()), 1.0)
assert np.allclose(bt_to_elo(p), 400 * np.log10(p))

def loglik(logp):
    q = np.exp(logp)
    return (W * (logp[:, None] - np.log(q[:, None] + q[None, :]))).sum()

ll_fit = loglik(np.log(p))
assert ll_fit >= loglik(np.log(true)) - 1e-6
res = minimize(lambda z: -loglik(np.r_[0.0, z]), np.zeros(M - 1), method="BFGS", options={"gtol": 1e-9})
ref = np.r_[0.0, res.x]
ref = ref - ref.mean()
assert np.isclose(-res.fun, ll_fit, rtol=0, atol=1e-4), (-res.fun, ll_fit)
assert np.allclose(np.log(p), ref, atol=1e-3)
# order independence of BT vs dependence of Elo
games = [(0, 1, 1), (1, 2, 1), (0, 2, 1), (2, 0, 1), (1, 0, 1)] * 5
def run_elo(seq):
    r = [1500.0] * 3
    for a, b, s in seq:
        r[a], r[b] = elo_update(r[a], r[b], s)
    return np.array(r)
assert not np.allclose(run_elo(games), run_elo(games[::-1]))
print("elo-pairwise-ranking OK")
