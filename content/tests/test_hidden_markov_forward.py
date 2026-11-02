import itertools
import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "hidden-markov-forward.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# Tiny model: brute-force sum over every hidden path is the reference.
rng = np.random.default_rng(0)
N, M, T = 3, 4, 6
pi = rng.dirichlet(np.ones(N))
A = rng.dirichlet(np.ones(N), N)
B = rng.dirichlet(np.ones(M), N)
obs = rng.integers(M, size=T)
total = 0.0
for path in itertools.product(range(N), repeat=T):
    p = pi[path[0]] * B[path[0], obs[0]]
    for t in range(1, T):
        p *= A[path[t - 1], path[t]] * B[path[t], obs[t]]
    total += p
assert np.isclose(ns["forward"](pi, A, B, obs), np.log(total), rtol=1e-12, atol=1e-12)
# Unscaled textbook recursion agrees too
alpha = pi * B[:, obs[0]]
for o in obs[1:]:
    alpha = (alpha @ A) * B[:, o]
assert np.isclose(np.log(alpha.sum()), ns["forward"](pi, A, B, obs))
# Long sequence: scaled version stays finite where the raw product underflows
long_obs = rng.integers(M, size=5000)
ll = ns["forward"](pi, A, B, long_obs)
assert np.isfinite(ll) and ll < 0
a = pi * B[:, long_obs[0]]
for o in long_obs[1:]:
    a = (a @ A) * B[:, o]
assert a.sum() == 0.0  # raw forward underflows
try:
    from hmmlearn.hmm import CategoricalHMM
    h = CategoricalHMM(n_components=N)
    h.startprob_, h.transmat_, h.emissionprob_ = pi, A, B
    h.n_features = M
    assert np.isclose(h.score(obs[:, None]), ns["forward"](pi, A, B, obs))
    print("hmmlearn checked")
except ImportError:
    print("hmmlearn not installed; brute force used")
print("hidden-markov-forward OK")
