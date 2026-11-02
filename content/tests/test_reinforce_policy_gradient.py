import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "reinforce-policy-gradient.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

assert np.allclose(ns["returns"]([1.0, 0.0, 2.0], 0.5), [1.0 + 0.5 * (0 + 0.5 * 2), 1.0, 2.0])

rng = np.random.default_rng(0)
T, d, A = 9, 5, 3
X = rng.normal(size=(T, d))
W = rng.normal(size=(d, A))
a = rng.integers(0, A, T)
r = rng.normal(size=T)
G = ns["returns"](r, 0.95)
for b in (0.0, G.mean()):
    g = ns["reinforce_grad"](W, X, a, G, b)
    Wt = torch.tensor(W, requires_grad=True)
    logp = torch.log_softmax(torch.tensor(X) @ Wt, 1)[torch.arange(T), torch.tensor(a)]
    surrogate = -((torch.tensor(G - b)) * logp).sum()   # minimise = ascend J
    surrogate.backward()
    assert np.allclose(g, -Wt.grad.numpy())

# Expected-gradient check: on a one-step bandit the sample average of
# the estimator approaches the exact gradient of expected reward.
rew = np.array([1.0, 0.0, 2.0])
th = rng.normal(size=(1, 3))
p = np.exp(th) / np.exp(th).sum()
exact = p[0] * (rew - p[0] @ rew)         # d E[r] / d theta
acts = rng.choice(3, size=200000, p=p[0])
est = ns["reinforce_grad"](th, np.ones((len(acts), 1)), acts, rew[acts], 0.0)[0] / len(acts)
assert np.allclose(est, exact, atol=0.01), (est, exact)
print("reinforce-policy-gradient OK")
