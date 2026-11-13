import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "glove-embeddings.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
n, d = 8, 4
X = rng.integers(1, 300, size=(n, n)).astype(float)  # positive co-occurrence counts
P = [rng.normal(scale=0.1, size=(n, d)), rng.normal(scale=0.1, size=(n, d)),
     np.zeros(n), np.zeros(n)]

# weighting function: cap at 1 above xmax
w = ns["weight"](np.array([1.0, 10.0, 100.0, 500.0]))
assert np.allclose(w, [0.01 ** 0.75, 0.1 ** 0.75, 1.0, 1.0])

# loss and gradients match torch autograd
T = [torch.tensor(p, requires_grad=True) for p in P]
Xt = torch.tensor(X)
fw = torch.clamp((Xt / 100.0) ** 0.75, max=1.0)
R = T[0] @ T[1].T + T[2][:, None] + T[3][None, :] - torch.log(Xt)
L = (fw * R ** 2).sum()
L.backward()
assert np.isclose(L.item(), ns["loss"](P, X))
for g, t in zip(ns["grads"](P, X), T):
    assert np.allclose(g, t.grad.numpy(), atol=1e-9)

# gradient descent decreases the loss over 200 steps
Q, losses = P, [ns["loss"](P, X)]
for _ in range(200):
    Q = ns["step"](Q, X, 0.001)
    losses.append(ns["loss"](Q, X))
assert all(a >= b for a, b in zip(losses, losses[1:]))
assert losses[-1] < 0.1 * losses[0], (losses[0], losses[-1])
print("glove-embeddings OK")
