import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "backpropagation.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
X = rng.normal(size=(7, 4))
y = rng.normal(size=(7, 2))
p = {"W1": rng.normal(size=(4, 5)), "b1": rng.normal(size=5),
     "W2": rng.normal(size=(5, 2)), "b2": rng.normal(size=2)}
loss, g = ns["loss_and_grads"](p, X, y)
t = {k: torch.tensor(v, requires_grad=True) for k, v in p.items()}
h = torch.tanh(torch.tensor(X) @ t["W1"] + t["b1"])
tl = torch.nn.functional.mse_loss(h @ t["W2"] + t["b2"], torch.tensor(y))
tl.backward()
assert np.allclose(loss, tl.item())
for k in p:
    assert np.allclose(g[k], t[k].grad.numpy()), k
print("backpropagation OK")
