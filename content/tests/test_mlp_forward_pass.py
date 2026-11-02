import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "mlp-forward-pass.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
sizes = [4, 6, 5, 3]
layers = ns["init_mlp"](sizes, rng)
for (W, b), m, n in zip(layers, sizes[:-1], sizes[1:]):
    assert W.shape == (m, n) and b.shape == (n,)
# random biases so the bias path is exercised too
layers = [(W, rng.normal(size=b.shape)) for W, b in layers]
X = rng.normal(size=(9, 4))

mods = []
for i, (W, b) in enumerate(layers):
    lin = torch.nn.Linear(*W.shape).double()
    lin.weight.data = torch.tensor(W.T.copy())
    lin.bias.data = torch.tensor(b.copy())
    mods.append(lin)
    if i < len(layers) - 1:
        mods.append(torch.nn.ReLU())
net = torch.nn.Sequential(*mods)
logits = net(torch.tensor(X))
out = ns["forward"](layers, X)
assert np.allclose(out, logits.detach().numpy())
assert np.allclose(ns["softmax"](out), torch.softmax(logits, 1).detach().numpy())
print("mlp-forward-pass OK")
