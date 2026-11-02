import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "layer-normalization.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
torch.manual_seed(0)
rng = np.random.default_rng(0)
B, T, D = 4, 5, 8
x = rng.normal(2.0, 3.0, size=(B, T, D))
gamma, beta = rng.normal(size=D), rng.normal(size=D)
ln = torch.nn.LayerNorm(D, eps=1e-5).double()
with torch.no_grad():
    ln.weight.copy_(torch.tensor(gamma))
    ln.bias.copy_(torch.tensor(beta))
out = ns["layer_norm"](x, gamma, beta)
assert np.allclose(out, ln(torch.tensor(x)).detach().numpy())
y = ns["layer_norm"](x, np.ones(D), np.zeros(D))
assert np.allclose(y.mean(axis=-1), 0, atol=1e-8)
assert np.allclose(y.var(axis=-1), 1, atol=1e-3)
print("layer-normalization OK")
