import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "residual-connections.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
torch.manual_seed(0)
rng = np.random.default_rng(0)
B, D, Hd = 5, 6, 9
l1 = torch.nn.Linear(D, Hd).double()
l2 = torch.nn.Linear(Hd, D).double()
W1, b1 = l1.weight.detach().numpy().T, l1.bias.detach().numpy()
W2, b2 = l2.weight.detach().numpy().T, l2.bias.detach().numpy()
x = rng.normal(size=(B, D))
out = ns["residual_block"](x, W1, b1, W2, b2)
ref = torch.tensor(x) + l2(torch.relu(l1(torch.tensor(x))))
assert np.allclose(out, ref.detach().numpy())

# zero F gives exact identity
z = ns["residual_block"](x, W1, b1, np.zeros_like(W2), np.zeros_like(b2))
assert np.allclose(z, x)

# Jacobian is I + dF/dx (checked with autograd on one example)
f = lambda v: v + l2(torch.relu(l1(v)))
fF = lambda v: l2(torch.relu(l1(v)))
v = torch.tensor(x[0])
J = torch.autograd.functional.jacobian(f, v)
JF = torch.autograd.functional.jacobian(fF, v)
assert torch.allclose(J, torch.eye(D, dtype=torch.float64) + JF)
print("residual-connections OK")
