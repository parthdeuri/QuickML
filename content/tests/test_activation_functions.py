import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "activation-functions.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
x = np.random.default_rng(0).normal(size=50) * 3
x[x == 0] = 1.0
xt = torch.tensor(x, requires_grad=True)


def grad_of(f):
    xt.grad = None
    f(xt).sum().backward()
    return xt.grad.numpy().copy()


assert np.allclose(ns["relu"](x), F.relu(xt).detach().numpy())
assert np.allclose(ns["sigmoid"](x), torch.sigmoid(xt).detach().numpy())
assert np.allclose(ns["gelu"](x), F.gelu(xt, approximate="tanh").detach().numpy())
assert np.allclose(ns["d_relu"](x), grad_of(F.relu))
assert np.allclose(ns["d_sigmoid"](x), grad_of(torch.sigmoid))
assert np.allclose(ns["d_tanh"](x), grad_of(torch.tanh))
print("activation-functions OK")
