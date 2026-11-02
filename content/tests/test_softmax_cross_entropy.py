import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "softmax-cross-entropy.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
z = rng.normal(size=(8, 5)) * 3
z[0, 0] = 1000.0  # would overflow a naive exp
y = rng.integers(0, 5, 8)
zt = torch.tensor(z, requires_grad=True)
loss = F.cross_entropy(zt, torch.tensor(y))
loss.backward()
assert np.allclose(ns["softmax"](z), torch.softmax(zt, 1).detach().numpy())
assert np.allclose(ns["cross_entropy"](z, y), loss.item())
assert np.allclose(ns["grad_logits"](z, y), zt.grad.numpy())
print("softmax-cross-entropy OK")
