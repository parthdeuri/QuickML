import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "focal-loss.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
z = rng.normal(size=20) * 3
y = (rng.random(20) < 0.3).astype(float)


def ref_focal(zt, yt, gamma, alpha):
    # direct formula from the paper, built on torch ops
    p = torch.sigmoid(zt)
    pt = p * yt + (1 - p) * (1 - yt)
    at = alpha * yt + (1 - alpha) * (1 - yt)
    ce = F.binary_cross_entropy_with_logits(zt, yt, reduction="none")  # = -log pt
    return (at * (1 - pt) ** gamma * ce).mean()


for gamma, alpha in [(2.0, 0.25), (0.0, 0.5), (1.0, 0.75), (5.0, 0.25)]:
    zt = torch.tensor(z, requires_grad=True)
    loss = ref_focal(zt, torch.tensor(y), gamma, alpha)
    loss.backward()
    assert np.isclose(ns["focal_loss"](z, y, gamma, alpha), loss.item()), (gamma, alpha)
    assert np.allclose(ns["focal_grad"](z, y, gamma, alpha), zt.grad.numpy()), (gamma, alpha)

# gamma = 0, alpha = 0.5 is half of binary cross-entropy
bce = F.binary_cross_entropy_with_logits(torch.tensor(z), torch.tensor(y)).item()
assert np.isclose(ns["focal_loss"](z, y, 0.0, 0.5), 0.5 * bce)
print("focal-loss OK")
