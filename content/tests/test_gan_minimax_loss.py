import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "gan-minimax-loss.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
n = 16
lr, lf = rng.normal(size=n) * 2, rng.normal(size=n) * 2
d_loss, g_sat, g_ns, gs, gn = ns["gan_losses"](lr, lf)

bce = F.binary_cross_entropy_with_logits
r, f = torch.tensor(lr, requires_grad=True), torch.tensor(lf, requires_grad=True)
ones, zeros = torch.ones(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64)
assert np.isclose(d_loss, (bce(r, ones) + bce(f, zeros)).item())
sat = -bce(f, zeros)  # log(1 - D(G(z))), G minimises
non_sat = bce(f, ones)  # -log D(G(z))
assert np.isclose(g_sat, sat.item()) and np.isclose(g_ns, non_sat.item())
sat.backward(retain_graph=True)
assert np.allclose(gs, f.grad.numpy())
f.grad = None
non_sat.backward()
assert np.allclose(gn, f.grad.numpy())
# direct log form matches too
D = lambda x: 1 / (1 + np.exp(-x))
assert np.isclose(d_loss, -np.mean(np.log(D(lr))) - np.mean(np.log(1 - D(lf))))
assert np.isclose(g_sat, np.mean(np.log(1 - D(lf))))
# confident discriminator: saturating gradient vanishes, non-saturating does not
conf = np.full(n, -12.0)
_, _, _, gs2, gn2 = ns["gan_losses"](lr, conf)
assert np.abs(gs2).max() < 1e-4 * np.abs(gn2).min()
# at D = 1/2 (logit 0) the D loss equals 2 log 2
assert np.isclose(ns["gan_losses"](np.zeros(n), np.zeros(n))[0], 2 * np.log(2))
# extreme logits stay finite in the losses
assert np.isfinite(ns["gan_losses"](np.array([-800.0]), np.array([800.0]))[0])
print("gan-minimax-loss OK")
