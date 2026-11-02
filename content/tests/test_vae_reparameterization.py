import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "vae-reparameterization.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
n, d, k = 6, 5, 3
x = rng.normal(size=(n, d))
shapes = [(d, k), (k,), (d, k), (k,), (k, d), (d,)]
P = [rng.normal(size=s) * 0.5 for s in shapes]
eps = rng.normal(size=(n, k))
loss, g = ns["vae_loss_grads"](x, P, eps)

ts = [torch.tensor(p, requires_grad=True) for p in P]
Wm, bm, Ws, bs, Wd, bd = ts
xt, et = torch.tensor(x), torch.tensor(eps)
mu, lv = xt @ Wm + bm, xt @ Ws + bs
z = mu + torch.exp(0.5 * lv) * et          # gradient flows through mu and sigma
xh = z @ Wd + bd
rec = 0.5 * ((xh - xt) ** 2).sum()
q = torch.distributions.Normal(mu, torch.exp(0.5 * lv))
kl = torch.distributions.kl_divergence(q, torch.distributions.Normal(0.0, 1.0)).sum()
ref = (rec + kl) / n
ref.backward()
assert np.allclose(loss, ref.item())
for mine, t in zip(g, ts):
    assert np.allclose(mine, t.grad.numpy()), (mine, t.grad.numpy())
# the encoder heads really receive gradient through the sample
assert np.abs(g[2]).max() > 0 and np.abs(g[0]).max() > 0
# z = mu + sigma * eps has mean mu and std sigma
m, s = 1.5, 0.4
zs = m + s * np.random.default_rng(1).normal(size=200000)
assert abs(zs.mean() - m) < 0.01 and abs(zs.std() - s) < 0.01
print("vae-reparameterization OK")
