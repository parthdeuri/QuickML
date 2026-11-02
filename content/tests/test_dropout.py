import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

# The card's mask comes from numpy's generator, so it cannot equal torch's
# random mask; we check the properties both implementations must satisfy.
card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "dropout.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
p = 0.3
rng = np.random.default_rng(0)
x = rng.normal(size=(200, 50)) + 3.0

out, mask = ns["dropout"](x, p, rng)
kept = mask != 0
assert np.allclose(out[kept], x[kept] / (1 - p))   # survivors scaled up
assert np.all(out[~kept] == 0)                      # dropped are zero
assert 0.6 < kept.mean() < 0.8                      # roughly 1 - p kept

# inverted scaling: mean over many seeded draws approaches the input
acc = np.zeros_like(x)
trials = 2000
r2 = np.random.default_rng(1)
for _ in range(trials):
    acc += ns["dropout"](x, p, r2)[0]
assert np.abs(acc / trials - x).mean() < 0.1

# eval mode is the identity, like torch's
o_eval, _ = ns["dropout"](x, p, rng, train=False)
assert np.array_equal(o_eval, x)
assert torch.equal(F.dropout(torch.tensor(x), p, training=False), torch.tensor(x))

# torch's own train-mode output obeys the same scaling rule
torch.manual_seed(0)
xt = torch.tensor(x)
yt = F.dropout(xt, p, training=True)
tk = yt != 0
assert torch.allclose(yt[tk], xt[tk] / (1 - p))

# backward with a fixed mask matches autograd of x * mask
d_out = rng.normal(size=x.shape)
xg = torch.tensor(x, requires_grad=True)
(xg * torch.tensor(mask)).backward(torch.tensor(d_out))
assert np.allclose(ns["dropout_backward"](d_out, mask), xg.grad.numpy())
print("dropout OK")
