import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "dpo-loss.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
dpo_loss = ns["dpo_loss"]

rng = np.random.default_rng(0)
N = 12
for scale in (1.0, 30.0, 500.0):                    # large scales test numerical stability
    pi_w, pi_l = -rng.uniform(5, 50, N) * scale / 10, -rng.uniform(5, 50, N) * scale / 10
    ref_w, ref_l = -rng.uniform(5, 50, N) * scale / 10, -rng.uniform(5, 50, N) * scale / 10
    for beta in (0.1, 0.5, 2.0):
        loss, gw, gl = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta)
        t = [torch.tensor(a, dtype=torch.float64, requires_grad=True)
             for a in (pi_w, pi_l, ref_w, ref_l)]
        # formula in terms of log-ratios, built from torch.logsigmoid
        margin = beta * ((t[0] - t[2]) - (t[1] - t[3]))
        ref = -torch.nn.functional.logsigmoid(margin).mean()
        ref.backward()
        assert abs(loss - ref.item()) < 1e-9 * max(1, abs(loss)), (scale, beta, loss, ref.item())
        assert np.allclose(gw, t[0].grad.numpy(), atol=1e-12, rtol=1e-9)
        assert np.allclose(gl, t[1].grad.numpy(), atol=1e-12, rtol=1e-9)
        assert np.allclose(t[2].grad.numpy(), -gw) and np.allclose(t[3].grad.numpy(), -gl)
        assert np.isfinite(loss)

# policy == reference: margin 0, loss = log 2, gradient = -/+ beta/(2N)
x = rng.normal(size=N)
y = rng.normal(size=N)
loss, gw, gl = dpo_loss(x, y, x, y, beta=0.3)
assert abs(loss - np.log(2)) < 1e-12
assert np.allclose(gw, -0.3 / (2 * N)) and np.allclose(gl, 0.3 / (2 * N))
# raising the chosen / lowering the rejected log-prob lowers the loss
l2, _, _ = dpo_loss(x + 1, y - 1, x, y, beta=0.3)
assert l2 < loss
# equals the sigmoid-of-log-ratios form: -log sigma(beta*(log ratio_w - log ratio_l))
beta = 0.3
pw, pl, rw, rl = (rng.normal(size=N) for _ in range(4))
direct = -np.log(1 / (1 + np.exp(-beta * (np.log(np.exp(pw) / np.exp(rw)) - np.log(np.exp(pl) / np.exp(rl)))))).mean()
assert abs(dpo_loss(pw, pl, rw, rl, beta)[0] - direct) < 1e-12
print("dpo-loss OK")
