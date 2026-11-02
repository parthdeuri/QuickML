import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "ppo-clipped-objective.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
ppo = ns["ppo_clip_loss"]

def torch_ref(lp_new, lp_old, adv, eps):
    t = torch.tensor(lp_new, dtype=torch.float64, requires_grad=True)
    a = torch.tensor(adv, dtype=torch.float64)
    ratio = torch.exp(t - torch.tensor(lp_old, dtype=torch.float64))
    obj = torch.minimum(ratio * a, torch.clamp(ratio, 1 - eps, 1 + eps) * a)
    loss = -obj.mean()
    loss.backward()
    return loss.item(), t.grad.numpy()

rng = np.random.default_rng(0)
N = 200
for eps in (0.1, 0.2, 0.3):
    lp_old = -rng.uniform(0.1, 3, N)
    lp_new = lp_old + rng.normal(scale=0.4, size=N)       # ratios on both sides of the band
    adv = rng.normal(size=N)
    loss, grad = ppo(lp_new, lp_old, adv, eps)
    rl, rg = torch_ref(lp_new, lp_old, adv, eps)
    assert abs(loss - rl) < 1e-12 and np.allclose(grad, rg, atol=1e-12)
    ratio = np.exp(lp_new - lp_old)
    assert (ratio > 1 + eps).any() and (ratio < 1 - eps).any()   # all regions exercised

# clipping regions, one sample at a time (N=1 so gradient = dobj/dlogp)
eps = 0.2
def g(ratio, adv):
    return ppo(np.log([ratio]), np.zeros(1), np.array([adv]), eps)

for adv in (1.5, -1.5):
    for ratio, expect_active in ((1.0, True), (1.1, True), (0.9, True)):    # inside the band
        loss, grad = g(ratio, adv)
        assert abs(grad[0] - (-ratio * adv)) < 1e-12 and expect_active
# positive advantage
loss, grad = g(1.5, 1.0)                      # ratio above 1+eps: clipped, no gradient
assert grad[0] == 0 and abs(loss + 1.2) < 1e-12
loss, grad = g(0.5, 1.0)                      # ratio below 1-eps: unclipped, gradient flows
assert abs(grad[0] + 0.5) < 1e-12 and abs(loss + 0.5) < 1e-12
# negative advantage
loss, grad = g(0.5, -1.0)                     # ratio below 1-eps: clipped, no gradient
assert grad[0] == 0 and abs(loss - 0.8) < 1e-12
loss, grad = g(1.5, -1.0)                     # ratio above 1+eps: unclipped, gradient flows
assert abs(grad[0] - 1.5) < 1e-12 and abs(loss - 1.5) < 1e-12
# ratio 1 (new == old): loss = -mean(adv), gradient = -adv/N
adv = rng.normal(size=N)
x = rng.normal(size=N)
loss, grad = ppo(x, x, adv)
assert abs(loss + adv.mean()) < 1e-12 and np.allclose(grad, -adv / N)
print("ppo-clipped-objective OK")
