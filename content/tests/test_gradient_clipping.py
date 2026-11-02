import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "gradient-clipping.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
shapes = [(3, 4), (5,), (2, 2, 2)]

for scale, max_norm in [(5.0, 1.0), (0.01, 1.0)]:  # clipped and untouched cases
    grads = [rng.normal(size=s) * scale for s in shapes]
    params = [torch.zeros(s, dtype=torch.float64, requires_grad=True) for s in shapes]
    for p, g in zip(params, grads):
        p.grad = torch.tensor(g.copy())
    tn = torch.nn.utils.clip_grad_norm_(params, max_norm)
    mine, total = ns["clip_by_global_norm"](grads, max_norm)
    assert np.isclose(total, tn.item())
    for m, p in zip(mine, params):
        assert np.allclose(m, p.grad.numpy())

grads = [rng.normal(size=s) * 3 for s in shapes]
params = [torch.zeros(s, dtype=torch.float64, requires_grad=True) for s in shapes]
for p, g in zip(params, grads):
    p.grad = torch.tensor(g.copy())
torch.nn.utils.clip_grad_value_(params, 0.7)
mine = ns["clip_by_value"](grads, 0.7)
for m, p in zip(mine, params):
    assert np.allclose(m, p.grad.numpy())
print("gradient-clipping OK")
