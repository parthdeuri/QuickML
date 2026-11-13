import numpy as np, yaml, torch
import torch.nn as nn
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "adapter-layers.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
af, pc, lf = ns["adapter_forward"], ns["param_count"], ns["lora_forward"]

rng = np.random.default_rng(0)
d, r, n = 16, 4, 5
x = rng.normal(size=(n, d))
Wd, bd = rng.normal(size=(d, r)) / 4, rng.normal(size=r)
Wu, bu = rng.normal(size=(r, d)) / 4, rng.normal(size=d)
# near-identity init: W_up = 0, b_up = 0 gives output == input
assert np.array_equal(af(x, Wd, bd, np.zeros((r, d)), np.zeros(d)), x)

# parameter count matches an equivalent torch module
seq = nn.Sequential(nn.Linear(d, r), nn.ReLU(), nn.Linear(r, d))
assert sum(p.numel() for p in seq.parameters()) == pc(d, r) == 2 * d * r + r + d

# output equals torch reference (x + Sequential(x))
with torch.no_grad():
    seq[0].weight.copy_(torch.tensor(Wd.T)); seq[0].bias.copy_(torch.tensor(bd))
    seq[2].weight.copy_(torch.tensor(Wu.T)); seq[2].bias.copy_(torch.tensor(bu))
    xt = torch.tensor(x)
    ref = (xt + seq.double()(xt)).numpy()
assert np.allclose(af(x, Wd, bd, Wu, bu), ref, atol=1e-12)

# frozen base gets no gradient; adapter weights do
base = nn.Linear(d, d).double()
for p in base.parameters():
    p.requires_grad_(False)
seq.double()
out = xt + seq(base(xt))
out.sum().backward()
assert base.weight.grad is None and base.bias.grad is None
assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in seq.parameters())

# LoRA forward: merging gives identical outputs (no extra layer at inference)
W = rng.normal(size=(d, d)); A = rng.normal(size=(r, d)); B = rng.normal(size=(d, r))
Wm = W + (8 / r) * B @ A
assert np.allclose(lf(x, W, A, B, 8), x @ Wm.T)
print("adapter-layers OK")
