import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "depthwise-separable-convolution.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
x = rng.normal(size=(2, 3, 9, 8))
dw = rng.normal(size=(3, 3, 2))
pw = rng.normal(size=(5, 3))
for stride, pad in [(1, 0), (2, 1), (1, 2), (3, 0)]:
    mine = ns["depthwise_separable"](x, dw, pw, stride=stride, pad=pad)
    xt = torch.tensor(x)
    mid = F.conv2d(xt, torch.tensor(dw)[:, None], stride=stride, padding=pad, groups=3)
    ref = F.conv2d(mid, torch.tensor(pw)[:, :, None, None]).numpy()
    assert mine.shape == ref.shape and np.allclose(mine, ref), (stride, pad)

# module form with nn.Conv2d(groups=) composition
d = torch.nn.Conv2d(3, 3, (3, 2), groups=3, bias=False).double()
p = torch.nn.Conv2d(3, 5, 1, bias=False).double()
d.weight.data = torch.tensor(dw)[:, None]
p.weight.data = torch.tensor(pw)[:, :, None, None]
assert np.allclose(ns["depthwise_separable"](x, dw, pw), p(d(torch.tensor(x))).detach().numpy())
print("depthwise-separable-convolution OK")
