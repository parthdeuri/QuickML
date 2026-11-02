import numpy as np, yaml, torch
import torch.nn as nn
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "resnet-basic-block.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
torch.manual_seed(0)
rng = np.random.default_rng(0)


class RefBlock(nn.Module):
    """BasicBlock composed from nn.Conv2d / BatchNorm2d / ReLU."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = self.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return self.relu(out + (x if self.short is None else self.short(x)))


def run(cin, cout, stride):
    x = rng.normal(size=(2, cin, 6, 6))
    p = {"w1": rng.normal(size=(cout, cin, 3, 3)) * 0.3, "w2": rng.normal(size=(cout, cout, 3, 3)) * 0.3}
    for k in ("1", "2"):
        p["g" + k] = rng.uniform(0.5, 1.5, cout)
        p["b" + k] = rng.normal(size=cout)
    ref = RefBlock(cin, cout, stride).double().train()
    with torch.no_grad():
        ref.c1.weight.copy_(torch.tensor(p["w1"]))
        ref.c2.weight.copy_(torch.tensor(p["w2"]))
        for bn_, k in ((ref.b1, "1"), (ref.b2, "2")):
            bn_.weight.copy_(torch.tensor(p["g" + k]))
            bn_.bias.copy_(torch.tensor(p["b" + k]))
        if ref.short is not None:
            p["ws"] = rng.normal(size=(cout, cin, 1, 1)) * 0.3
            p["gs"] = rng.uniform(0.5, 1.5, cout)
            p["bs"] = rng.normal(size=cout)
            ref.short[0].weight.copy_(torch.tensor(p["ws"]))
            ref.short[1].weight.copy_(torch.tensor(p["gs"]))
            ref.short[1].bias.copy_(torch.tensor(p["bs"]))
    out = ns["basic_block"](x, p, stride)
    r = ref(torch.tensor(x)).detach().numpy()
    assert out.shape == r.shape, (out.shape, r.shape)
    assert np.allclose(out, r, atol=1e-8), np.abs(out - r).max()
    return out


a = run(4, 4, 1)      # identity shortcut
b = run(4, 8, 2)      # projection shortcut, stride 2
assert a.shape == (2, 4, 6, 6) and b.shape == (2, 8, 3, 3)
assert (a >= 0).all() and (b >= 0).all()
print("resnet-basic-block OK")
