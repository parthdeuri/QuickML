import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "convolution-2d.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
x = rng.normal(size=(2, 3, 9, 8))
w = rng.normal(size=(4, 3, 3, 2))
b = rng.normal(size=4)
for stride, pad in [(1, 0), (2, 1), (1, 2), (3, 0)]:
    mine = ns["conv2d"](x, w, b, stride=stride, pad=pad)
    ref = F.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b),
                   stride=stride, padding=pad).numpy()
    assert mine.shape == ref.shape and np.allclose(mine, ref), (stride, pad)
print("convolution-2d OK")
