import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "transposed-convolution.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
x = rng.normal(size=(2, 3, 5, 4))
w = rng.normal(size=(3, 4, 3, 2))
for stride, pad in [(1, 0), (2, 0), (2, 1), (3, 1)]:
    mine = ns["conv_transpose2d"](x, w, stride=stride, pad=pad)
    ref = F.conv_transpose2d(torch.tensor(x), torch.tensor(w),
                             stride=stride, padding=pad).numpy()
    assert mine.shape == ref.shape and np.allclose(mine, ref), (stride, pad)
print("transposed-convolution OK")
