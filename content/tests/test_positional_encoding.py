import math
import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "positional-encoding.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
T, d = 12, 16
pe = ns["sinusoidal_pe"](T, d)

# original formula, one entry at a time, in torch float64
ref = torch.zeros(T, d, dtype=torch.float64)
for pos in range(T):
    for j in range(d // 2):
        a = torch.tensor(pos / 10000 ** (2 * j / d), dtype=torch.float64)
        ref[pos, 2 * j] = torch.sin(a)
        ref[pos, 2 * j + 1] = torch.cos(a)
assert pe.shape == (T, d)
assert np.allclose(pe, ref.numpy())

# shifting by k is a rotation of each (sin, cos) pair
k = 3
for j in range(d // 2):
    w = 1 / 10000 ** (2 * j / d)
    R = np.array([[math.cos(w * k), math.sin(w * k)],
                  [-math.sin(w * k), math.cos(w * k)]])
    for p in range(T - k):
        assert np.allclose(R @ pe[p, 2 * j:2 * j + 2], pe[p + k, 2 * j:2 * j + 2])
assert len({tuple(np.round(r, 9)) for r in pe}) == T
print("positional-encoding OK")
