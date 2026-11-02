import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "data-augmentation.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
data = np.random.default_rng(0)
x = data.normal(size=(3, 12, 10))
mean = np.array([0.1, -0.2, 0.3])
std = np.array([0.5, 1.5, 2.0])
size = 6

flips = 0
for seed in range(20):
    mine = ns["augment"](x, np.random.default_rng(seed), size, mean, std)
    # replay the same random draws, apply torch tensor ops
    r = np.random.default_rng(seed)
    t = torch.tensor(x)
    if r.random() < 0.5:
        t = torch.flip(t, dims=[2])
        flips += 1
    top = int(r.integers(0, t.shape[1] - size + 1))
    left = int(r.integers(0, t.shape[2] - size + 1))
    t = t.narrow(1, top, size).narrow(2, left, size)
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    ref = ((t - m) / s).numpy()
    assert mine.shape == (3, size, size) and np.allclose(mine, ref), seed
assert 0 < flips < 20  # both branches exercised
print("data-augmentation OK")
