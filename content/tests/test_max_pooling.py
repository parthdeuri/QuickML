import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "max-pooling.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
for H, W, k, s in [(8, 8, 2, 2), (9, 7, 3, 2), (10, 10, 3, 1), (7, 9, 2, 3)]:
    x = rng.normal(size=(H, W))
    out = ns["max_pool2d"](x, k, s)
    ref = F.max_pool2d(torch.tensor(x)[None, None], kernel_size=k, stride=s)
    assert out.shape == tuple(ref.shape[2:]), (H, W, k, s)
    assert np.allclose(out, ref[0, 0].numpy()), (H, W, k, s)
print("max-pooling OK")
