import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "label-smoothing.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
logits = rng.normal(size=(8, 5)) * 3
y = rng.integers(0, 5, size=8)
for eps in [0.0, 0.1, 0.3]:
    ref = F.cross_entropy(torch.tensor(logits), torch.tensor(y), label_smoothing=eps).item()
    assert np.isclose(ns["smoothed_ce"](logits, y, eps), ref), eps
print("label-smoothing OK")
