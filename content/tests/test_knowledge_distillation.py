import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "knowledge-distillation.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
s = rng.normal(size=(8, 5)) * 2
t = rng.normal(size=(8, 5)) * 2
y = rng.integers(0, 5, size=8)
st, tt, yt = torch.tensor(s), torch.tensor(t), torch.tensor(y)
for T, alpha in [(1.0, 0.5), (2.0, 0.5), (4.0, 0.9), (3.0, 0.0)]:
    soft = F.kl_div(F.log_softmax(st / T, dim=1), F.softmax(tt / T, dim=1),
                    reduction="batchmean")
    ref = (1 - alpha) * F.cross_entropy(st, yt) + alpha * T * T * soft
    assert np.isclose(ns["distill_loss"](s, t, y, T=T, alpha=alpha), ref.item()), (T, alpha)
print("knowledge-distillation OK")
