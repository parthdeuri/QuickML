import numpy as np, torch, yaml
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import entropy as sp_entropy

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "entropy-kl-divergence.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
p = rng.dirichlet(np.ones(6))
q = rng.dirichlet(np.ones(6))
p0 = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])  # has zero entries

assert np.isclose(ns["entropy"](p), sp_entropy(p))
assert np.isclose(ns["entropy"](p0), sp_entropy(p0))
assert np.isclose(ns["kl"](p, q), sp_entropy(p, q))
assert np.isclose(ns["kl"](p0, q), sp_entropy(p0, q))
tp, tq = torch.tensor(p), torch.tensor(q)
assert np.isclose(ns["kl"](p, q), F.kl_div(tq.log(), tp, reduction="sum").item())
assert np.isclose(ns["cross_entropy"](p, q), -(tp * tq.log()).sum().item())
assert ns["kl"](p, q) >= 0 and np.isclose(ns["kl"](p, p), 0)
assert not np.isclose(ns["kl"](p, q), ns["kl"](q, p))  # asymmetric
print("entropy-kl-divergence OK")
