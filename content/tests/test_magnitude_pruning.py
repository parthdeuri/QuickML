import numpy as np, yaml, torch
import torch.nn.utils.prune as prune
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "magnitude-pruning.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
W = rng.normal(size=(20, 30))
for s in (0.0, 0.3, 0.5, 0.9):
    M = ns["magnitude_mask"](W, s)
    assert abs((~M).mean() - s) <= 1 / W.size  # exact up to rounding
    P = W * M
    assert (P[~M] == 0).all() and (P[M] == W[M]).all()
    # every kept weight is at least as large as every pruned one
    if (~M).any():
        assert np.abs(W[M]).min() >= np.abs(W[~M]).max()
    if 0 < s:
        t = torch.nn.Linear(30, 20, bias=False)
        t.weight.data = torch.tensor(W)
        prune.l1_unstructured(t, "weight", amount=s)
        assert (t.weight_mask.numpy().astype(bool) == M).all()

# ties: sparsity still hit exactly by index order
Z = np.zeros((4, 4))
assert (~ns["magnitude_mask"](Z, 0.5)).sum() == 8

# global vs layer-wise differ when scales differ
A, B = rng.normal(size=(30, 30)) * 0.1, rng.normal(size=(30, 30)) * 1.0
g = ns["global_masks"]([A, B], 0.5)
tot_pruned = sum((~m).sum() for m in g)
assert tot_pruned == round(0.5 * (A.size + B.size))
assert (~g[0]).mean() > 0.8 and (~g[1]).mean() < 0.2
l = [ns["magnitude_mask"](X, 0.5) for X in (A, B)]
assert abs((~l[0]).mean() - 0.5) < 0.01 and abs((~l[1]).mean() - 0.5) < 0.01
assert abs((~g[0]).mean() - (~l[0]).mean()) > 0.3

# channel pruning keeps highest-norm rows
C = rng.normal(size=(10, 6)) * np.arange(1, 11)[:, None]
for p in (1, 2):
    Cp, keep = ns["prune_channels"](C, 0.3, p)
    assert Cp.shape == (3, 6) and (Cp == C[keep]).all()
    norms = np.linalg.norm(C, ord=p, axis=1)
    assert set(keep) == set(np.argsort(-norms)[:3])
    assert (np.diff(keep) > 0).all()
print("magnitude-pruning OK")
