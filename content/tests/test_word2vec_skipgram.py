import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "word2vec-skipgram.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
assert ns["pairs"]([0, 1, 2], 1) == [(0, 1), (1, 0), (1, 2), (2, 1)]
assert len(ns["pairs"](list(range(5)), 2)) == 14
rng = np.random.default_rng(0)
V, d = 7, 4
W = rng.normal(size=(V, d))
U = rng.normal(size=(V, d))
for c, o in [(0, 3), (5, 5), (2, 6)]:
    loss, gv, gU = ns["sg_loss_grads"](W, U, c, o)
    Wt = torch.tensor(W, requires_grad=True)
    Ut = torch.tensor(U, requires_grad=True)
    l = -torch.log_softmax(Ut @ Wt[c], 0)[o]
    l.backward()
    assert np.allclose(loss, l.item())
    assert np.allclose(gv, Wt.grad[c].numpy())
    assert np.allclose(np.delete(Wt.grad.numpy(), c, 0), 0)
    assert np.allclose(gU, Ut.grad.numpy())
# training on a tiny corpus lowers the loss
ids = [0, 1, 2, 0, 1, 3, 0, 1, 2]
P = ns["pairs"](ids, 1)
def total(W, U):
    return sum(ns["sg_loss_grads"](W, U, c, o)[0] for c, o in P)
before = total(W, U)
for _ in range(50):
    for c, o in P:
        _, gv, gU = ns["sg_loss_grads"](W, U, c, o)
        W[c] -= 0.1 * gv
        U -= 0.1 * gU
assert total(W, U) < before
print("word2vec-skipgram OK")
