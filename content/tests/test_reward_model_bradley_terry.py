import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "reward-model-bradley-terry.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
bt_loss = ns["bt_loss"]

rng = np.random.default_rng(0)
N, dim = 40, 6
Xc, Xr = rng.normal(size=(N, dim)), rng.normal(size=(N, dim))
for scale in (1.0, 50.0):                          # large scale: stability
    w = rng.normal(size=dim) * scale
    loss, grad = bt_loss(w, Xc, Xr)
    tw = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    rc, rr = torch.tensor(Xc) @ tw, torch.tensor(Xr) @ tw
    ref = -F.logsigmoid(rc - rr).mean()
    ref.backward()
    assert abs(loss - ref.item()) < 1e-10 * max(1, loss), (loss, ref.item())
    assert np.allclose(grad, tw.grad.numpy(), atol=1e-10, rtol=1e-9)

# Bradley-Terry probability: sigma(rc - rr) = exp(rc) / (exp(rc) + exp(rr))
rc_, rr_ = 0.7, -0.4
assert abs(1 / (1 + np.exp(-(rc_ - rr_))) - np.exp(rc_) / (np.exp(rc_) + np.exp(rr_))) < 1e-15
# equal rewards => loss log 2
assert abs(bt_loss(np.zeros(dim), Xc, Xr)[0] - np.log(2)) < 1e-12
# shift invariance: adding the same vector to both feature sets leaves loss unchanged
shift = rng.normal(size=dim)
w = rng.normal(size=dim)
assert abs(bt_loss(w, Xc + shift, Xr + shift)[0] - bt_loss(w, Xc, Xr)[0]) < 1e-12

# gradient descent on pairs generated from a true Bradley-Terry model improves accuracy
w_true = rng.normal(size=dim)
A, B = rng.normal(size=(2000, dim)), rng.normal(size=(2000, dim))
p_a = 1 / (1 + np.exp(-(A - B) @ w_true))
a_wins = rng.random(2000) < p_a
C = np.where(a_wins[:, None], A, B)
R = np.where(a_wins[:, None], B, A)
w = np.zeros(dim)
l0 = bt_loss(w, C, R)[0]
for _ in range(500):
    w -= 0.5 * bt_loss(w, C, R)[1]
assert bt_loss(w, C, R)[0] < l0
acc = (((C - R) @ w) > 0).mean()
acc0 = (((C - R) @ w_true) > 0).mean()
assert acc > acc0 - 0.03, (acc, acc0)           # near the true model's pair accuracy
print("reward-model-bradley-terry OK")
