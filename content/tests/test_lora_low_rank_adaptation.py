import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "lora-low-rank-adaptation.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
n, din, dout, r, alpha = 5, 12, 9, 3, 6.0
x = rng.normal(size=(n, din))
W = rng.normal(size=(dout, din))
A = rng.normal(size=(r, din)) * 0.3
B = rng.normal(size=(dout, r)) * 0.3
dy = rng.normal(size=(n, dout))
y = ns["lora_forward"](x, W, A, B, alpha)
# merged weights give the same outputs as the unmerged path
Wm = ns["lora_merge"](W, A, B, alpha)
assert np.allclose(y, x @ Wm.T)
assert np.allclose(Wm, W + (alpha / r) * B @ A)
assert np.linalg.matrix_rank(Wm - W) <= r
# B = 0: identical to the base model
assert np.allclose(ns["lora_forward"](x, W, A, np.zeros_like(B), alpha), x @ W.T)
# gradients against autograd (W frozen)
tx = torch.tensor(x, requires_grad=True)
tA, tB = torch.tensor(A, requires_grad=True), torch.tensor(B, requires_grad=True)
tW = torch.tensor(W)
ty = tx @ tW.T + (alpha / r) * (tx @ tA.T) @ tB.T
assert np.allclose(y, ty.detach().numpy())
(ty * torch.tensor(dy)).sum().backward()
dB, dA, dx = ns["lora_grads"](x, W, A, B, alpha, dy)
assert np.allclose(dB, tB.grad.numpy()) and np.allclose(dA, tA.grad.numpy())
assert np.allclose(dx, tx.grad.numpy())
# with B = 0: A gets no gradient, B still does
dB0, dA0, _ = ns["lora_grads"](x, W, A, np.zeros_like(B), alpha, dy)
assert np.all(dA0 == 0) and np.abs(dB0).max() > 0
# parameter count: r (d_in + d_out) trainable vs d_in * d_out for full fine-tuning
assert A.size + B.size == r * (din + dout)
assert W.size == din * dout
print("lora-low-rank-adaptation OK")
