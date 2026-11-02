import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "rmsprop-adagrad.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
torch.manual_seed(0)
rng = np.random.default_rng(0)
# badly scaled quadratic  f(w) = 0.5 * sum(c * w^2) + sum(d * w)
c = np.array([100.0, 1.0, 0.01, 5.0])
d = rng.normal(size=4)
w0 = rng.normal(size=4)
grad = lambda w: c * w + d


def torch_run(opt_cls, n, **kw):
    w = torch.tensor(w0, requires_grad=True)
    opt = opt_cls([w], **kw)
    ct, dt = torch.tensor(c), torch.tensor(d)
    for _ in range(n):
        opt.zero_grad()
        (0.5 * (ct * w ** 2).sum() + (dt * w).sum()).backward()
        opt.step()
    return w.detach().numpy()


n = 30
a = ns["minimize"](ns["adagrad_step"], grad, w0, n, lr=0.1)
assert np.allclose(a, torch_run(torch.optim.Adagrad, n, lr=0.1), atol=1e-10)
a2 = ns["minimize"](ns["adagrad_step"], grad, w0, n, lr=0.3, eps=1e-6)
assert np.allclose(a2, torch_run(torch.optim.Adagrad, n, lr=0.3, eps=1e-6), atol=1e-10)

# PyTorch RMSprop defaults: alpha=0.99, eps=1e-8 added OUTSIDE the sqrt
r = ns["minimize"](ns["rmsprop_step"], grad, w0, n, lr=0.01)
assert np.allclose(r, torch_run(torch.optim.RMSprop, n, lr=0.01), atol=1e-10)
r2 = ns["minimize"](ns["rmsprop_step"], grad, w0, n, lr=0.005, alpha=0.9, eps=1e-3)
assert np.allclose(r2, torch_run(torch.optim.RMSprop, n, lr=0.005, alpha=0.9, eps=1e-3), atol=1e-10)

# Adagrad accumulator only grows, so the effective step lr/sqrt(G) only shrinks
w, G = w0.copy(), np.zeros(4)
prev = np.zeros(4)
for _ in range(20):
    w, G = ns["adagrad_step"](w, grad(w), G, lr=0.1)
    assert (G >= prev).all()
    prev = G
print("rmsprop-adagrad OK")
