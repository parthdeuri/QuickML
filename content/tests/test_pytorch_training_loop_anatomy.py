import yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "pytorch-training-loop-anatomy.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
torch.set_default_dtype(torch.float64)
mse = torch.nn.functional.mse_loss

def make(seed=0, drop=0.0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Dropout(drop), torch.nn.Linear(8, 1))

torch.manual_seed(1)
X, Y = torch.randn(12, 4), torch.randn(12, 1)

# accumulation of k microbatches (loss/k) == one big batch
a, b = make(), make()
oa = torch.optim.SGD(a.parameters(), lr=0.1)
ob = torch.optim.SGD(b.parameters(), lr=0.1)
ns["accumulate_step"](a, [(X[i:i + 4], Y[i:i + 4]) for i in range(0, 12, 4)], mse, oa)
ns["train_step"](b, X, Y, mse, ob)
for pa, pb in zip(a.parameters(), b.parameters()):
    assert torch.allclose(pa, pb, rtol=0, atol=1e-12)

# skipping zero_grad: second gradient is the sum of two gradients
m = make()
g1 = torch.autograd.grad(mse(m(X), Y), list(m.parameters()))
for _ in range(2):
    mse(m(X), Y).backward()
for p, g in zip(m.parameters(), g1):
    assert torch.allclose(p.grad, 2 * g, atol=1e-12)

# eval-mode dropout is deterministic, train mode is not
d = make(drop=0.5)
d.eval()
assert torch.equal(d(X), d(X))
d.train()
assert not torch.equal(d(X), d(X))

# clipping caps the global grad norm (after backward, before step)
c = make()
oc = torch.optim.SGD(c.parameters(), lr=0.1)
ref = make()
mse(ref(X * 50), Y).backward()
raw = torch.sqrt(sum((p.grad ** 2).sum() for p in ref.parameters())).item()
assert raw > 0.5
before = [p.detach().clone() for p in c.parameters()]
ns["train_step"](c, X * 50, Y, mse, oc, clip=0.5)
step_sq = sum(((p - q) ** 2).sum() for p, q in zip(c.parameters(), before))
assert abs(step_sq.sqrt().item() - 0.1 * 0.5) < 1e-6

# evaluate: params unchanged, no graph, equals the mean loss
e = make(drop=0.5)
e.train()
snap = [p.detach().clone() for p in e.parameters()]
loader = [(X[:5], Y[:5]), (X[5:], Y[5:])]
val = ns["evaluate"](e, loader, mse)
assert not e.training
for p, q in zip(e.parameters(), snap):
    assert torch.equal(p, q) and p.grad is None
e.eval()
assert abs(val - mse(e(X), Y).item()) < 1e-12
print("pytorch-training-loop-anatomy OK")
