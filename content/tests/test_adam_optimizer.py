import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "adam-optimizer.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
A = rng.normal(size=(6, 4))
y = rng.normal(size=6)
w0 = rng.normal(size=4)
lr = 0.05

wt = torch.tensor(w0.copy(), requires_grad=True)
opt = torch.optim.Adam([wt], lr=lr)
w, st = w0.copy(), ns["init_state"](w0)
for _ in range(30):
    opt.zero_grad()
    loss = ((torch.tensor(A) @ wt - torch.tensor(y)) ** 2).mean()
    loss.backward()
    opt.step()
    g = 2 * A.T @ (A @ w - y) / len(y)
    w = ns["adam_step"](w, g, st, lr=lr)
    assert np.allclose(w, wt.detach().numpy())
print("adam-optimizer OK")
