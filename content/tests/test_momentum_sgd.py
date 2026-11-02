import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "momentum-sgd.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
A = rng.normal(size=(6, 4))
y = rng.normal(size=6)
w0 = rng.normal(size=4)

for nesterov in [False, True]:
    wt = torch.tensor(w0.copy(), requires_grad=True)
    opt = torch.optim.SGD([wt], lr=0.05, momentum=0.9, nesterov=nesterov)
    w, buf = w0.copy(), np.zeros(4)
    for _ in range(25):
        opt.zero_grad()
        loss = ((torch.tensor(A) @ wt - torch.tensor(y)) ** 2).mean()
        loss.backward()
        opt.step()
        g = 2 * A.T @ (A @ w - y) / len(y)
        w, buf = ns["sgd_step"](w, g, buf, lr=0.05, mu=0.9, nesterov=nesterov)
        assert np.allclose(w, wt.detach().numpy()), nesterov
print("momentum-sgd OK")
