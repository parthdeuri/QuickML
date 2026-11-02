import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "learning-rate-schedules.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
base, epochs = 0.1, 30


def torch_lrs(make):
    p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    opt = torch.optim.SGD([p], lr=base)
    sched = make(opt)
    lrs = []
    for _ in range(epochs):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return np.array(lrs)


S = torch.optim.lr_scheduler
ref = torch_lrs(lambda o: S.StepLR(o, step_size=7, gamma=0.5))
mine = np.array([ns["step_lr"](base, e, 7, 0.5) for e in range(epochs)])
assert np.allclose(ref, mine)

ref = torch_lrs(lambda o: S.CosineAnnealingLR(o, T_max=epochs, eta_min=0.001))
mine = np.array([ns["cosine_lr"](base, e, epochs, 0.001) for e in range(epochs)])
assert np.allclose(ref, mine)

# warmup variant has no torch scheduler: check it against LambdaLR and its shape
W, total = 5, 30
f = lambda t: ns["warmup_cosine"](1.0, t, W, total)
ref = torch_lrs(lambda o: S.LambdaLR(o, f))
mine = np.array([base * f(t) for t in range(epochs)])
assert np.allclose(ref, mine)
assert np.all(np.diff(mine[:W]) > 0)            # ramps up
assert np.isclose(mine[W - 1], base) and np.isclose(mine[W], base)
assert np.all(np.diff(mine[W:]) < 0)            # then decays
print("learning-rate-schedules OK")
