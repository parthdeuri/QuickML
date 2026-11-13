import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "adamw-decoupled-weight-decay.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
w0 = rng.normal(size=6)
grads = [rng.normal(size=6) * (10.0 ** rng.integers(-2, 2, size=6)) for _ in range(20)]
kw = dict(lr=1e-2, b1=0.9, b2=0.99, eps=1e-8, wd=0.1)

def run_torch(cls):
    p = torch.tensor(w0.copy(), dtype=torch.float64, requires_grad=True)
    opt = cls([p], lr=kw["lr"], betas=(kw["b1"], kw["b2"]), eps=kw["eps"], weight_decay=kw["wd"])
    for g in grads:
        p.grad = torch.tensor(g, dtype=torch.float64)
        opt.step()
    return p.detach().numpy()

def run_card(step):
    w, st = w0.copy(), ns["init_state"](w0)
    for g in grads:
        w = step(st, w, g, **kw)
    return w

w_aw, w_l2 = run_card(ns["adamw_step"]), run_card(ns["adam_l2_step"])
assert np.allclose(w_aw, run_torch(torch.optim.AdamW), rtol=0, atol=1e-12)
assert np.allclose(w_l2, run_torch(torch.optim.Adam), rtol=0, atol=1e-12)
assert np.abs(w_aw - w_l2).max() > 1e-3

# zero gradient, fresh state: AdamW shrinks by exactly lr*wd*w
w = np.array([1.0, -2.0, 0.5])
out = ns["adamw_step"](ns["init_state"](w), w, np.zeros(3), **kw)
assert np.allclose(out, w * (1 - kw["lr"] * kw["wd"]), atol=1e-15)
# Adam+L2: shrink depends on gradient history (large history -> smaller decay)
def l2_shrink(hist):
    st, ww = ns["init_state"](w), w.copy()
    for _ in range(5):
        ww = ns["adam_l2_step"](st, ww, np.full(3, hist), **kw)
    before = ww.copy()
    ww = ns["adam_l2_step"](st, ww, np.zeros(3), **kw)
    return np.abs(ww - before)
small, large = l2_shrink(0.01), l2_shrink(100.0)
assert not np.allclose(small, large)
# AdamW zero-gradient decay at later steps is lr*wd*w plus only the momentum tail
st, ww = ns["init_state"](w), w.copy()
for _ in range(5):
    ww = ns["adamw_step"](st, ww, np.full(3, 100.0), **kw)
decay_only = kw["lr"] * kw["wd"] * ww
st2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in st.items()}
nxt = ns["adamw_step"](st, ww, np.zeros(3), **kw)
assert np.allclose(ww - nxt - decay_only, kw["lr"] * (st["m"] / (1 - kw["b1"] ** st["t"])) / (np.sqrt(st["v"] / (1 - kw["b2"] ** st["t"])) + kw["eps"]), atol=1e-12)
print("adamw-decoupled-weight-decay OK")
