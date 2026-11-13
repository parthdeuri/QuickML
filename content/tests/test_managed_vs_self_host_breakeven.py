import math, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "managed-vs-self-host-breakeven.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
api, sh, cpt, be = ns["api_cost"], ns["selfhost_cost"], ns["cost_per_token"], ns["breakeven_tokens"]

# illustrative parameters
p, G, O, cap, u = 1e-5, 2000.0, 1000.0, 8e8, 0.5   # u*cap = 4e8 tokens per GPU
seg = u * cap

# hand-checked values
assert abs(api(3e8, p) - 3000.0) < 1e-6
assert sh(1e8, G, O, cap, u) == 3000.0           # 1 GPU
assert sh(4e8, G, O, cap, u) == 3000.0           # exactly one GPU of capacity
assert sh(5e8, G, O, cap, u) == 5000.0           # 2 GPUs
assert abs(cpt(sh(2e8, G, O, cap, u), 2e8) - 1.5e-5) < 1e-18

# cost jumps exactly at multiples of u*cap
for k in (1, 2, 3):
    assert sh(k * seg, G, O, cap, u) == k * G + O
    assert sh(k * seg * (1 + 1e-9), G, O, cap, u) == (k + 1) * G + O

# break-even equals a brute-force scan on a fine grid
t_max, step = 2e9, 1e6
t_star = be(p, G, O, cap, u, t_max)
assert abs(t_star - 3e8) < 1e-3
grid = [i * step for i in range(1, int(t_max / step) + 1)]
brute = next(t for t in grid if sh(t, G, O, cap, u) <= api(t, p) + 1e-9)
assert t_star <= brute <= t_star + step, (t_star, brute)
# several other parameter sets
for p2, G2, O2, u2 in [(2e-5, 2000, 1000, 0.5), (1e-5, 1000, 500, 0.25),
                       (1e-5, 3000, 200, 0.8), (4e-6, 2000, 1000, 0.5)]:
    ts = be(p2, G2, O2, cap, u2, t_max)
    br = next((t for t in grid if sh(t, G2, O2, cap, u2) <= api(t, p2) + 1e-9), None)
    if ts is None:
        assert br is None, (p2, G2, O2, u2, br)
    else:
        assert br is not None and ts - 1e-6 <= br <= ts + step, (ts, br)

# below break-even the API is cheaper; above it self-hosting is cheaper
assert api(2.5e8, p) < sh(2.5e8, G, O, cap, u)
assert sh(3.5e8, G, O, cap, u) < api(3.5e8, p)
# next step up: the API is cheaper again, then self-hosting wins past 5e8
assert api(4.5e8, p) < sh(4.5e8, G, O, cap, u)
assert sh(6e8, G, O, cap, u) < api(6e8, p)
# self-hosting never wins when G/(u*cap) is above p
assert be(1e-5, 2000.0, 1000.0, 2e8, 0.5, 1e10) is None

# higher utilisation lowers (never raises) the cost per token
for t in (1e8, 3e8, 7e8, 1.3e9):
    cs = [cpt(sh(t, G, O, cap, uu), t) for uu in (0.2, 0.3, 0.5, 0.7, 0.9)]
    assert all(a >= b for a, b in zip(cs, cs[1:])), cs
assert cpt(sh(3e8, G, O, cap, 0.25), 3e8) > cpt(sh(3e8, G, O, cap, 0.5), 3e8)
print("managed-vs-self-host-breakeven OK")
