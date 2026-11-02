import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "rotary-position-embedding.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
T, d = 9, 8
x = rng.normal(size=(T, d))
out = ns["rope"](x)
# complex-rotation definition via torch
theta = 10000.0 ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
ang = torch.arange(T, dtype=torch.float64)[:, None] * theta[None]
z = torch.view_as_complex(torch.tensor(x).reshape(T, d // 2, 2).contiguous())
ref = torch.view_as_real(z * torch.polar(torch.ones_like(ang), ang)).reshape(T, d)
assert np.allclose(out, ref.numpy())
assert np.allclose(np.linalg.norm(out, axis=1), np.linalg.norm(x, axis=1))
assert np.allclose(out[0], x[0])  # position 0 is the identity
# relative-position property: same q, k content at every position
q0, k0 = rng.normal(size=d), rng.normal(size=d)
S = ns["rope"](np.tile(q0, (T, 1))) @ ns["rope"](np.tile(k0, (T, 1))).T
for m in range(T):
    for n in range(T):
        for s in range(T - max(m, n)):
            assert np.isclose(S[m + s, n + s], S[m, n])
assert not np.isclose(S[3, 1], S[3, 2])  # but it does depend on m - n
print("rotary-position-embedding OK")
