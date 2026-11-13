import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "sliding-window-attention.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
band_mask, window_attn, ring_update = ns["band_mask"], ns["window_attn"], ns["ring_update"]

rng = np.random.default_rng(0)
T, d = 12, 6
q, k, v = (rng.normal(size=(T, d)) for _ in range(3))
tq, tk, tv = (torch.tensor(m)[None] for m in (q, k, v))

# band mask properties
M = band_mask(5, 2)
assert M.tolist() == [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 1, 1, 0, 0],
                      [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]]
assert (band_mask(T, 3).sum(1) == [1, 2] + [3] * (T - 2)).all()

# equals torch full attention with the explicit band mask
for W in (1, 3, 5):
    ref = F.scaled_dot_product_attention(
        tq, tk, tv, attn_mask=torch.tensor(band_mask(T, W)))[0].numpy()
    assert np.allclose(window_attn(q, k, v, W), ref, atol=1e-12), W

# W >= T equals full causal attention
ref = F.scaled_dot_product_attention(tq, tk, tv, is_causal=True)[0].numpy()
for W in (T, T + 5):
    assert np.allclose(window_attn(q, k, v, W), ref, atol=1e-12)

# rolling cache keeps exactly the last W keys
W = 4
K, V = np.zeros((W, d)), np.zeros((W, d))
for p in range(T):
    ring_update(K, V, k[p], v[p], p)
    last = range(max(0, p - W + 1), p + 1)
    for pp in last:
        assert np.array_equal(K[pp % W], k[pp])
        assert np.array_equal(V[pp % W], v[pp])
assert sorted(map(tuple, K.round(12))) == sorted(map(tuple, k[-W:].round(12)))

# receptive field: perturb token 0, farthest changed output = L*(W-1)
def stack(x, L, W, Ws):
    for l in range(L):
        Wq, Wk, Wv = Ws[l]
        x = x + window_attn(x @ Wq, x @ Wk, x @ Wv, W)
    return x

Tn, W, L = 40, 4, 3
Ws = [[rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3)] for _ in range(L)]
x0 = rng.normal(size=(Tn, d))
x1 = x0.copy()
x1[0] += 1.0
diff = np.abs(stack(x0, L, W, Ws) - stack(x1, L, W, Ws)).max(1) > 1e-12
assert np.flatnonzero(diff).max() == L * (W - 1), np.flatnonzero(diff)
print("sliding-window-attention OK")
