import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "context-window-limits.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
d = 8


def measured(T):
    Q = rng.normal(size=(T, d)).astype(np.float32)
    K = rng.normal(size=(T, d)).astype(np.float32)
    S = Q @ K.T / np.float32(np.sqrt(d))  # stays float32
    return S


for T in (16, 32, 64):
    S = measured(T)
    flops, sb = ns["attn_cost"](T, d, bytes_per=4)
    assert S.shape == (T, T)
    assert S.nbytes == sb, (S.nbytes, sb)
    assert flops == 2 * (2 * T * T * d)  # QK^T plus A V

# doubling T: score bytes x4, FLOPs x4, KV bytes x2
f1, b1 = ns["attn_cost"](100, d)
f2, b2 = ns["attn_cost"](200, d)
assert b2 == 4 * b1 and f2 == 4 * f1
k1 = ns["kv_bytes"](100, 4, 2, 16)
k2 = ns["kv_bytes"](200, 4, 2, 16)
assert k2 == 2 * k1
assert k1 == 2 * 4 * 2 * 100 * 16 * 2  # 2 * layers * kv_heads * T * d_head * bytes

# sliding window is linear in T once T > w
w = 32
assert ns["window_score_bytes"](200, w) == 2 * ns["window_score_bytes"](100, w)
assert ns["window_score_bytes"](10, w) == ns["attn_cost"](10, d)[1]

# RoPE position interpolation
L_train, L_new, dh = 512, 2048, 16
pos = np.arange(L_new)
ang = ns["rope_angles"](pos, dh, L_train, L_new)
scaled = pos * (L_train / L_new)
assert scaled.max() < L_train
base = ns["rope_angles"](np.arange(L_train), dh)
assert ang.max() < ns["rope_angles"]([L_train], dh).max()  # inside trained range
assert np.allclose(ang, np.outer(scaled, 10000.0 ** (-np.arange(0, dh, 2) / dh)))
# unscaled extension would leave the trained range
assert ns["rope_angles"](pos, dh).max() > base.max()
# L_new == L_train is the identity
assert np.allclose(ns["rope_angles"](pos[:L_train], dh, 512, 512), base)
print("context-window-limits OK")
