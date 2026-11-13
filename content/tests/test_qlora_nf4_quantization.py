import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "qlora-nf4-quantization.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

lv = ns["make_nf4_levels"]()
assert lv.shape == (16,) and np.all(np.diff(lv) > 0)
assert lv.min() == -1.0 and lv.max() == 1.0 and (lv == 0).sum() == 1
assert (lv > 0).sum() == 8 and (lv < 0).sum() == 7

rng = np.random.default_rng(0)
w = rng.normal(size=64 * 400)
idx, s = ns["quantize_blockwise"](w, 64)
assert idx.max() < 16 and idx.shape == (400, 64) and s.shape == (400, 1)
# each block's max abs maps to a level of magnitude 1
blocks = w.reshape(-1, 64)
am = np.abs(blocks).argmax(axis=1)
top = ns["make_nf4_levels"]()[idx[np.arange(400), am]]
assert np.allclose(np.abs(top), 1.0)
assert np.allclose(s[:, 0], np.abs(blocks).max(axis=1))
rec = ns["dequantize"](idx, s)
err_nf4 = np.mean((w - rec) ** 2)

# uniform 4-bit absmax baseline (levels -7..7 over absmax), same blocks
su = np.abs(blocks).max(axis=1, keepdims=True) / 7
rec_u = (np.clip(np.round(blocks / su), -7, 7) * su).ravel()
err_u = np.mean((w - rec_u) ** 2)
assert err_nf4 < err_u, (err_nf4, err_u)
# zeros are reproduced exactly
z = np.zeros(64); z[0] = 1.0
i2, s2 = ns["quantize_blockwise"](z, 64)
assert np.array_equal(ns["dequantize"](i2, s2), z)

# bytes-per-param arithmetic
b = ns["bytes_per_param"]
assert np.isclose(b(64, False), (4 + 32 / 64) / 8)          # 0.5625
assert np.isclose(b(64, True), (4 + 8 / 64 + 32 / (64 * 256)) / 8)
assert np.isclose(8 / 64 + 32 / (64 * 256), 0.126953125)
assert b(64, True) < b(64, False) and b(64, True) > 0.5
print("qlora-nf4-quantization OK")
