import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "int8-quantization.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
qs, dq, qa, mm = ns["quantize_sym"], ns["dequantize"], ns["quantize_asym"], ns["int8_matmul"]

rng = np.random.default_rng(0)
# error bound: |x - dequant(quant(x))| <= scale / 2
x = rng.normal(size=(5, 40))
q, s = qs(x)
assert q.dtype == np.int8 and np.abs(q).max() <= 127 and s.shape == (1, 1)
assert np.abs(x - dq(q, s)).max() <= s.item() / 2 + 1e-12
assert abs(s.item() - np.abs(x).max() / 127) < 1e-15

# per-channel beats per-tensor when channel ranges are uneven
W = rng.normal(size=(16, 64)) * np.logspace(-2, 1, 16)[:, None]
qt, st = qs(W)
qc, sc = qs(W, axis=1)
et = np.abs(W - dq(qt, st)).max(axis=1)
ec = np.abs(W - dq(qc, sc)).max(axis=1)
assert (ec <= sc[:, 0] / 2 + 1e-12).all() and (et <= st.item() / 2 + 1e-12).all()
assert np.abs(W - dq(qc, sc)).mean() < np.abs(W - dq(qt, st)).mean()
assert (ec <= et + 1e-12).all(), (ec, et)

# int8 matmul vs float matmul: per-row x scales, per-channel w scales
X = rng.normal(size=(8, 64))
xq, sx = qs(X, axis=1)
Y = mm(xq, sx, qc, sc)
ref = X @ W.T
# |err| <= sum_k (|x_k| s_w/2 + |w_k| s_x/2 + s_x s_w/4)
bound = (np.abs(X).sum(1, keepdims=True) * (sc / 2).T + (sx / 2) * np.abs(W).sum(1)[None, :]
         + 64 * sx * sc.T / 4)
assert (np.abs(Y - ref) <= bound + 1e-9).all()
assert np.abs(Y - ref).max() < 0.05 * np.abs(ref).max()
# integer part is exact
acc = xq.astype(np.int64) @ qc.astype(np.int64).T
assert np.allclose(Y, acc * (sx * sc.T))

# matches torch per-tensor int8 quantisation with the same scale, zero point 0
x32 = rng.normal(size=(6, 30)).astype(np.float32)
q1, s1 = qs(x32.astype(np.float64))
scale = float(s1.item())
tq = torch.quantize_per_tensor(torch.tensor(x32), scale, 0, torch.qint8)
assert np.array_equal(tq.int_repr().numpy(), q1)
assert np.allclose(tq.dequantize().numpy(), dq(q1, s1), atol=1e-5)

# asymmetric: covers the full range with error <= s / 2
a = rng.uniform(0.0, 6.0, size=500)
qa_, sa, z = qa(a)
assert qa_.min() == -128 and qa_.max() == 127
assert np.abs(a - sa * (qa_.astype(float) - z)).max() <= sa / 2 + 1e-12
print("int8-quantization OK")
