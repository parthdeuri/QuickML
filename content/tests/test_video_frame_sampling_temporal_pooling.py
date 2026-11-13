import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "video-frame-sampling-temporal-pooling.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# uniform_indices: sorted, unique, in range, evenly spaced, centre-of-segment
for n in (10, 37, 100, 1000, 1234):
    for T in (1, 2, 3, 8, 16, 33):
        if T > n:
            continue
        idx = ns["uniform_indices"](n, T)
        assert len(idx) == T and len(set(idx.tolist())) == T
        assert np.all(np.diff(idx) > 0) and idx.min() >= 0 and idx.max() <= n - 1
        if T > 1:
            g = np.diff(idx)
            assert g.max() - g.min() <= 1, (n, T)
        ref = np.floor((np.arange(T) + 0.5) * n / T).astype(int)
        assert np.array_equal(idx, ref), (n, T)
assert ns["uniform_indices"](100, 4).tolist() == [12, 37, 62, 87]

# budget arithmetic, hand-checked
assert ns["budget_frames"](4096, 196, 512) == 18      # 3584 // 196
assert ns["budget_frames"](1000, 100, 0) == 10
assert ns["budget_frames"](1000, 300, 50) == 3
assert ns["budget_frames"](100, 50, 200) == 0
for ctx in (512, 2048, 8192):
    for m in (16, 64, 144, 576):
        for r in (0, 100, 600):
            t = ns["budget_frames"](ctx, m, r)
            assert t * m <= max(0, ctx - r) and (t + 1) * m > ctx - r

# temporal mean pool: ceil(T/k) x d with correct means (last group shorter)
rng = np.random.default_rng(0)
f = rng.normal(size=(10, 4))
p = ns["temporal_mean_pool"](f, 3)
assert p.shape == (int(np.ceil(10 / 3)), 4)
assert np.allclose(p[0], f[0:3].mean(0)) and np.allclose(p[3], f[9])
assert np.allclose(ns["temporal_mean_pool"](f, 1), f)
assert np.allclose(ns["temporal_mean_pool"](f, 5)[1], f[5:].mean(0))

# keyframes land on planted scene cuts
cuts = [0, 20, 45, 70]
shots = rng.normal(size=(4, 8)) * 5
seq = np.concatenate([np.tile(shots[i], (l, 1)) for i, l in enumerate([20, 25, 25, 30])])
seq = seq + rng.normal(size=seq.shape) * 0.05
kf = ns["keyframes_by_change"](seq, 2.0)
assert kf == cuts, kf
assert ns["keyframes_by_change"](np.ones((5, 3)), 0.5) == [0]
print("video-frame-sampling-temporal-pooling OK")
