import numpy as np, yaml
from pathlib import Path
from scipy.special import softmax

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "temperature-sampling.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

z = np.array([2.0, 1.0, 0.1, -1.0])
for T in [0.3, 0.5, 1.0, 2.0, 10.0]:
    assert np.allclose(ns["softmax_t"](z, T), softmax(z / T), atol=1e-12)
assert np.allclose(ns["softmax_t"](z, 1.0), softmax(z))
# stable for huge logits
assert np.isfinite(ns["softmax_t"]([1000.0, 999.0], 0.5)).all()

# entropy increases with T, towards log V
Ts = [0.05, 0.2, 0.5, 1, 2, 5, 20, 100]
H = [ns["entropy"](ns["softmax_t"](z, T)) for T in Ts]
assert all(a < b for a, b in zip(H, H[1:])), H
assert H[-1] < np.log(len(z)) and np.log(len(z)) - H[-1] < 1e-3
# hand-computed uniform entropy
assert abs(ns["entropy"](np.full(4, 0.25)) - np.log(4)) < 1e-12

# empirical frequencies match probabilities
for T in [0.7, 1.0, 2.0]:
    rng = np.random.default_rng(0)
    s = ns["sample"](z, T, rng, n=20000)
    freq = np.bincount(s, minlength=len(z)) / 20000
    assert np.abs(freq - softmax(z / T)).max() < 0.015, (T, freq)
# seeded: reproducible
a = ns["sample"](z, 1.0, np.random.default_rng(3), n=50)
b = ns["sample"](z, 1.0, np.random.default_rng(3), n=50)
assert (a == b).all()

# tiny T is greedy
s = ns["sample"](z, 1e-4, np.random.default_rng(0), n=200)
assert (s == np.argmax(z)).all()
print("temperature-sampling OK")
