import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "weight-initialization.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
torch.manual_seed(0)
rng = np.random.default_rng(0)
fi, fo = 600, 400

# Xavier uniform: compare bound and variance with torch.nn.init
gain = torch.nn.init.calculate_gain("tanh")
W = ns["xavier_uniform"](fi, fo, rng, gain=gain)
Wt = torch.empty(fo, fi, dtype=torch.float64)
torch.nn.init.xavier_uniform_(Wt, gain=gain)
assert W.shape == tuple(Wt.shape)
bound = gain * np.sqrt(6 / (fi + fo))
assert np.abs(W).max() <= bound and Wt.abs().max().item() <= bound
target = gain ** 2 * 2 / (fi + fo)
assert np.isclose(W.var(), target, rtol=0.03)
assert np.isclose(Wt.var().item(), target, rtol=0.03)

# He normal: compare with kaiming_normal_ (fan_in, relu)
H = ns["he_normal"](fi, fo, rng)
Ht = torch.empty(fo, fi, dtype=torch.float64)
torch.nn.init.kaiming_normal_(Ht, mode="fan_in", nonlinearity="relu")
assert np.isclose(H.std(), 2 ** 0.5 / fi ** 0.5, rtol=0.03)
assert np.isclose(H.std(), Ht.std().item(), rtol=0.05)

# Signal through a deep ReLU stack: He stays O(1), bad scales do not
relu = lambda z: np.maximum(0, z)
width, depth = 256, 30
r_he = ns["signal_rms"](ns["he_normal"], relu, width, depth, rng)
small = lambda a, b, g: 0.01 * g.normal(size=(b, a))
large = lambda a, b, g: g.normal(size=(b, a))
r_small = ns["signal_rms"](small, relu, width, depth, rng)
r_large = ns["signal_rms"](large, relu, width, depth, rng)
assert 0.3 < r_he < 3, r_he
assert r_small < 1e-3 and r_large > 1e3, (r_small, r_large)
print("weight-initialization OK")
