import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "kv-cache.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
T, d = 10, 6
X = rng.normal(size=(T, d))
W = {n: rng.normal(size=(d, d)) / np.sqrt(d) for n in "qkv"}
cache = ns["new_cache"](d)
inc = np.stack([ns["decode_step"](X[t], W, cache) for t in range(T)])
# full recomputation: causal attention over the whole sequence at once
Q, K, V = X @ W["q"], X @ W["k"], X @ W["v"]
ref = F.scaled_dot_product_attention(*(torch.tensor(m)[None] for m in (Q, K, V)), is_causal=True)[0].numpy()
# equal up to floating-point summation order (tight tolerance)
assert np.allclose(inc, ref, rtol=0, atol=1e-12), np.abs(inc - ref).max()
assert cache["K"].shape == (T, d) and cache["V"].shape == (T, d)
assert np.allclose(cache["K"], K) and np.allclose(cache["V"], V)
# cache size formula: 2 * t * d * bytes (one layer, one head)
assert cache["K"].nbytes + cache["V"].nbytes == 2 * T * d * X.itemsize
print("kv-cache OK")
