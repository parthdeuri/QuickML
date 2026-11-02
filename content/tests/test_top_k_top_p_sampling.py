import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "top-k-top-p-sampling.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def brute(logits, top_k, top_p):
    """Sort-based definition: keep top-k, renormalise, keep smallest prefix with mass >= p."""
    idx = sorted(range(len(logits)), key=lambda i: -logits[i])
    if top_k > 0:
        idx = idx[:top_k]
    probs = softmax(np.array([logits[i] for i in idx]))
    kept, mass = [], 0.0
    for i, p in zip(idx, probs):
        kept.append(i)
        mass += p
        if mass >= top_p:
            break
    return set(kept)


rng = np.random.default_rng(0)
for trial in range(300):
    logits = rng.normal(size=int(rng.integers(5, 40))) * rng.uniform(0.5, 3)
    k = int(rng.integers(0, len(logits) + 1))
    p = float(rng.uniform(0.05, 1.0))
    z = ns["filter_logits"](logits, top_k=k, top_p=p)
    got = set(np.flatnonzero(np.isfinite(z)))
    assert got == brute(logits, k, p), (trial, k, p)
    assert np.allclose(z[list(got)], logits[list(got)])
# top_p = 1 and top_k = 0 keep everything
assert np.all(np.isfinite(ns["filter_logits"](logits)))
# sampling frequencies match the renormalised truncated distribution
logits = np.random.default_rng(1).normal(size=12) * 1.5
kw, tau, N = dict(top_k=6, top_p=0.8), 0.8, 20000
target = softmax(ns["filter_logits"](logits / tau, **kw))
srng = np.random.default_rng(2)
counts = np.bincount([ns["sample"](logits, srng, tau, **kw) for _ in range(N)], minlength=12)
freq = counts / N
assert np.all(freq[target == 0] == 0)
tol = 5 * np.sqrt(target * (1 - target) / N)  # five standard errors per token
assert np.all(np.abs(freq - target) <= tol + 1e-12), (freq, target)
print("top-k-top-p-sampling OK")
