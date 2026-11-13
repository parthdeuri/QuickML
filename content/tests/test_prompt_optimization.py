import numpy as np, yaml
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "prompt-optimization.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# Wilson interval matches statsmodels
for k, n in [(0, 10), (3, 10), (10, 10), (45, 100), (1, 200)]:
    for z, alpha in [(1.96, 0.05), (1.6449, 0.10)]:
        lo, hi = ns["wilson_interval"](k, n, z)
        rlo, rhi = proportion_confint(k, n, alpha=alpha, method="wilson")
        assert abs(lo - rlo) < 1e-3 and abs(hi - rhi) < 1e-3, (k, n, lo, hi, rlo, rhi)
lo, hi = ns["wilson_interval"](45, 100, 1.959963984540054)
rlo, rhi = proportion_confint(45, 100, method="wilson")
assert abs(lo - rlo) < 1e-12 and abs(hi - rhi) < 1e-12

# deterministic stub model: example difficulty vs prompt quality
def model_fn(prompt, x):
    quality = sum(w in prompt for w in ("step", "concise", "example"))
    hard = (x * 37 % 100) / 100
    return x % 2 if hard < 0.3 + 0.2 * quality else 1 - x % 2

data = [(x, x % 2) for x in range(200)]
muts = [lambda p: p + " Think step by step.",
        lambda p: p + " Be concise.",
        lambda p: p.upper(),
        lambda p: p + " Give an example."]
start = "Classify the text."
best, hist = ns["hill_climb"](start, muts, data, model_fn)
s0 = ns["evaluate"](start, data, model_fn).mean()
assert hist[0] == s0
assert ns["evaluate"](best, data, model_fn).mean() >= s0
assert all(b >= a for a, b in zip(hist, hist[1:])), hist
assert hist[-1] > s0
# mutations that only hurt never lower the score
b2, h2 = ns["hill_climb"](best, [lambda p: ""], data, model_fn)
assert b2 == best and h2[-1] == h2[0]

# paired bootstrap
rng = np.random.default_rng(1)
a = (rng.random(300) < 0.6).astype(float)
assert ns["paired_bootstrap_prob"](a, a.copy(), 1000, 0) == 0.5
good = (rng.random(300) < 0.95).astype(float)
bad = (rng.random(300) < 0.40).astype(float)
assert ns["paired_bootstrap_prob"](good, bad, 1000, 0) > 0.999
assert ns["paired_bootstrap_prob"](bad, good, 1000, 0) < 0.001
p1 = ns["paired_bootstrap_prob"](good, a, 500, 7)
p2 = ns["paired_bootstrap_prob"](good, a, 500, 7)
assert p1 == p2
print("prompt-optimization OK")
