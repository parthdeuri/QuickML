import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "bias-variance-tradeoff.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
f = lambda x: np.sin(3 * x)
x0, sigma = 0.5, 0.3

# Independent check: expected squared error at x0 against fresh noisy targets
# should equal bias^2 + variance + sigma^2 (different seed, 15% tolerance).
rng = np.random.default_rng(123)
for degree in (1, 3, 9):
    b2, var, noise = ns["bias_variance"](f, x0, degree, sigma=sigma, runs=1500)
    errs = []
    for _ in range(1500):
        x = rng.uniform(-1, 1, 30)
        y = f(x) + sigma * rng.normal(size=30)
        pred = np.polyval(np.polyfit(x, y, degree), x0)
        errs.append((pred - (f(x0) + sigma * rng.normal())) ** 2)
    assert np.isclose(np.mean(errs), b2 + var + noise, rtol=0.15), degree

b_lo, v_lo, _ = ns["bias_variance"](f, x0, 1)
b_hi, v_hi, _ = ns["bias_variance"](f, x0, 9)
assert b_lo > b_hi and v_hi > v_lo
print("bias-variance-tradeoff OK")
