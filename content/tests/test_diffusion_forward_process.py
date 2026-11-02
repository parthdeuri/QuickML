import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "diffusion-forward-process.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

T = 200
betas, abar = ns["make_schedule"](T)
alphas = 1 - betas
assert np.allclose(abar, np.cumprod(alphas))
x0 = np.array([1.5, -0.7, 0.3])
# iterate the one-step kernel N(sqrt(1-b) x, b) analytically on (mean, variance)
m, v = x0.copy(), 0.0
for t in range(T):
    m, v = np.sqrt(alphas[t]) * m, alphas[t] * v + betas[t]
    assert np.allclose(m, np.sqrt(abar[t]) * x0)
    assert np.isclose(v, 1 - abar[t])
# Monte-Carlo: simulate the chain step by step and compare with q_sample
rng = np.random.default_rng(0)
N, t_star = 100000, 60
x = np.tile(x0, (N, 1))
for t in range(t_star + 1):
    x = ns["q_step"](x, t, betas, rng.normal(size=x.shape))
y = ns["q_sample"](np.tile(x0, (N, 1)), t_star, abar, rng.normal(size=(N, 3)))
mean_t, var_t = np.sqrt(abar[t_star]) * x0, 1 - abar[t_star]
for s in (x, y):
    # tolerance: five standard errors of the sample mean and variance
    assert np.all(np.abs(s.mean(0) - mean_t) < 5 * np.sqrt(var_t / N))
    assert np.all(np.abs(s.var(0) - var_t) < 5 * var_t * np.sqrt(2 / N))
# terminal distribution is close to N(0, 1) for a long schedule
b, ab = ns["make_schedule"](1000)
assert ab[-1] < 1e-3 and np.all(np.diff(ab) < 0)
# noise recovery: x0 = (x_t - sqrt(1-abar) eps) / sqrt(abar)
eps = rng.normal(size=3)
xt = ns["q_sample"](x0, 10, abar, eps)
assert np.allclose((xt - np.sqrt(1 - abar[10]) * eps) / np.sqrt(abar[10]), x0)
# SNR strictly decreasing in t
snr = abar / (1 - abar)
assert np.all(np.diff(snr) < 0)
print("diffusion-forward-process OK")
