import numpy as np, yaml
from pathlib import Path
from scipy.optimize import curve_fit

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "scaling-laws.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
loss, optimal_ND, fit = ns["loss"], ns["optimal_ND"], ns["fit"]

# closed-form optimum: constraint satisfied and matches brute force
for C in [1e20, 1e22, 1e24]:
    N, D = optimal_ND(C)
    assert abs(6 * N * D / C - 1) < 1e-12
    Ng = np.logspace(np.log10(N) - 1, np.log10(N) + 1, 200001)
    Lg = loss(Ng, C / (6 * Ng))
    i = Lg.argmin()
    assert abs(Ng[i] / N - 1) < 1e-3, (C, Ng[i], N)
    assert loss(N, D) <= Lg.min() + 1e-12
    # perturbing N along the constraint can only raise the loss
    for f in (0.9, 1.1):
        assert loss(N * f, C / (6 * N * f)) > loss(N, D)

# optimal N and D both grow with budget; N* ~ C^(b/(a+b))
a, b = 0.34, 0.28
N1, D1 = optimal_ND(1e21)
N2, D2 = optimal_ND(1e23)
assert abs(np.log(N2 / N1) / np.log(100) - b / (a + b)) < 1e-9
assert abs(np.log(D2 / D1) / np.log(100) - a / (a + b)) < 1e-9

# recover parameters from noiseless synthetic data
rng = np.random.default_rng(0)
N = 10 ** rng.uniform(7, 10, 40)
D = 10 ** rng.uniform(8, 12, 40)
true = (1.69, 406.4, 410.7, 0.34, 0.28)
L = loss(N, D, *true[:3], *true[3:])
est = fit(N, D, L)
assert np.allclose(est, true, rtol=1e-5), est

f = lambda X, E, A, B, a, b: E + A / X[0]**a + B / X[1]**b
popt, _ = curve_fit(f, (N, D), L, p0=[1.5, 300, 300, 0.3, 0.3],
                    method="trf", max_nfev=20000, x_scale=[1, 100, 100, 0.1, 0.1])
assert np.allclose(popt, true, rtol=1e-4), popt
assert np.allclose(est, popt, rtol=1e-4)
print("scaling-laws OK")
