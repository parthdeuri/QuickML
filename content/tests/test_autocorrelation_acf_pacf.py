import numpy as np, yaml
from pathlib import Path
from statsmodels.tsa.stattools import acf, pacf

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "autocorrelation-acf-pacf.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(1)
e = rng.normal(size=600)
x = np.zeros(600)
for t in range(2, 600):
    x[t] = 0.6 * x[t - 1] - 0.3 * x[t - 2] + e[t]
K = 12
r = ns["acf"](x, K)
# statsmodels acf uses the divisor n at every lag (adjusted=False), as here
assert np.allclose(r, acf(x, nlags=K, adjusted=False, fft=False), atol=1e-12)
p = ns["pacf"](r, K)
# Durbin-Levinson on the biased ACF == statsmodels "ywm" (Yule-Walker with no
# sample-size adjustment); "yw" with adjusted covariances differs slightly
assert np.allclose(p, pacf(x, nlags=K, method="ywm"), atol=1e-10)
# independent check: solve the Toeplitz Yule-Walker system for order k
for k in (1, 3, 5):
    R = np.array([[r[abs(i - j)] for j in range(k)] for i in range(k)])
    assert np.isclose(np.linalg.solve(R, r[1:k + 1])[-1], p[k], atol=1e-10)
# AR(2): PACF cuts off after lag 2, true phi_22 = -0.3
assert abs(p[2] + 0.3) < 0.08 and np.all(np.abs(p[3:]) < 3 / np.sqrt(600))
print("autocorrelation-acf-pacf OK")
