import numpy as np, yaml
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "arima-model.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(2)
n, p, d, h = 800, 2, 1, 6
e = rng.normal(size=n)
z = np.zeros(n)
for t in range(2, n):
    z[t] = 0.2 + 0.5 * z[t - 1] - 0.3 * z[t - 2] + e[t]
y = 50 + np.cumsum(z)
fc, beta = ns["forecast"](y, p, d, h)
dz = np.diff(y)
# exact match: AutoReg is conditional least squares with a constant
ar = AutoReg(dz, lags=p, trend="c").fit()
assert np.allclose(beta, ar.params, atol=1e-8)
pred = ar.predict(start=len(dz), end=len(dz) + h - 1)
assert np.allclose(fc, y[-1] + np.cumsum(pred), atol=1e-8)
# the simulated coefficients are recovered within sampling error
assert np.allclose(beta[1:], [0.5, -0.3], atol=0.08)
# statsmodels ARIMA maximises the exact likelihood, so the AR coefficients
# on the differenced series are only close, not identical
m = ARIMA(dz, order=(p, 0, 0), trend="c").fit()
assert np.allclose(m.params[1:p + 1], beta[1:], atol=0.01)
# d = 0 shape, and d = 2 integration against AutoReg on the 2nd difference
assert ns["forecast"](z, 2, 0, 4)[0].shape == (4,)
yy = np.cumsum(y)
f2, _ = ns["forecast"](yy, 2, 2, 4)
d2 = np.diff(yy, 2)
ar2 = AutoReg(d2, lags=2, trend="c").fit()
q = ar2.predict(start=len(d2), end=len(d2) + 3)
lvl1 = np.diff(yy)[-1] + np.cumsum(q)
assert np.allclose(f2, yy[-1] + np.cumsum(lvl1), atol=1e-6)
print("arima-model OK")
