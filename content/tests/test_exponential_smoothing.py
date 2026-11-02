import numpy as np, yaml
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "exponential-smoothing.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
y = 10 + 0.5 * np.arange(60) + np.cumsum(rng.normal(0, 1, 60))
a, b, l0, b0 = 0.3, 0.15, y[0], y[1] - y[0]
# simple smoothing: fixed alpha, known initial level, no optimisation
ref = SimpleExpSmoothing(y, initialization_method="known", initial_level=l0).fit(
    smoothing_level=a, optimized=False)
fit, level = ns["ses"](y, a, l0)
assert np.allclose(fit, ref.fittedvalues, atol=1e-10)
assert np.allclose(level, ref.forecast(3), atol=1e-10)
# Holt (additive trend, no damping), known initial level and slope
ref = ExponentialSmoothing(y, trend="add", initialization_method="known",
                           initial_level=l0, initial_trend=b0).fit(
    smoothing_level=a, smoothing_trend=b, optimized=False)
fit, fc = ns["holt"](y, a, b, l0, b0, h=8)
assert np.allclose(fit, ref.fittedvalues, atol=1e-10)
assert np.allclose(fc, ref.forecast(8), atol=1e-10)
# weights sum to one: a constant series stays constant
const, _ = ns["ses"](np.full(20, 4.0), 0.4, 4.0)
assert np.allclose(const, 4.0)
print("exponential-smoothing OK")
