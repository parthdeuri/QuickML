import numpy as np, yaml
from pathlib import Path
from sklearn.linear_model import Perceptron

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "perceptron-learning-rule.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
fit, predict, margin = ns["perceptron_fit"], ns["predict"], ns["margin"]

# separable data with a known unit-norm separator and a margin gap
rng = np.random.default_rng(0)
ws = np.array([1.0, -2.0]) / np.sqrt(5)
bs = 0.3
P = rng.uniform(-3, 3, size=(600, 2))
s = P @ ws + bs
keep = np.abs(s) > 0.3
X, y = P[keep][:100], np.where(s[keep] > 0, 1, -1)[:100]

w, b, k = fit(X, y, epochs=500)
assert (predict(X, w, b) == y).all(), "training error must be 0"
# bound with bias absorbed into an augmented, unit-norm separator
Xa = np.c_[X, np.ones(len(X))]
wa = np.r_[ws, bs]
wa = wa / np.linalg.norm(wa)
R = np.linalg.norm(Xa, axis=1).max()
gamma = margin(Xa, y, wa)
assert gamma > 0
bound = (R / gamma) ** 2
assert 0 < k <= bound, (k, bound)
# the learned separator has a positive margin too, and a mistake-free pass stops the loop
assert margin(Xa, y, np.r_[w, b]) > 0
_, _, k_more = fit(X, y, epochs=5000)
assert k_more == k
# learning rate only rescales the weights from a zero start
w2, b2, k2 = fit(X, y, epochs=500, eta=0.1)
assert k2 == k and np.allclose(w2 * 10, w) and np.isclose(b2 * 10, b)

# XOR: never reaches zero training error
Xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
yx = np.array([-1, 1, 1, -1])
for ep in (1, 2, 5, 50, 500):
    wx, bx, kx = fit(Xx, yx, epochs=ep)
    assert (predict(Xx, wx, bx) != yx).any()
    assert kx >= ep  # at least one mistake in every epoch

# match sklearn (same data order, no shuffle, no penalty, no tol)
for data, tgt, ep in ((X, y, 30), (Xx, yx, 7)):
    clf = Perceptron(eta0=1.0, shuffle=False, penalty=None, tol=None,
                     max_iter=ep, fit_intercept=True).fit(data, tgt)
    w3, b3, _ = fit(data, tgt, epochs=ep)
    assert np.allclose(clf.coef_[0], w3), (clf.coef_, w3)
    assert np.allclose(clf.intercept_[0], b3)
    assert (clf.predict(data) == predict(data, w3, b3)).all()

print("perceptron-learning-rule OK")
