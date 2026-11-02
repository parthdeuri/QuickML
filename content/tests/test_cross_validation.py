import numpy as np, yaml
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "cross-validation.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
X = rng.normal(size=(103, 3))
y = X @ [1.0, -2.0, 0.5] + 0.3 * rng.normal(size=103)

ref = list(KFold(5, shuffle=True, random_state=0).split(X))
mine = list(ns["kfold_indices"](len(X), 5, 0))
assert len(mine) == len(ref)
for (tr, va), (rtr, rva) in zip(mine, ref):
    assert np.array_equal(np.sort(va), rva)
    assert np.array_equal(np.sort(tr), rtr)
assert sorted(np.concatenate([va for _, va in mine])) == list(range(103))


def fit(A, b):
    A = np.c_[np.ones(len(A)), A]
    return np.linalg.lstsq(A, b, rcond=None)[0]


def mse(w, A, b):
    return np.mean((np.c_[np.ones(len(A)), A] @ w - b) ** 2)


scores = ns["cross_val_score"](fit, mse, X, y, k=5, seed=0)
sk = -cross_val_score(LinearRegression(), X, y, scoring="neg_mean_squared_error",
                      cv=KFold(5, shuffle=True, random_state=0))
# sklearn's mean of per-fold MSE matches since folds are identical
assert np.allclose(scores, sk, atol=1e-10)
print("cross-validation OK")
