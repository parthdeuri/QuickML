import numpy as np, yaml, pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer as SkCT
from sklearn.pipeline import Pipeline as SkPipe
from sklearn.preprocessing import StandardScaler as SkScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "sklearn-pipeline-column-transformer.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
n = 80
X = np.empty((n, 4), dtype=object)
X[:, 0] = rng.normal(5, 3, n)
X[:, 1] = rng.normal(-2, 0.5, n)
X[:, 2] = rng.choice(["red", "green", "blue"], n)
X[:, 3] = rng.choice(["a", "b"], n)
df = pd.DataFrame({"n0": X[:, 0].astype(float), "n1": X[:, 1].astype(float),
                   "c0": X[:, 2], "c1": X[:, 3]})
y = df.n0 * 2 - df.n1 + (df.c0 == "red") * 3 + rng.normal(0, 0.1, n)
y = y.to_numpy()
tr, te = np.arange(60), np.arange(60, n)

# 1. ColumnTransformer matches sklearn (fit on train, apply to test)
ct = ns["ColumnTransformer"]([(ns["StandardScaler"](), [0, 1]), (ns["OneHot"](), [2, 3])])
mine = ct.fit_transform(X[tr])
sk = SkCT([("num", SkScaler(), ["n0", "n1"]),
           ("cat", OneHotEncoder(sparse_output=False), ["c0", "c1"])])
ref = sk.fit_transform(df.iloc[tr])
assert mine.shape == ref.shape == (60, 7)
assert np.allclose(mine, ref, atol=1e-12)
assert np.allclose(ct.transform(X[te]), sk.transform(df.iloc[te]), atol=1e-12)

# 2. Pipeline with closed-form linear regression matches sklearn
pipe = ns["Pipeline"]([ns["ColumnTransformer"]([(ns["StandardScaler"](), [0, 1]), (ns["OneHot"](), [2, 3])]), ns["LinearReg"]()])
pipe.fit(X[tr], y[tr])
skp = SkPipe([("ct", SkCT([("num", SkScaler(), ["n0", "n1"]),
                           ("cat", OneHotEncoder(sparse_output=False), ["c0", "c1"])])),
              ("lr", LinearRegression())]).fit(df.iloc[tr], y[tr])
assert np.allclose(pipe.predict(X[te]), skp.predict(df.iloc[te]), atol=1e-8)
assert np.abs(pipe.predict(X[te]) - y[te]).mean() < 0.5

# 3. clone-before-fit: templates stay unfitted, fitted copies are separate
assert not hasattr(pipe.steps[0].parts[0][0], "mean_")
assert hasattr(pipe.fitted_[0].fitted_[0][0], "mean_")
assert not hasattr(pipe.steps[0], "fitted_")

# 4. leakage: fit on train reproduces train stats; fit on all changes test
Xn = df[["n0", "n1"]].to_numpy()
good = ns["StandardScaler"]().fit(Xn[tr])
assert np.allclose(good.mean_, Xn[tr].mean(0)) and np.allclose(good.std_, Xn[tr].std(0))
Ztr = good.transform(Xn[tr])
assert np.allclose(Ztr.mean(0), 0, atol=1e-12) and np.allclose(Ztr.std(0), 1)
leaky = ns["StandardScaler"]().fit(Xn)
assert not np.allclose(leaky.mean_, good.mean_)
assert not np.allclose(leaky.transform(Xn[te]), good.transform(Xn[te]))
assert not np.allclose(leaky.transform(Xn[tr]).mean(0), 0, atol=1e-6)

# 5. refit per fold: each fold's scaler matches its own training rows
for k in range(3):
    fold_te = np.arange(k * 20, k * 20 + 20)
    fold_tr = np.setdiff1d(np.arange(n), fold_te)
    s = ns["StandardScaler"]().fit(Xn[fold_tr])
    assert np.allclose(s.mean_, Xn[fold_tr].mean(0))

print("sklearn-pipeline-column-transformer OK")
