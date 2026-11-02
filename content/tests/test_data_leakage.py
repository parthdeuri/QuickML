import numpy as np, yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "data-leakage.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
# test rows come from a shifted distribution so the leak is visible
X = np.r_[rng.normal(0, 1, (80, 3)), rng.normal(3, 2, (20, 3))]
n = 80

tr, te = ns["split_then_scale"](X, n)
sc = StandardScaler().fit(X[:n])
assert np.allclose(tr, sc.transform(X[:n])) and np.allclose(te, sc.transform(X[n:]))
assert np.allclose(tr.mean(0), 0, atol=1e-12)
assert np.abs(te.mean(0)).min() > 1.0  # honest test set shows the shift

ltr, lte = ns["leaky_scale"](X, n)
lsc = StandardScaler().fit(X)
assert np.allclose(np.r_[ltr, lte], lsc.transform(X))
assert not np.allclose(ltr, tr)            # train stats were altered by test rows
assert np.abs(lte.mean(0)).max() < np.abs(te.mean(0)).min()  # shift partly hidden

# Classic selection leak: pure-noise features, random labels.
Xn = rng.normal(size=(60, 1000))
yn = rng.integers(0, 2, 60)
cv = KFold(5, shuffle=True, random_state=0)
sel = SelectKBest(f_classif, k=10).fit(Xn, yn)          # labels used on all data
leaky = cross_val_score(LogisticRegression(), sel.transform(Xn), yn, cv=cv).mean()
pipe = make_pipeline(SelectKBest(f_classif, k=10), LogisticRegression())
honest = cross_val_score(pipe, Xn, yn, cv=cv).mean()
assert leaky > 0.75 and leaky > honest + 0.2, (leaky, honest)
print("data-leakage OK")
