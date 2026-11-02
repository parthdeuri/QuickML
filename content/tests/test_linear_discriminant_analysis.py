import numpy as np, yaml
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "linear-discriminant-analysis.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
cov = np.array([[1.0, 0.4, 0, 0], [0.4, 1.0, 0, 0], [0, 0, 0.5, 0.1], [0, 0, 0.1, 0.8]])
means = np.array([[0, 0, 0, 0], [3, 1, 0, 1], [1, 4, 1, -1.0]])
X = np.vstack([rng.multivariate_normal(m, cov, 100) for m in means])
y = np.repeat(np.arange(3), 100)
W = ns["lda_fit"](X, y, 2)


def unit(M):
    return M / np.linalg.norm(M, axis=0)


for solver in ("eigen", "svd"):
    ref = LDA(solver=solver, n_components=2).fit(X, y)
    Wr = ref.scalings_[:, :2]
    # directions match up to sign (scale conventions differ between solvers)
    cos = np.abs((unit(W) * unit(Wr)).sum(0))
    assert np.allclose(cos, 1, atol=1e-6), (solver, cos)
    # projections match up to sign and a per-component scale
    Z, Zr = (X - X.mean(0)) @ W, ref.transform(X)
    for j in range(2):
        r = np.corrcoef(Z[:, j], Zr[:, j])[0, 1]
        assert abs(abs(r) - 1) < 1e-8, (solver, j, r)

# Balanced classes (equal priors) + Sw-normalised W: nearest class mean in
# projected space reproduces sklearn's predictions.
ref = LDA(solver="eigen").fit(X, y)
Zm = np.array([(X[y == c] @ W).mean(0) for c in range(3)])
d2 = (((X @ W)[:, None, :] - Zm) ** 2).sum(2)
pred = d2.argmin(1)
assert (pred == ref.predict(X)).mean() == 1.0


# Fisher ratio is maximised: the top direction beats random directions
def fisher(w):
    Sw = sum((((X[y == c] - X[y == c].mean(0)) @ w) ** 2).sum() for c in range(3))
    Sb = sum(100 * ((X[y == c].mean(0) - X.mean(0)) @ w) ** 2 for c in range(3))
    return Sb / Sw


assert all(fisher(W[:, 0]) >= fisher(rng.normal(size=4)) for _ in range(50))
print("linear-discriminant-analysis OK")
