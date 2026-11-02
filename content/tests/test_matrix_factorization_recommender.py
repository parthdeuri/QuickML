import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "matrix-factorization-recommender.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
n, m, k = 40, 30, 3
U0, V0 = rng.normal(size=(n, k)), rng.normal(size=(m, k))
R = U0 @ V0.T + 0.1 * rng.normal(size=(n, m))
M = (rng.random((n, m)) < 0.5).astype(float)
M[7] = 0  # a user with no ratings
lam = 0.1
U, V, loss = ns["als"](R, M, k=k, lam=lam, iters=15)
# loss is non-increasing (exact block minimisation)
assert all(b <= a + 1e-9 for a, b in zip(loss, loss[1:])), loss
assert loss[-1] < loss[0]

# closed-form step equals the normal-equation solution via least squares
V1 = rng.normal(size=(m, k))
U1 = ns["solve_side"](R, M, V1, lam)
for i in (0, 5, 20):
    obs = M[i] > 0
    Aug = np.vstack([V1[obs], np.sqrt(lam) * np.eye(k)])
    b = np.r_[R[i, obs], np.zeros(k)]
    assert np.allclose(U1[i], np.linalg.lstsq(Aug, b, rcond=None)[0])
assert np.allclose(U1[7], 0)  # no ratings -> zero vector (cold start)


# the step is a minimiser: perturbing it never lowers the row objective
def row_obj(i, u):
    obs = M[i] > 0
    return ((R[i, obs] - V1[obs] @ u) ** 2).sum() + lam * u @ u


for _ in range(20):
    assert row_obj(5, U1[5] + 1e-3 * rng.normal(size=k)) >= row_obj(5, U1[5])

# held-out RMSE beats the global-mean baseline
hold = (M == 0) & (np.arange(n)[:, None] != 7)
rmse = np.sqrt(((R - U @ V.T)[hold] ** 2).mean())
base = np.sqrt(((R[hold] - R[M > 0].mean()) ** 2).mean())
assert rmse < base, (rmse, base)
print("matrix-factorization-recommender OK")
