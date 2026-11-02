import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "svd.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
A = rng.normal(size=(30, 8)) @ np.diag([5, 4, 3, 2, 1.5, 1, 0.8, 0.5])
U, s, Vt = ns["svd"](A)
Ur, sr, Vtr = np.linalg.svd(A, full_matrices=False)

assert np.allclose(s, sr)
assert np.allclose((U * s) @ Vt, A)
assert np.allclose(U.T @ U, np.eye(8), atol=1e-8) and np.allclose(Vt @ Vt.T, np.eye(8))
sign = np.sign((Vt * Vtr).sum(1))[:, None]          # signs are arbitrary
assert np.allclose(Vt * sign, Vtr, atol=1e-8)
assert np.allclose(U * sign.T, Ur, atol=1e-8)

# Eckart-Young: truncation error equals discarded singular values, in both norms
for k in range(0, 8):
    Ak = ns["low_rank"](U, s, Vt, k) if k else np.zeros_like(A)
    assert np.isclose(np.linalg.norm(A - Ak), ns["rank_error"](s, k))
    assert np.isclose(np.linalg.norm(A - Ak, 2), s[k] if k < 8 else 0)
    assert np.allclose(Ak, (Ur[:, :k] * sr[:k]) @ Vtr[:k])
    # no random rank-k matrix does better (Frobenius)
    for _ in range(30):
        B = rng.normal(size=(30, k)) @ rng.normal(size=(k, 8)) if k else Ak
        assert np.linalg.norm(A - B) >= ns["rank_error"](s, k) - 1e-9
    # least-squares-optimal rank-k within the column space of the top-k left vectors
    P = Ur[:, :k] @ Ur[:, :k].T
    assert np.isclose(np.linalg.norm(A - P @ A), ns["rank_error"](s, k))
print("svd OK")
