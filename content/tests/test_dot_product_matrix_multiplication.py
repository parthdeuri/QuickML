import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "dot-product-matrix-multiplication.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

import itertools

rng = np.random.default_rng(0)
A, B = rng.normal(size=(4, 5)), rng.normal(size=(5, 3))
C = ns["matmul_naive"](A, B)
assert C.shape == (4, 3)
assert np.allclose(C, np.matmul(A, B)) and np.allclose(C, np.einsum("ik,kj->ij", A, B))


def orders(i, j):
    # every parenthesisation of matrices i..j as nested pairs
    if i == j:
        yield i
        return
    for s in range(i, j):
        for l in orders(i, s):
            for r in orders(s + 1, j):
                yield (l, r)


for n in range(3, 7):
    for trial in range(5):
        dims = [int(v) for v in rng.integers(1, 30, size=n + 1)]
        brute = min(ns["chain_cost"](dims, o) for o in orders(0, n - 1))
        best, split = ns["best_chain_cost"](dims)
        assert best == brute, (dims, best, brute)

# associativity: same answer for any grouping
x, W1, W2 = rng.normal(size=(1, 8)), rng.normal(size=(8, 8)), rng.normal(size=(8, 8))
assert np.allclose((x @ W1) @ W2, x @ (W1 @ W2))

# skinny x: (1x1000)(1000x1000)(1000x1000)
dims = [1, 1000, 1000, 1000]
left = ns["chain_cost"](dims, ((0, 1), 2))
right = ns["chain_cost"](dims, (0, (1, 2)))
assert left == 2 * 1 * 1000 * 1000 * 2 == 4_000_000
assert right == 2 * 1000 * 1000 * 1000 + 2 * 1 * 1000 * 1000 == 2_002_000_000
assert right // left == 500
assert ns["best_chain_cost"](dims)[0] == left
print("dot-product-matrix-multiplication OK")
