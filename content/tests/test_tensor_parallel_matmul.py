import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "tensor-parallel-matmul.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
gelu, tp_mlp = ns["gelu"], ns["tp_mlp"]

rng = np.random.default_rng(0)
B, T, d, h = 2, 5, 8, 32
x = rng.normal(size=(B, T, d))
W1 = rng.normal(size=(d, h)) / np.sqrt(d)
W2 = rng.normal(size=(h, d)) / np.sqrt(h)
ref = gelu(x @ W1) @ W2
for n in (2, 4, 8):
    out = tp_mlp(x, W1, W2, n)
    assert out.shape == ref.shape
    assert np.allclose(out, ref, rtol=0, atol=1e-12), (n, np.abs(out - ref).max())
    assert [a.shape for a in ns["column_split"](W1, n)] == [(d, h // n)] * n
    assert [b.shape for b in ns["row_split"](W2, n)] == [(h // n, d)] * n

# failing alternative: row-split W1 (devices see partial sums), activation applied locally
n = 4
xs = np.split(x, n, axis=-1)
A_rows = ns["row_split"](W1, n)
bad = ns["all_reduce_sum"]([gelu(xi @ a) @ W2 / n for xi, a in zip(xs, A_rows)])
assert not np.allclose(bad, ref, atol=1e-3), "row-split first matrix must fail"
# without the nonlinearity the same row split would be exact, so the gelu is the culprit
lin = ns["all_reduce_sum"]([(xi @ a) @ W2 for xi, a in zip(xs, A_rows)])
assert np.allclose(lin, (x @ W1) @ W2, atol=1e-12)
# summing activations of partial sums, then applying gelu once, would be right but needs communication first
mid = ns["all_reduce_sum"]([xi @ a for xi, a in zip(xs, A_rows)])
assert np.allclose(gelu(mid) @ W2, ref, atol=1e-12)

# comm bytes, hand-checked: 2*4*8 values * 2 bytes = 128; 1*10*16*4 = 640
assert ns["comm_bytes"](2, 4, 8, 2) == 128
assert ns["comm_bytes"](1, 10, 16, 4) == 640
print("tensor-parallel-matmul OK")
