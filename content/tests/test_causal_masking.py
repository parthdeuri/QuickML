import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "causal-masking.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
T, d = 7, 5
Q, K, V = (rng.normal(size=(T, d)) for _ in range(3))
out, A = ns["causal_attention"](Q, K, V)
ref = F.scaled_dot_product_attention(*(torch.tensor(m)[None] for m in (Q, K, V)), is_causal=True)
assert np.allclose(out, ref[0].numpy())
assert np.all(np.triu(A, 1) == 0) and np.allclose(A.sum(-1), 1)
# no leak: perturb tokens j.. in Q, K, V; outputs before j must not change
for j in range(1, T):
    Q2, K2, V2 = Q.copy(), K.copy(), V.copy()
    for m in (Q2, K2, V2):
        m[j:] += rng.normal(size=(T - j, d))
    out2, _ = ns["causal_attention"](Q2, K2, V2)
    assert np.array_equal(out2[:j], out[:j]), j
    assert not np.allclose(out2[j:], out[j:])
# changing only the first token does change later outputs (the past is visible)
V3 = V.copy()
V3[0] += 1
assert not np.allclose(ns["causal_attention"](Q, K, V3)[0][1:], out[1:])
print("causal-masking OK")
