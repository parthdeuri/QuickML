import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "self-attention.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
T, d, dk, dv = 6, 8, 4, 5
X = rng.normal(size=(T, d))
Wq, Wk = rng.normal(size=(d, dk)), rng.normal(size=(d, dk))
Wv = rng.normal(size=(d, dv))
Q, K, V = (torch.tensor(X @ W) for W in (Wq, Wk, Wv))
for causal in (False, True):
    out, A = ns["self_attention"](X, Wq, Wk, Wv, causal=causal)
    ref = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    assert np.allclose(out, ref.numpy()), causal
    assert np.allclose(A.sum(axis=1), 1)
    if causal:
        assert np.all(np.triu(A, 1) == 0)
print("self-attention OK")
