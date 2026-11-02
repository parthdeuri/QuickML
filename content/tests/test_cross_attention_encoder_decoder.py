import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "cross-attention-encoder-decoder.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
Td, Te, d, dk, dv = 5, 7, 8, 4, 6      # target and source lengths differ
Xd = rng.normal(size=(Td, d))
Xe = rng.normal(size=(Te, d))
Wq, Wk = rng.normal(size=(d, dk)), rng.normal(size=(d, dk))
Wv = rng.normal(size=(d, dv))
mask = np.array([True] * 5 + [False] * 2)   # last two source tokens are padding

Q, K, V = (torch.tensor(a @ W) for a, W in ((Xd, Wq), (Xe, Wk), (Xe, Wv)))
for m in (None, mask):
    out, A = ns["cross_attention"](Xd, Xe, Wq, Wk, Wv, src_mask=m)
    am = None if m is None else torch.tensor(m)[None, :].expand(Td, Te)
    ref = F.scaled_dot_product_attention(Q, K, V, attn_mask=am)
    assert out.shape == (Td, dv) and A.shape == (Td, Te)
    assert np.allclose(out, ref.numpy()), m is None
    assert np.allclose(A.sum(axis=1), 1)
    if m is not None:
        assert np.all(A[:, ~m] == 0)

# changing padded source rows must not change the output
Xe2 = Xe.copy()
Xe2[5:] = rng.normal(size=(2, d)) * 100
o1, _ = ns["cross_attention"](Xd, Xe, Wq, Wk, Wv, src_mask=mask)
o2, _ = ns["cross_attention"](Xd, Xe2, Wq, Wk, Wv, src_mask=mask)
assert np.allclose(o1, o2)
print("cross-attention-encoder-decoder OK")
