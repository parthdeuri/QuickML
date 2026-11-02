import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "multi-head-attention.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
torch.manual_seed(0)
rng = np.random.default_rng(0)
T, d, h = 6, 12, 3
X = rng.normal(size=(T, d))
Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4))
out, A = ns["multi_head_attention"](X, Wq, Wk, Wv, Wo, h)

mha = torch.nn.MultiheadAttention(d, h, bias=False, batch_first=True).double()
with torch.no_grad():
    mha.in_proj_weight.copy_(torch.tensor(np.concatenate([Wq.T, Wk.T, Wv.T])))
    mha.out_proj.weight.copy_(torch.tensor(Wo.T))
x = torch.tensor(X)[None]
ref, w = mha(x, x, x, need_weights=True, average_attn_weights=False)
assert out.shape == (T, d) and A.shape == (h, T, T)
assert np.allclose(out, ref[0].detach().numpy())
assert np.allclose(A, w[0].detach().numpy())
assert np.allclose(A.sum(axis=-1), 1)
print("multi-head-attention OK")
