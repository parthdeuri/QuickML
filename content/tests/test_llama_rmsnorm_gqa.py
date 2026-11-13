import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "llama-rmsnorm-gqa.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
# --- RMSNorm vs torch
x = rng.normal(size=(5, 8)) * 3 + 1
g = rng.normal(size=8)
eps = 1e-6
mine = ns["rms_norm"](x, g, eps)
tx, tg = torch.tensor(x), torch.tensor(g)
if hasattr(torch.nn, "RMSNorm"):
    m = torch.nn.RMSNorm(8, eps=eps, dtype=torch.float64)
    with torch.no_grad():
        m.weight.copy_(tg)
    ref = m(tx).detach().numpy()
else:
    ref = (tx * torch.rsqrt(tx.pow(2).mean(-1, keepdim=True) + eps) * tg).numpy()
assert np.allclose(mine, ref, atol=1e-12)
# no mean subtraction: differs from layer norm unless the mean is zero
assert abs(mine.mean()) > 1e-3

# --- GQA vs torch
T, d, n_q = 7, 16, 4
dh = d // n_q
X = rng.normal(size=(T, d))

def torch_ref(X, Wq, Wk, Wv, n_kv, interleave=True):
    tX = torch.tensor(X)
    q = (tX @ torch.tensor(Wq)).reshape(T, n_q, dh).transpose(0, 1)
    k = (tX @ torch.tensor(Wk)).reshape(T, n_kv, dh).transpose(0, 1)
    v = (tX @ torch.tensor(Wv)).reshape(T, n_kv, dh).transpose(0, 1)
    if n_kv != n_q:
        k = k.repeat_interleave(n_q // n_kv, dim=0)
        v = v.repeat_interleave(n_q // n_kv, dim=0)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return o.transpose(0, 1).reshape(T, d).numpy()

for n_kv in (4, 2, 1):  # MHA, GQA, MQA
    Wq = rng.normal(size=(d, d)) / 4
    Wk = rng.normal(size=(d, n_kv * dh)) / 4
    Wv = rng.normal(size=(d, n_kv * dh)) / 4
    out = ns["gqa"](X, Wq, Wk, Wv, n_q, n_kv)
    assert out.shape == (T, d)
    assert np.allclose(out, torch_ref(X, Wq, Wk, Wv, n_kv), atol=1e-12), n_kv
    # KV cache bytes ratio
    full = ns["kv_bytes"](100, n_q, dh)
    assert ns["kv_bytes"](100, n_kv, dh) / full == n_kv / n_q
    assert full / ns["kv_bytes"](100, n_kv, dh) == n_q / n_kv

# MQA: all heads share the one KV head -> identical K/V per head
print("llama-rmsnorm-gqa OK")
