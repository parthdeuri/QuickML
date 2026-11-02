import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "graph-attention-layer.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
torch.manual_seed(0)
rng = np.random.default_rng(0)
N, F_in, F_out = 7, 5, 4
A = np.triu((rng.random((N, N)) < 0.4).astype(float), 1)
A = A + A.T
X = rng.normal(size=(N, F_in))
W = rng.normal(size=(F_in, F_out))
a_i = rng.normal(size=F_out)
a_j = rng.normal(size=F_out)

out, alpha = ns["gat_layer"](A, X, W, a_i, a_j)

# 1) dense torch GAT: concat form aT[h_i || h_j], masked softmax, LeakyReLU
Xt, Wt = torch.tensor(X), torch.tensor(W)
a = torch.tensor(np.concatenate([a_i, a_j]))
H = Xt @ Wt
pairs = torch.cat([H[:, None, :].expand(N, N, F_out), H[None, :, :].expand(N, N, F_out)], dim=-1)
e = F.leaky_relu(pairs @ a, negative_slope=0.2)
mask = (torch.tensor(A) + torch.eye(N, dtype=torch.float64)) > 0
att = torch.softmax(e.masked_fill(~mask, float("-inf")), dim=1)
ref = att @ H
assert np.allclose(alpha, att.numpy())
assert np.allclose(out, ref.numpy())
assert np.allclose(alpha.sum(1), 1) and (alpha[~mask.numpy()] == 0).all()

# 2) torch_geometric GATConv if installed (single head, self loops, no bias)
try:
    from torch_geometric.nn import GATConv
    conv = GATConv(F_in, F_out, heads=1, negative_slope=0.2, add_self_loops=True, bias=False).double()
    with torch.no_grad():
        conv.lin.weight.copy_(Wt.T)
        conv.att_dst.copy_(torch.tensor(a_i).view(1, 1, -1))   # receiver i
        conv.att_src.copy_(torch.tensor(a_j).view(1, 1, -1))   # neighbour j
    ei = torch.tensor(np.stack(np.nonzero(A)), dtype=torch.long)
    pyg = conv(Xt, ei).detach().numpy()
    assert np.allclose(out, pyg)
    print("  (also matched torch_geometric GATConv)")
except ImportError:
    print("  torch_geometric not installed; used dense torch reference")
print("graph-attention-layer OK")
