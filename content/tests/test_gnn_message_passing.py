import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "gnn-message-passing.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
torch.manual_seed(0)
rng = np.random.default_rng(0)
N, F_in, F_out = 7, 5, 4
A = np.triu((rng.random((N, N)) < 0.4).astype(float), 1)
A = A + A.T                                   # symmetric, no self loops
X = rng.normal(size=(N, F_in))
W = rng.normal(size=(F_in, F_out))

out = ns["gcn_layer"](A, X, W)

# 1) dense torch reference D^-1/2 (A+I) D^-1/2 X W
At = torch.tensor(A) + torch.eye(N, dtype=torch.float64)
dinv = At.sum(1).pow(-0.5)
ref = torch.relu(dinv[:, None] * At * dinv[None, :] @ torch.tensor(X) @ torch.tensor(W))
assert np.allclose(out, ref.numpy())

# 2) per-node message-passing form (explicit loops over neighbours)
deg = A.sum(1) + 1
for i in range(N):
    nb = list(np.nonzero(A[i])[0]) + [i]
    m = sum((X[j] @ W) / np.sqrt(deg[i] * deg[j]) for j in nb)
    assert np.allclose(out[i], np.maximum(0, m))

# 3) torch_geometric GCNConv if installed
try:
    from torch_geometric.nn import GCNConv
    conv = GCNConv(F_in, F_out, bias=False).double()
    with torch.no_grad():
        conv.lin.weight.copy_(torch.tensor(W.T))
    ei = torch.tensor(np.stack(np.nonzero(A)), dtype=torch.long)
    pyg = torch.relu(conv(torch.tensor(X), ei)).detach().numpy()
    assert np.allclose(out, pyg)
    print("  (also matched torch_geometric GCNConv)")
except ImportError:
    print("  torch_geometric not installed; used dense torch + loop reference")

# two layers run and keep shape
W2 = rng.normal(size=(F_out, 3))
assert ns["gcn_two_layers"](A, X, W, W2).shape == (N, 3)
print("gnn-message-passing OK")
