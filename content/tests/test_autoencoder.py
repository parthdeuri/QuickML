import numpy as np, yaml, torch
from pathlib import Path
from sklearn.decomposition import PCA

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "autoencoder.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
ident, one = (lambda z: z), (lambda h: 1.0)
tanh, dtanh = np.tanh, (lambda h: 1 - h ** 2)

rng = np.random.default_rng(1)
Z = rng.normal(size=(300, 6)) * np.array([3, 2, 1, .3, .2, .1])
X = Z @ rng.normal(size=(6, 6))
X = X - X.mean(0)

# 1) linear autoencoder reaches PCA's reconstruction error (k = 2)
P = ns["train_ae"](X, 2, ident, one)
ae_err = ((ns["reconstruct"](P, X, ident) - X) ** 2).sum(1).mean()
pca = PCA(2).fit(X)
pca_err = ((pca.inverse_transform(pca.transform(X)) - X) ** 2).sum(1).mean()
assert ae_err >= pca_err - 1e-9
assert abs(ae_err - pca_err) / pca_err < 1e-3, (ae_err, pca_err)
# the decoder's row space matches the principal subspace
W2 = P[2]
Q = np.linalg.qr(W2.T)[0]
proj = pca.components_.T @ pca.components_
assert np.allclose(Q @ Q.T, proj, atol=1e-2)

# 2) one tanh gradient step matches torch autograd
lr, k = 0.1, 3
W1, b1, W2, b2 = ns["train_ae"](X, k, tanh, dtanh, lr=lr, steps=1)
r = np.random.default_rng(0)
d = X.shape[1]
init = [r.normal(0, 0.1, (d, k)), np.zeros(k), r.normal(0, 0.1, (k, d)), np.zeros(d)]
ts = [torch.tensor(a, requires_grad=True) for a in init]
Xt = torch.tensor(X)
out = torch.tanh(Xt @ ts[0] + ts[1]) @ ts[2] + ts[3]
loss = ((out - Xt) ** 2).sum() / (2 * len(X))
loss.backward()
for new, t, a in zip((W1, b1, W2, b2), ts, init):
    assert np.allclose(new, a - lr * t.grad.numpy())

# 3) training lowers the nonlinear loss
def err(P):
    return ((ns["reconstruct"](P, X, tanh) - X) ** 2).sum(1).mean()
assert err(ns["train_ae"](X, 3, tanh, dtanh, lr=0.01, steps=3000)) < err(ns["train_ae"](X, 3, tanh, dtanh, steps=1))
print("autoencoder OK")
