"""Smoke test: every family / compare code sketch runs and gives sane output."""
from pathlib import Path

import numpy as np
import yaml

PAT = Path(__file__).resolve().parent.parent / "patterns"
rng = np.random.default_rng(0)


def load(pid):
    d = yaml.safe_load((PAT / f"{pid}.yaml").read_text(encoding="utf-8"))
    ns = {}
    exec(d["template"], ns)
    return ns


# every template executes (defines its functions)
ids = sorted(p.stem for p in PAT.glob("*.yaml"))
for pid in ids:
    load(pid)

X = rng.normal(size=(20, 4))
y = rng.integers(0, 2, 20)

ns = load("compare-batchnorm-vs-layernorm")
b = ns["batch_norm"](X)
assert np.allclose(b.mean(0), 0, atol=1e-8) and np.allclose(b.var(0), 1, atol=1e-3)
l = ns["layer_norm"](X)
assert np.allclose(l.mean(1), 0, atol=1e-8)

ns = load("compare-pr-vs-roc")
s = rng.random(20)
t, f = ns["tpr_fpr"](y, s, 0.5)
p, r = ns["precision_recall"](y, s, 0.5)
assert 0 <= t <= 1 and 0 <= f <= 1 and 0 <= p <= 1 and np.isclose(r, t)

ns = load("compare-kmeans-vs-gmm")
C = X[:3]
assert ns["kmeans_assign"](X, C).shape == (20,)
r = ns["gmm_responsibilities"](rng.normal(size=(20, 3)))
assert np.allclose(r.sum(1), 1)

ns = load("compare-kmeans-vs-dbscan")
assert ns["kmeans_labels"](X, C).shape == (20,)
assert ns["is_core"](X, 100.0, 5).all()

ns = load("compare-lstm-vs-attention")
h = ns["rnn_step"](np.zeros(3), X[0], np.eye(3), rng.normal(size=(4, 3)), np.zeros(3))
assert h.shape == (3,) and np.all(np.abs(h) <= 1)
out = ns["attend"](X, X, X)
assert out.shape == X.shape

ns = load("compare-vae-vs-gan")
loss = ns["vae_loss"](X, X, np.zeros_like(X), np.zeros_like(X))
assert np.isclose(loss, 0.0)  # perfect reconstruction, KL to N(0,1) is 0
assert ns["gan_d_loss"](np.full(5, 0.5), np.full(5, 0.5)) > 0

ns = load("compare-mse-vs-cross-entropy")
assert np.isclose(ns["mse"](np.ones(3), np.ones(3)), 0)
lg = rng.normal(size=(6, 3))
yy = rng.integers(0, 3, 6)
import torch
import torch.nn.functional as F
ref = F.cross_entropy(torch.tensor(lg), torch.tensor(yy)).item()
assert np.isclose(ns["cross_entropy"](lg, yy), ref)

ns = load("compare-momentum-vs-adam")
w, v = ns["momentum_step"](np.ones(3), np.ones(3), np.zeros(3))
assert np.allclose(w, 0.9)
w, m, s2 = ns["adam_step"](np.ones(3), np.ones(3), np.zeros(3), np.zeros(3), 1)
assert np.allclose(w, 1 - 1e-3, atol=1e-6)  # first Adam step is about lr

ns = load("compare-pca-vs-autoencoder")
assert ns["pca_encode"](X, 2).shape == (20, 2)

ns = load("compare-ridge-vs-lasso")
w = ns["ridge_fit"](X, X @ np.ones(4), 1e-9)
assert np.allclose(w, 1, atol=1e-4)

ns = load("compare-forest-vs-boosting")
trees = [lambda Z: np.ones(len(Z)), lambda Z: np.zeros(len(Z))]
assert np.allclose(ns["forest_predict"](trees, X), 0.5)
assert np.allclose(ns["boosting_predict"](trees, X, 0.1), 0.1)

ns = load("compare-naive-bayes-vs-logistic")
assert np.allclose(ns["lr_probs"](np.zeros(4), 0.0, X), 0.5)

ns = load("transfer-and-adaptation")
xx = rng.normal(size=(2, 5))
W = rng.normal(size=(5, 4))
A, B = rng.normal(size=(5, 2)), np.zeros((2, 4))
assert np.allclose(ns["lora_forward"](xx, W, A, B, 8, 2), xx @ W)
tl = rng.normal(size=(3, 6))
assert np.isclose(ns["distill_loss"](tl, tl), 0.0)

ns = load("generative-models")
z = ns["reparameterise"](np.zeros((2, 3)), np.zeros((2, 3)), np.random.default_rng(1))
assert z.shape == (2, 3)

print("family templates OK:", len(ids), "files")
