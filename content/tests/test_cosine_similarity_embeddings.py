import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "cosine-similarity-embeddings.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
A, B = rng.normal(size=(5, 8)), rng.normal(size=(9, 8))
S = ns["cosine_matrix"](A, B)
ref = F.cosine_similarity(torch.tensor(A)[:, None], torch.tensor(B)[None], dim=-1)
assert np.allclose(S, ref.numpy())
assert np.all(np.abs(S) <= 1 + 1e-12)
# scale invariance
assert np.allclose(ns["cosine_matrix"](3.7 * A, 0.2 * B), S)
# unit vectors: ||a - b||^2 = 2 - 2 cos
An = A / np.linalg.norm(A, axis=1, keepdims=True)
Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
d2 = ((An[:, None] - Bn[None]) ** 2).sum(-1)
assert np.allclose(d2, 2 - 2 * S)
# self-similarity is 1, opposite direction is -1, zero vector is safe
assert np.allclose(np.diag(ns["cosine_matrix"](A, A)), 1)
assert np.allclose(ns["cosine_matrix"](A, -A).diagonal(), -1)
assert np.all(ns["cosine_matrix"](np.zeros((1, 8)), B) == 0)
# top-k neighbours vs brute force
q = rng.normal(size=8)
idx, sc = ns["top_k_neighbors"](q, B, 3)
brute = sorted(range(len(B)), key=lambda i: -np.dot(q, B[i]) / (np.linalg.norm(q) * np.linalg.norm(B[i])))
assert list(idx) == brute[:3] and np.all(np.diff(sc) <= 0)
print("cosine-similarity-embeddings OK")
