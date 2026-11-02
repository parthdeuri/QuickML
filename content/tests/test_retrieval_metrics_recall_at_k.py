import numpy as np, yaml
from pathlib import Path
from sklearn.metrics import ndcg_score

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "retrieval-metrics-recall-at-k.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
recall_at_k, mrr, ndcg_at_k = ns["recall_at_k"], ns["mrr"], ns["ndcg_at_k"]

rng = np.random.default_rng(0)
Q, D = 30, 20
scores = rng.normal(size=(Q, D))               # continuous => no ties
rel = (rng.random((Q, D)) < 0.2).astype(float) * rng.integers(1, 4, size=(Q, D))
rel[rel.sum(1) == 0, 0] = 1                    # every query has a relevant doc
for k in (1, 3, 5, 10, D):
    mine = ndcg_at_k(scores, rel, k)
    ref = ndcg_score(rel, scores, k=k, ignore_ties=True)    # linear gain
    assert abs(mine - ref) < 1e-12, (k, mine, ref)

# Recall@k and MRR against explicit per-query loops
for k in (1, 3, 10):
    vals = []
    for q in range(Q):
        top = np.argsort(-scores[q])[:k]
        vals.append(len(set(top) & set(np.flatnonzero(rel[q] > 0))) / (rel[q] > 0).sum())
    assert abs(recall_at_k(scores, rel, k) - np.mean(vals)) < 1e-12
rr = []
for q in range(Q):
    order = np.argsort(-scores[q])
    rr.append(1 / (1 + next(i for i, d in enumerate(order) if rel[q, d] > 0)))
assert abs(mrr(scores, rel) - np.mean(rr)) < 1e-12

# hand-checked: ranking puts relevant docs at ranks 2 and 4 (binary labels)
s = np.array([[4.0, 3.0, 2.0, 1.0]])
r = np.array([[0.0, 1.0, 0.0, 1.0]])
assert recall_at_k(s, r, 1) == 0.0 and recall_at_k(s, r, 2) == 0.5 and recall_at_k(s, r, 4) == 1.0
assert mrr(s, r) == 0.5
dcg = 1 / np.log2(3) + 1 / np.log2(5)
idcg = 1 / np.log2(2) + 1 / np.log2(3)
assert abs(ndcg_at_k(s, r, 4) - dcg / idcg) < 1e-12
# perfect ordering gives nDCG 1 and MRR 1
perfect = np.array([[1.0, 2.0, 0.0, 0.0]])
assert abs(ndcg_at_k(np.array([[3.0, 4.0, 1.0, 0.0]]), perfect, 3) - 1) < 1e-12
assert mrr(np.array([[3.0, 4.0, 1.0, 0.0]]), perfect) == 1.0
print("retrieval-metrics-recall-at-k OK")
