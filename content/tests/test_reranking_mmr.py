import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "reranking-mmr.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
mmr, rerank, diversity = ns["mmr"], ns["rerank"], ns["diversity"]


def unit(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def brute(q, D, k, lam):
    chosen = []
    for _ in range(k):
        best, best_s = None, -1e18
        for i in range(len(D)):
            if i in chosen:
                continue
            red = max([float(D[i] @ D[j]) for j in chosen] or [0.0])
            s = lam * float(D[i] @ q) - (1 - lam) * red
            if s > best_s:
                best, best_s = i, s
        chosen.append(best)
    return chosen


rng = np.random.default_rng(0)
for trial in range(20):
    D = unit(rng.normal(size=(30, 8)))
    q = unit(rng.normal(size=(1, 8)))[0]
    for lam in (0.0, 0.3, 0.5, 0.8, 1.0):
        got = mmr(q, D, 6, lam)
        assert got == brute(q, D, 6, lam), (trial, lam)
        assert len(set(got)) == len(got)
    # lam = 1 is plain top-k by similarity
    assert mmr(q, D, 6, 1.0) == list(np.argsort(-(D @ q), kind="stable")[:6])
    # lam = 0: after the first pick, always the least similar to the chosen set
    sel = mmr(q, D, 5, 0.0)
    for t in range(1, 5):
        rest = [i for i in range(len(D)) if i not in sel[:t]]
        red = {i: max(D[i] @ D[j] for j in sel[:t]) for i in rest}
        assert red[sel[t]] == min(red.values())
# k larger than n returns every item once
assert sorted(mmr(q, D[:4], 10, 0.5)) == [0, 1, 2, 3]

# rerank: stable sort by injected score, returns top_n
cands = ["a", "bb", "cc", "d", "eee", "ff"]
out = rerank(cands, lambda s: len(s), 4)
assert out == ["eee", "bb", "cc", "ff"], out  # ties keep input order
assert rerank(cands, len, 100) == ["eee", "bb", "cc", "ff", "a", "d"]
assert cands == ["a", "bb", "cc", "d", "eee", "ff"]  # input untouched

# redundant set: 4 near-copies of the query direction plus 4 spread out
q = np.array([1.0, 0.0, 0.0, 0.0])
base = [q + 0.02 * rng.normal(size=4) for _ in range(4)]
spread = [np.eye(4)[1] + 0.9 * q, np.eye(4)[2] + 0.9 * q, np.eye(4)[3] + 0.9 * q, np.eye(4)[1] - np.eye(4)[2] + 0.9 * q]
D = unit(np.array(base + spread))
plain = list(np.argsort(-(D @ q), kind="stable")[:4])
div_plain = diversity(D, plain)
div_mmr = diversity(D, mmr(q, D, 4, 0.5))
assert div_mmr <= div_plain and div_mmr < 0.9 * div_plain, (div_mmr, div_plain)
# diversity of two orthogonal vectors is 0, of identical ones is 1
assert abs(diversity(np.eye(3), [0, 1])) < 1e-12
assert abs(diversity(np.ones((2, 3)) / np.sqrt(3), [0, 1]) - 1) < 1e-12
print("reranking-mmr OK")
