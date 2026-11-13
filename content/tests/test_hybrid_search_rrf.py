import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "hybrid-search-rrf.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
bm25, rrf = ns["bm25_scores"], ns["rrf"]

# hand-computed BM25 on a 3-document corpus
docs = [["cat", "sat", "mat"], ["cat", "cat", "dog", "ran"], ["dog", "ran"]]
k1, b = 1.5, 0.75
avgdl = 9 / 3
idf = np.log(1 + (3 - 2 + 0.5) / (2 + 0.5))  # "cat" in 2 docs
exp = []
for dd in docs:
    f = dd.count("cat")
    exp.append(idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(dd) / avgdl)))
got = bm25(["cat"], docs, k1, b)
assert np.allclose(got, exp) and got[2] == 0
# explicit number for doc 0: idf*2.5/(1+1.5*(0.25+0.75))
assert abs(got[0] - idf * 2.5 / 2.5) < 1e-12

# monotonic in tf with saturation, fixed length via padding
vals = []
for f in range(1, 12):
    d = ["x"] * f + ["pad"] * (12 - f)
    vals.append(bm25(["x"], [d, ["y"] * 12, ["z"] * 12])[0])
diffs = np.diff(vals)
assert (diffs > 0).all() and (np.diff(diffs) < 0).all()
assert vals[-1] < (k1 + 1) * np.log(1 + (3 - 1 + 0.5) / 1.5)

# RRF hand-computed
A, B = ["a", "b", "c"], ["b", "a", "d"]
res = dict(rrf([A, B], k=60))
assert abs(res["a"] - (1 / 61 + 1 / 62)) < 1e-15
assert abs(res["b"] - (1 / 62 + 1 / 61)) < 1e-15
assert abs(res["c"] - 1 / 63) < 1e-15 and abs(res["d"] - 1 / 63) < 1e-15
# tie handling deterministic: a/b tie broken by id, c/d likewise
assert [d for d, _ in rrf([A, B])] == ["a", "b", "c", "d"]
assert [d for d, _ in rrf([B, A])] == ["a", "b", "c", "d"]

# top-ranked in both lists wins
L1, L2 = ["x", "y", "z"], ["x", "z", "y"]
assert rrf([L1, L2])[0][0] == "x"

# invariance to monotonic transforms of the scores
rng = np.random.default_rng(0)
s1, s2 = rng.normal(size=20), rng.normal(size=20)
def ranks(s):
    return [int(i) for i in np.argsort(-s, kind="stable")]
base = rrf([ranks(s1), ranks(s2)])
t1, t2 = np.exp(3 * s1) + 5, 100 * s2 - 7
assert base == rrf([ranks(t1), ranks(t2)])
print("hybrid-search-rrf OK")
