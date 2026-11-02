import numpy as np, yaml
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "rag-dense-retrieval.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
chunk, embed, retrieve = ns["chunk"], ns["embed"], ns["retrieve"]

rng = np.random.default_rng(0)
vocab = [f"tok{i}" for i in range(300)]
doc = list(rng.choice(vocab, size=500))

# --- chunking: coverage, overlap, size, and an exact small case
for size, overlap in ((8, 2), (10, 0), (6, 3)):
    cs = chunk(doc, size, overlap)
    step = size - overlap
    assert all(len(c) <= size for c in cs) and len(cs[0]) == size
    assert [c for c in cs[0]] == doc[:size]
    for a, b in zip(cs, cs[1:]):
        assert a[step:] == b[:overlap] if overlap else True      # overlap words repeat
    assert cs[-1][-1] == doc[-1]                                  # tail covered
    assert (len(cs) - 1) * step + size >= len(doc)                # no word dropped
assert chunk(list("abcdefghij"), 4, 1) == [list("abcd"), list("defg"), list("ghij")]
assert chunk(list("abc"), 8, 2) == [list("abc")]

# --- embeddings are deterministic and unit length
assert np.allclose(np.linalg.norm(embed(doc[:8])), 1)
assert np.array_equal(embed(doc[:8]), embed(doc[:8]))

# --- top-k vs brute force (explicit loop with cosine) and sklearn NearestNeighbors
chunks = chunk(doc, 8, 2)
query = list(rng.choice(vocab, size=6)) + chunks[17][:4]
k = 5
top, sims = retrieve(query, chunks, k)

q = embed(query)
brute = []
for c in chunks:
    v = embed(c)
    brute.append(float(v @ q / (np.linalg.norm(v) * np.linalg.norm(q))))
brute = np.array(brute)
assert np.array_equal(top, np.argsort(-brute)[:k])
assert np.allclose(sims, np.sort(brute)[::-1][:k])

E = np.stack([embed(c) for c in chunks])
nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(E)
dist, idx = nn.kneighbors(q[None, :])
assert np.array_equal(top, idx[0])
assert np.allclose(sims, 1 - dist[0])

# --- a chunk used verbatim as the query is its own nearest neighbour
t, s = retrieve(chunks[40], chunks, 1)
assert t[0] == 40 and abs(s[0] - 1) < 1e-12
print("rag-dense-retrieval OK")
