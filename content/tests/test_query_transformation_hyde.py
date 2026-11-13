import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "query-transformation-hyde.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
embed, top_k, hyde_retrieve = ns["embed"], ns["top_k"], ns["hyde_retrieve"]
rrf, multi = ns["rrf"], ns["multi_query_retrieve"]

corpus = [
    "why is the grass green in spring and why is the lawn wet",
    "why is the sea salty and the lake fresh",
    "rayleigh scattering of sunlight by air molecules makes the sky look blue",
    "the stock market fell on monday as bond yields rose",
    "how is the bread made and why is the dough sticky",
]
V = np.stack([embed(c) for c in corpus])
TARGET = 2

# determinism and unit norm
assert np.array_equal(embed("a b c"), embed("a b c"))
assert abs(np.linalg.norm(embed("a b c")) - 1) < 1e-12
assert not np.array_equal(embed("a b c", seed=1), embed("a b c", seed=0))

q = "why is the sky blue"
assert top_k(embed(q), V, 1)[0] != TARGET  # plain query misses in top-1
gen = lambda s: "sunlight scattering by air molecules makes the sky look blue"
assert hyde_retrieve(q, V, gen, 1)[0] == TARGET  # HyDE finds it
assert hyde_retrieve(q, V, gen, 3) == hyde_retrieve(q, V, gen, 3)
# the generator really is what drives retrieval
bad = lambda s: "the stock market fell as bond yields rose"
assert hyde_retrieve(q, V, bad, 1)[0] == 3

# RRF equals hand-computed values (k=60)
out = dict(rrf([[0, 1], [1, 2]]))
assert abs(out[0] - 1 / 61) < 1e-15
assert abs(out[1] - (1 / 62 + 1 / 61)) < 1e-15
assert abs(out[2] - 1 / 62) < 1e-15
assert [d for d, _ in rrf([[0, 1], [1, 2]])] == [1, 0, 2]
out = dict(rrf([["a", "b"], ["b", "a"]], rrf_k=1))
assert abs(out["a"] - (1 / 2 + 1 / 3)) < 1e-15 and out["a"] == out["b"]

# multi-query contains documents found by any paraphrase
qs = [q, "sunlight scattering sky blue molecules", "stock market bond yields"]
per = [top_k(embed(s), V, 3) for s in qs]
fused = multi(qs, V, 5)
ids = [d for d, _ in fused]
assert set(ids) == set().union(*per), (ids, per)
assert TARGET in ids and 3 in ids
assert fused == multi(qs, V, 5)
scores = [s for _, s in fused]
assert scores == sorted(scores, reverse=True)
print("query-transformation-hyde OK")
