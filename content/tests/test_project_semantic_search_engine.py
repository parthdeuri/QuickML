import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "project-semantic-search-engine.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
SearchEngine = ns["SearchEngine"]

DOCS = {
    "d1": "error code zx9 means the pump has overheated",
    "d2": "the pump makes noise when running hot",
    "d3": "how to restart the pump after a failure",
    "d4": "warranty terms for the heating unit",
    "d5": "the pump is overheated if the fan stops",
    "d6": "installing the pump in a cold room",
}

def build(dim=12):
    e = SearchEngine(dim=dim)
    for i, t in DOCS.items():
        e.add(i, t, {"lang": "fr" if i in ("d3", "d4") else "en"})
    return e

# --- metrics hand-checked with scripted rankings -------------------
class Scripted(SearchEngine):
    RANK = {"q1": ["b", "a", "c"], "q2": ["x", "z", "w"],
            "q3": ["p", "q", "r"]}
    def search(self, q, k=3, filters=None, mode="hybrid"):
        return self.RANK[q][:k]

m = Scripted().evaluate([("q1", {"a"}), ("q2", {"x", "y"}),
                         ("q3", {"m"})], k=3)
# recall: 1, 1/2, 0 -> 0.5 ; rr: 1/2, 1, 0 -> 0.5
assert abs(m["recall@k"] - 0.5) < 1e-12 and abs(m["mrr"] - 0.5) < 1e-12
m1 = Scripted().evaluate([("q1", {"a"}), ("q2", {"x", "y"})], k=1)
# k=1: recall 0 and 1/2 -> 0.25 ; rr 0 and 1 -> 0.5
assert abs(m1["recall@k"] - 0.25) < 1e-12 and abs(m1["mrr"] - 0.5) < 1e-12

# --- hybrid rescues a query the dense signal gets wrong ------------
e = build()
q = "zx9 pump"
assert e.search(q, 1, mode="dense") != ["d1"]
assert e.search(q, 1, mode="lex") == ["d1"]
assert e.search(q, 1, mode="hybrid") == ["d1"]
labelled = [(q, {"d1"}), ("noise when hot", {"d2"}),
            ("cold room install", {"d6"})]
dense = e.evaluate(labelled, k=1, mode="dense")
hyb = e.evaluate(labelled, k=1, mode="hybrid")
assert hyb["mrr"] > dense["mrr"], (hyb, dense)
assert hyb["recall@k"] >= dense["recall@k"]

# --- filters are always respected, in every mode --------------------
for mode in ("dense", "lex", "hybrid"):
    for query in ("pump", "zx9 pump", "warranty heating", "restart"):
        got = e.search(query, 6, {"lang": "fr"}, mode)
        assert set(got) <= {"d3", "d4"}, (mode, query, got)
assert e.search("zx9 pump", 3, {"lang": "none"}) == []

# --- deleted documents never appear ---------------------------------
e.delete("d1")
for mode in ("dense", "lex", "hybrid"):
    for query in ("zx9 pump", "error code overheated", "pump"):
        assert "d1" not in e.search(query, 10, mode=mode)
e.delete("nonexistent")  # harmless

# --- adding documents changes results consistently ------------------
e = build()
before = e.search("pump", 6, mode="lex")
e.add("d7", "gardening tips for spring", {"lang": "en"})
after = e.search("pump", 6, mode="lex")
assert after == before  # unrelated document does not disturb order
e.add("d8", "pump pump zx9 pump manual", {"lang": "en"})
assert set(e.search("zx9 pump", 2, mode="lex")) == {"d1", "d8"}
assert e.search("manual", 1, mode="lex") == ["d8"]
e.delete("d8"); e.delete("d7")
assert e.search("pump", 6, mode="lex") == before
assert e.search("zx9 pump", 1) == ["d1"]
print("project-semantic-search-engine OK")
